from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from Nodes.DataStructs import FileRouterPromptOutput, FileParserOutput, PathInformation, DataAnalysis
import Nodes.Prompts as prompts

class SubgraphState(BaseModel):
    current_info: PathInformation = Field(default_factory=PathInformation)
    current_data_info: DataAnalysis = Field(default_factory=DataAnalysis)
    log: str = ""
    user_input: str = ""
    route: str = ""

class FileTargetNode:
    def __init__(self, llm, manager, thread_id):
        self.llm = llm
        self.manager = manager
        self.thread_id = thread_id

    def build(self):
        builder = StateGraph(SubgraphState)
        builder.add_node("prompt_node", self.prePromptNode)
        builder.add_node("input_node", self.takeInput)
        builder.add_node("router_node", self.routerNode)
        builder.add_node("parse_node", self.parseInputNode)
        builder.add_node("main_node", self.loaderNode)
        builder.add_node("convo_node", self.conversationNode)
        builder.add_node("post_prompt_node", self.postPromptNode)

        builder.add_edge(START, "prompt_node")
        builder.add_edge("prompt_node", "input_node")
        builder.add_edge("input_node", "router_node")
        builder.add_conditional_edges("router_node", lambda s: s.route, {
            "PARSE": "parse_node",
            "CONVO": "convo_node",
            "END": END
        })
        builder.add_conditional_edges("convo_node",
                                      lambda s: "post_prompt_node" if s.log == "Post Prompted" else "prompt_node")
        builder.add_conditional_edges("parse_node",
                                      lambda s: "prompt_node" if s.current_info.missing_fields else "main_node")
        builder.add_conditional_edges("main_node",
                                      lambda s: "post_prompt_node" if "Successfully" in s.log else "prompt_node")
        builder.add_edge("post_prompt_node", "input_node")

        return builder.compile(checkpointer=MemorySaver())

    async def prePromptNode(self, state: SubgraphState):
        file_name = state.current_info.file_path.split("/")[-1] if state.current_info.file_path else "None"
        sys_content = prompts.pre_re_prompt_file.format(filepath=file_name, errors=state.log)

        response = await self.llm.ainvoke([SystemMessage(content=sys_content)])
        await self.manager.push("SYSTEM_READY: Pipeline initialized. Please upload your file.", self.thread_id)
        await self.manager.push(response.content, self.thread_id)

    @staticmethod
    def takeInput(state: SubgraphState):
        return interrupt(None)

    def routerNode(self, state: SubgraphState):
        prompt_text = prompts.router_prompt_file.format(user_input=state.user_input)
        llm_router = self.llm.with_structured_output(FileRouterPromptOutput)
        response = llm_router.invoke([SystemMessage(content=prompt_text)]).route

        if not state.current_info.file_path:
            route = "CONVO"
        elif response == "APPROVE" and state.log == "Post Prompted":
            route = "END"
        elif response in ("APPROVE", "COL"):
            route = "PARSE"
        elif response == "FILE":
            route = "CONVO"
        else:
            route = response

        return {"route": route}

    async def conversationNode(self, state: SubgraphState):
        sys_content = prompts.conversational_prompt_file.format(
            current_info=state.current_info.model_dump_json(),
            user_input=state.user_input
        )
        response = await self.llm.ainvoke([SystemMessage(content=sys_content)])
        await self.manager.push(response.content, self.thread_id)

    def parseInputNode(self, state: SubgraphState):
        llm_parser = self.llm.with_structured_output(FileParserOutput)
        prompt_content = prompts.parser_prompt_file.format(
            current_info=state.current_info.analysis_targets,
            user_input=state.user_input,
            cols=state.current_info.all_columns,
        )
        response = llm_parser.invoke([SystemMessage(content=prompt_content)]).analysisTargets

        try:
            updated_info = state.current_info.model_copy(deep=True)
            if response and "analysis_targets" in updated_info.missing_fields:
                updated_info.missing_fields.remove("analysis_targets")
            updated_info.analysis_targets = response
            return {"current_info": updated_info}
        except Exception as e:
            return {"log": f"Parsing error: {str(e)}"}

    async def postPromptNode(self, state: SubgraphState):
        file_name = state.current_info.file_path.split("/")[-1] if state.current_info.file_path else "None"
        sys_content = prompts.post_prompt_file.format(
            logs=state.log,
            file_path=file_name,
            targets=state.current_data_info.targets
        )
        response = await self.llm.ainvoke([SystemMessage(content=sys_content)])
        await self.manager.push(response.content, self.thread_id)
        return {"log": "Post Prompted"}

    async def loaderNode(self, state: SubgraphState):
        try:
            if not state.current_info.analysis_targets:
                return {}

            valid = [i for i in state.current_info.analysis_targets if i in state.current_info.all_columns]
            if len(valid) != len(state.current_info.analysis_targets):
                updated_info = state.current_info.model_copy(deep=True)
                updated_info.analysis_targets = valid
                if "analysis_targets" not in updated_info.missing_fields:
                    updated_info.missing_fields.append("analysis_targets")
                return {"current_info": updated_info, "log": "Some target columns were not found in the file."}

            updated_data = state.current_data_info.model_copy(deep=True)
            updated_data.file_path = state.current_info.file_path
            updated_data.targets = state.current_info.analysis_targets

            await self.manager.push({
                "rows": state.current_info.rows,
                "cols": len(state.current_info.all_columns),
                "targets": state.current_info.analysis_targets
            }, self.thread_id, "dashboard_data")

            return {"current_data_info": updated_data, "log": "Successfully loaded the data file"}
        except Exception as e:
            updated_info = state.current_info.model_copy(deep=True)
            updated_info.file_path = ""
            if "file_path" not in updated_info.missing_fields:
                updated_info.missing_fields.append("file_path")
            return {"current_info": updated_info, "log": str(e)}


async def runFileTarget(data_loader, config, manager, add_doc, state: SubgraphState):
    thread_id = config["metadata"]["thread_id"]
    thread_id_sub = f"{thread_id}_sub"
    config_sub = {"configurable": {"thread_id": thread_id_sub}}

    # Check if a runtime memory frame already exists before issuing fresh instantiation seeds
    existing_snapshot = data_loader.get_state(config_sub)
    if not existing_snapshot.values:
        from Nodes.DataStructs import PathInformation
        await data_loader.ainvoke({"current_info": PathInformation(), "user_input": ""}, config_sub)

    while True:
        snapshot = data_loader.get_state(config_sub)
        if not snapshot.next:
            break

        user_input = await manager.pull(thread_id)

        if user_input.get("current_info"):
            current_info = snapshot.values.get("current_info")
            missing_fields = list(getattr(current_info, "missing_fields", []))

            if user_input["current_info"].get("file_path"):
                if "file_path" in missing_fields:
                    missing_fields.remove("file_path")
            else:
                if "file_path" not in missing_fields:
                    missing_fields.append("file_path")

            user_input["current_info"]["missing_fields"] = missing_fields

        await data_loader.ainvoke(Command(resume=user_input), config_sub)

    final_snapshot = data_loader.get_state(config_sub)
    resolved_info = final_snapshot.values.get("current_info")
    resolved_data = final_snapshot.values.get("current_data_info")

    state.current_data_info = resolved_data
    state.current_data_info.page_data[0] = {
        "islocked": True,
        "file": state.current_data_info.file_path,
        "rows": resolved_info.rows,
        "cols": len(resolved_info.all_columns),
        "targets": state.current_data_info.targets
    }

    txt = (
        f"{state.current_data_info.file_path} loaded for analysis with target columns: "
        f"{', '.join(state.current_data_info.targets)}. The dataset has the following columns: "
        f"{', '.join(resolved_info.all_columns)}. It has a total of {resolved_info.rows} rows and "
        f"{len(resolved_info.all_columns)} columns."
    )
    add_doc(txt, {"source": "file metadata"}, "analysis_file")

    state.current_data_info.current_progress += 100
    await manager.push(100, thread_id, "status_update")
    await manager.push(1, thread_id, "page_change")
    return state