from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from Nodes.DataStructs import DataAnalysis, NullCleanupInfo, NullRouterPromptOutput, NullParserOutput
from Nodes.Prompts import pre_re_prompt_null, router_prompt_null, conversational_prompt_null, parser_prompt_null, post_prompt_null
import pandas as pd
import numpy as np
import copy


class SubgraphState(BaseModel):
    current_info: NullCleanupInfo = NullCleanupInfo()
    current_data_info: DataAnalysis = DataAnalysis()
    log: str = ""
    user_input: str = ""
    route: str = ""


class NullRemovalNode:
    def __init__(self, llm, manager, thread_id, minio_client):
        self.llm = llm
        self.manager = manager
        self.thread_id = thread_id
        self.minio_client = minio_client

        self.null_values = [
            "None", "nan", "NaN", "NAN", "null", "NULL", "undefined", "Undefined",
            "n/a", "N/A", "na", "NA", "n.a.", "N.A.", "n/p", "not available", "not applicable",
            "?", "-", "--", "---", ".", "...", "missing", "Missing", "MISSING", "unknown", "Unknown",
            "", " ", "  ", "\t", "\n", "none",
            -1, -99, -999, -9999, 0, 999, 9999, "999", "9999", "0", "-1",
            "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
            "0000-00-00", "01-01-1970", "1900-01-01", "NaT", "nat"
        ]
        self.data = {"search_config": {"default_nulls": self.null_values, "custom_nulls": []}}

    def build(self):
        builder = StateGraph(SubgraphState)
        builder.add_node("prompt_node", self.prePromptNode)
        builder.add_node("input_node", self.takeInput)
        builder.add_node("router_node", self.routerNode)
        builder.add_node("parse_node", self.parseInputNode)
        builder.add_node("main_node", self.nullNode)
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
        builder.add_conditional_edges("convo_node", lambda s: "post_prompt_node" if s.log == "Post Prompted" else "prompt_node")
        builder.add_edge("parse_node", "main_node")
        builder.add_conditional_edges("main_node", lambda s: "post_prompt_node" if "Successfully" in s.log else "prompt_node")
        builder.add_edge("post_prompt_node", "input_node")

        return builder.compile(checkpointer=MemorySaver())

    async def prePromptNode(self, state: SubgraphState):
        print("PREPROMPT")
        sys_content = pre_re_prompt_null.format(errors=state.log, inputs=state.current_info.nulls)
        response = await self.llm.ainvoke([SystemMessage(content=sys_content)])
        await self.manager.push(response.content, self.thread_id)
        state.log = ""
        return state

    @staticmethod
    def takeInput(state: SubgraphState):
        state_update = interrupt(None)
        return state_update

    def routerNode(self, state: SubgraphState):
        print("ROUTER")
        llm_router = self.llm.with_structured_output(NullRouterPromptOutput)
        prompt = router_prompt_null.format(user_input=state.user_input)
        response = llm_router.invoke([SystemMessage(content=prompt)]).route
        if response == "APPROVE" and state.log == "Post Prompted":
            return {"route": "END"}
        elif response == "APPROVE":
            return {"route": "PARSE"}

        return {"route": response}

    async def conversationNode(self, state: SubgraphState):
        print("CONVO")
        sys_content = conversational_prompt_null.format(
            current_info=state.current_info.model_dump_json(),
            defaults=self.null_values,
            user_input=state.user_input
        )
        response = await self.llm.ainvoke([SystemMessage(content=sys_content)])
        await self.manager.push(response.content, self.thread_id)

    def parseInputNode(self, state: SubgraphState):
        print("PARSE")
        llm_parser = self.llm.with_structured_output(NullParserOutput)
        prompt_content = parser_prompt_null.format(
            current_info=state.current_info.nulls,
            user_input=state.user_input
        )

        data = llm_parser.invoke([SystemMessage(content=prompt_content)])

        state.current_info.nulls = data.nulls
        self.data["search_config"]["custom_nulls"] = state.current_info.nulls
        return state

    async def postPromptNode(self, state: SubgraphState):
        print("POSTPROMPT")
        sys_content = post_prompt_null.format(
            logs=state.log,
            custom_nulls=state.current_info.nulls,
            defaults=self.null_values
        )
        response = await self.llm.ainvoke([SystemMessage(content=sys_content)])
        await self.manager.push(response.content, self.thread_id)
        await self.manager.push({"null_analysis": self.data}, self.thread_id, "dashboard_data")
        state.current_data_info.page_data[1] = {"null_analysis": self.data, "col_analysis": []}
        return {"current_data_info": state.current_data_info, "log": "Post Prompted"}

    def nullNode(self, state: SubgraphState):
        null_values = list(set(self.null_values + state.current_info.nulls))
        main_df = self.minio_client.get_df(state.current_data_info.file_path, self.thread_id)

        # Keep an exact copy for reporting calculations before editing strings
        working_df = main_df.copy()
        self.data["discovery_results"] = []

        report = pd.DataFrame([
            {**{"Col": col}, **{null_val: (working_df[col] == null_val).sum() for null_val in null_values}}
            for col in working_df.columns
        ])

        new_info = copy.deepcopy(state.current_info)
        new_data_info = copy.deepcopy(state.current_data_info)

        count = 0
        for null in report.columns[1:]:
            total_sum = report[null].sum()
            if total_sum > 0:
                affected_cols = report[report[null] > 0]["Col"].tolist()
                self.data["discovery_results"].append({
                    "value": str(null),
                    "count": int(total_sum),
                    "affected_columns": affected_cols
                })

                count += 1
                txt = (
                    f"'{null}' identified as a disguised null, found {int(total_sum)} times in columns: "
                    f"{', '.join([str(i) for i in affected_cols])}"
                )
                new_info.nulls_report.append([txt, {"source": "nulls", "column": str(null)}, f"null_{count}"])

        # --- FIX: Safe Type-Aware Replacement Strategy to Bypass Pandas Internal Bug ---

        # 1. Split numeric-typed null markers and string-typed markers
        numeric_nulls = [x for x in null_values if isinstance(x, (int, float)) and not isinstance(x, bool)]
        string_nulls = [str(x) for x in null_values if not isinstance(x, (int, float)) or isinstance(x, bool)]

        # Remove empty strings/spaces from general list to handle them gracefully via regex
        whitespace_markers = {"", " ", "  ", "\t", "\n"}
        string_nulls = [s for s in string_nulls if s not in whitespace_markers]

        # 2. Apply replacements columns safely without block memory tracking collapse
        for col in working_df.columns:
            # Handle standard string/object columns
            if working_df[col].dtype == object or isinstance(working_df[col].dtype, pd.StringDtype):
                # Clean native trailing whitespaces/blanks safely
                working_df[col] = working_df[col].replace(r'^\s*$', np.nan, regex=True)

                # Replace exact string matches
                mask = working_df[col].astype(str).isin(string_nulls)
                if mask.any():
                    working_df.loc[mask, col] = np.nan

            # Handle numerical columns safely
            if len(numeric_nulls) > 0 and pd.api.types.is_numeric_dtype(working_df[col]):
                mask = working_df[col].isin(numeric_nulls)
                if mask.any():
                    working_df.loc[mask, col] = np.nan

        # Replace your old reference name back to the finalized dataframe wrapper
        main_df = working_df

        # --- Proceed with normal file saving code below ---
        self.minio_client.write_df(report, f"{self.thread_id}/NULL_NODE/Nulls.parquet", self.thread_id)
        self.minio_client.write_df(main_df, f"{self.thread_id}/NULL_NODE/Nulls_removed.parquet", self.thread_id)
        self.minio_client.write_df(main_df, f"{self.thread_id}/TYPE_VALIDATION/Processed.parquet", self.thread_id)

        new_data_info.nulls = [str(x) for x in null_values]
        log_msg = "Null report added Successfully based on current list of suspected disguised nulls"

        return SubgraphState(
            current_info=new_info,
            current_data_info=new_data_info,
            log=log_msg,
            user_input=state.user_input,
            route=state.route
        )


async def runNullRemoval(null_processor, config, manager, add_doc, state: SubgraphState):
    thread_id = config["metadata"]["thread_id"]
    thread_id_sub = f"{thread_id}_sub"
    config_sub = {"configurable": {"thread_id": thread_id_sub}}

    await null_processor.ainvoke({"current_info": NullCleanupInfo(), "current_data_info": state.current_data_info, "user_input": ""}, config_sub)

    while True:
        snapshot = null_processor.get_state(config_sub)

        if not snapshot.next:
            break

        user_input = await manager.pull(thread_id)

        await null_processor.ainvoke(Command(resume=user_input), config_sub)

    state.current_data_info = snapshot.values.get("current_data_info")

    for txt, metadata, doc_id in snapshot.values.get("current_info").nulls_report:
        add_doc(txt, metadata, doc_id)

    await manager.push(33, thread_id, "status_update")
    state.current_data_info.current_progress += 33
    return state