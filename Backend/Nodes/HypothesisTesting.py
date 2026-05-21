from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from Nodes.DataStructs import AnalystState, MultiQuery, RouterOutput, CodeOutput
from Nodes.Prompts import MULTIOUERY_PROMPT, ROUTER_PROMPT, CODE_GEN_PROMPT, INITIAL_TITLE_PROMPT, FALLBACK_TITLE_PROMPT, SUMMARY_PROMPT


class AgenticAnalystNode:
    def __init__(self, llm_ollama, llm_gemini, vector_db, sandbox, pipe, thread_id):
        self.llm_ollama = llm_ollama
        self.llm_gemini = llm_gemini
        self.vector_db = vector_db
        self.sandbox = sandbox
        self.pipe = pipe
        self.thread_id = thread_id

    def build(self):
        builder = StateGraph(AnalystState)
        builder.add_node("retrieval_node", self.retrievalNode)
        builder.add_node("router_node", self.routerNode)
        builder.add_node("code_gen_node", self.codeGeneratorNode)
        builder.add_node("executor_node", self.executorNode)
        builder.add_node("final_gen_node", self.finalGeneratorNode)

        builder.add_edge(START, "retrieval_node")
        builder.add_edge("retrieval_node", "router_node")
        builder.add_conditional_edges("router_node", lambda s: s.route, {
            "CODE": "code_gen_node",
            "RAG": "final_gen_node"
        })
        builder.add_edge("code_gen_node", "executor_node")
        builder.add_conditional_edges("executor_node",
                                      lambda s: "code_gen_node" if s.errors and s.iterations < 5 else "final_gen_node"
                                      )
        builder.add_edge("final_gen_node", END)

        return builder.compile(checkpointer=MemorySaver())

    def retrievalNode(self, state: AnalystState):
        prompt_text = MULTIOUERY_PROMPT.format(user_input=state.user_input)
        llm_mq = self.llm_ollama.with_structured_output(MultiQuery)
        queries = llm_mq.invoke([SystemMessage(content=prompt_text)]).queries

        fused_scores = {}
        k = 60
        for q in queries:
            docs = self.vector_db.similarity_search(q)
            for rank, doc in enumerate(docs):
                content = doc.page_content
                fused_scores[content] = fused_scores.get(content, 0) + 1 / (k + rank + 1)

        reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return {"documents": [doc for doc, score in reranked[:5]]}

    def routerNode(self, state: AnalystState):
        prompt_text = ROUTER_PROMPT.format(user_input=state.user_input)
        llm_router = self.llm_ollama.with_structured_output(RouterOutput)
        response = llm_router.invoke([SystemMessage(content=prompt_text)])
        return {"route": response.route}

    async def codeGeneratorNode(self, state: AnalystState):
        feedback_text = f"FIX THIS ERROR: {state.errors}" if state.errors else ""
        prompt_text = CODE_GEN_PROMPT.format(
            user_input=state.user_input,
            documents=state.documents,
            feedback=feedback_text
        )

        llm_gen = self.llm_gemini.with_structured_output(CodeOutput)
        for _ in range(5):
            try:
                response = await llm_gen.ainvoke([HumanMessage(content=prompt_text)])
                break
            except Exception:
                continue
        return {"code": response.code, "iterations": state.iterations + 1}

    async def executorNode(self, state: AnalystState):
        res = await self.sandbox.execute(thread_id=self.thread_id, code=state.code)
        if res["exit_code"] != 0 or res["stderr"]:
            return {"errors": res["stderr"]}
        return {
            "result": res["stdout"],
            "errors": "",
            "generated_files": res["generated_files_minio"]
        }

    async def finalGeneratorNode(self, state: AnalystState):
        title_prompt = INITIAL_TITLE_PROMPT.format(user_input=state.user_input)
        title = "Analysis_Report"

        for _ in range(3):
            title_res = await self.llm_ollama.ainvoke([SystemMessage(content=title_prompt)])
            raw_title = title_res.content.strip().replace('"', '')
            if len(raw_title.split()) <= 10:
                title = raw_title
                break
            else:
                title_prompt = FALLBACK_TITLE_PROMPT.format(user_input=state.user_input)

        content = state.result if not state.errors else f"Technical Error: {state.errors}"
        rag_context = ", ".join(state.documents)

        summary_prompt = SUMMARY_PROMPT.format(
            rag_context=rag_context,
            code=state.code,
            content=content,
            user_input=state.user_input
        )

        summary_res = await self.llm_ollama.ainvoke([SystemMessage(content=summary_prompt)])
        state.final_summary = summary_res.content
        state.title = title
        return state


async def runAgenticAnalysisNode(analyst_node, config, manager, db_connector, state):
    thread_id = config["metadata"]["thread_id"]
    await manager.push("Hi, I have performed all preliminary analysis and tests. What can I help you with?", thread_id)
    all_data = db_connector.get_insight_titles(thread_id)
    await manager.push(all_data, thread_id, 'insight_info')
    state.current_data_info.page_data[4] = []

    while True:
        user_query = await manager.pull(thread_id)
        user_query = user_query["user_input"]
        if not user_query:
            continue

        inputs = {
            "user_input": user_query, "data_path": state.current_data_info.file_path, "final_summary": "",
            "code": "", "result": "", "documents": [], "route": "", "title": "", "subdir_path": "",
            "generated_files": [], "errors": "", "iterations": 0
        }
        await manager.push("Starting analysis...", thread_id, "process_update")
        final_state = {}

        async for event in analyst_node.astream(inputs, config=config, stream_mode="updates"):
            for node_name, output in event.items():
                if node_name == "retrieval_node":
                    await manager.push("Retrieving relevant documents...", thread_id, "process_update")
                elif node_name == "code_gen_node":
                    await manager.push("Generating statistical models...", thread_id, "process_update")
                elif node_name == "executor_node":
                    if output.get("errors"):
                        await manager.push("Error detected. Retrying code generation...", thread_id, "process_update")
                    else:
                        await manager.push("Code executed successfully. Reviewing results...", thread_id,
                                           "process_update")
                elif node_name == "final_gen_node":
                    final_state = output

        if final_state:
            await manager.push(final_state['final_summary'], thread_id, "llm_message")
            title = final_state.get("title", "Analysis_Report")
            try:
                await db_connector.add_insight(
                    thread_id=thread_id,
                    title=title,
                    chart_paths=final_state.get("generated_files", []),
                    code=final_state.get("code", ""),
                    code_output=final_state.get("result", ""),
                    summary=final_state.get("final_summary", ""),
                    documents_retrieved=final_state.get("documents", [])
                )
            except Exception:
                pass
            all_data.append(title)

        await manager.push("", thread_id, "process_update")
        await manager.push(all_data, thread_id, 'insight_info')
        state.current_data_info.page_data[4] = all_data
    return state