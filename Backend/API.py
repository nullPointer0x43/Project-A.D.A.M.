import io
import os
import anyio
import dotenv
import asyncio
import pandas as pd
from typing import Any
from fastapi import HTTPException
from fastapi.responses import Response
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from LLM import LLM_connector
from Minio import DataBaseMinio
from ChromaDB import Chroma_client
from Postgre import PostgreSQLConnector
from SandBox import SandBox
from Nodes.ColumnAnalysisNode import ColumnSmartParser, runColumnParsing
from Nodes.DataStructs import DataAnalysis
from Nodes.FileTargetsNode import FileTargetNode, runFileTarget
from Nodes.MissingnessAnalysisNode import runMissingnessNode
from Nodes.MultivariateAnalysisNode import runMultivariateNode
from Nodes.NullRemovalNode import NullRemovalNode, runNullRemoval
from Nodes.ReportNode import generate_report_pdf
from Nodes.UnivariateAnalysisNode import runUnivariateNode
from Nodes.HypothesisTesting import AgenticAnalystNode, runAgenticAnalysisNode

dotenv.load_dotenv()

APP_HOST = "0.0.0.0"
APP_PORT = 8000
ALLOWED_CORS_ORIGINS = ["*"]

MINIO_PARQUET_PROCESSED_PATH = "{thread_id}/MISSINGNESS_IMPUTED/Processed.parquet"
MINIO_PARQUET_ORIGINAL_PATH = "{thread_id}/ORIGINAL_CSV/{file_base_name}.parquet"


class GlobalState(BaseModel):
    current_info: Any = None
    current_data_info: DataAnalysis = DataAnalysis()
    log: str = ""
    user_input: str = ""


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, thread_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[thread_id] = websocket
        postgre_db.add_session(thread_id)

    def disconnect(self, thread_id: str):
        if thread_id in self.active_connections:
            del self.active_connections[thread_id]

    async def push(self, data, thread_id: str, msg_type="llm_message"):
        ws = self.active_connections.get(thread_id)
        if ws:
            try:
                await ws.send_json({"type": msg_type, "message": data})
            except Exception as e:
                print(f"WS push failed: {str(e)}")

    async def pull(self, thread_id: str):
        ws = self.active_connections.get(thread_id)
        if ws:
            data = await ws.receive_json()
            return data.get("content")
        print("Disconnected")
        return None


llm_server = LLM_connector()
postgre_db = PostgreSQLConnector()
minio_db = DataBaseMinio()
chroma_db = Chroma_client()
sandbox_connector = SandBox()
socket_manager = ConnectionManager()


async def file_node_runner(state: GlobalState, config: RunnableConfig):
    thread_id = config['metadata']['thread_id']
    data_loader_tool = FileTargetNode(llm=llm_server.llm_ollama, manager=socket_manager, thread_id=thread_id).build()
    return await runFileTarget(data_loader_tool, config, socket_manager, chroma_db.add_to_db, state)


async def null_node_runner(state: GlobalState, config: RunnableConfig):
    thread_id = config['metadata']['thread_id']
    null_removal_tool = NullRemovalNode(llm=llm_server.llm_ollama, manager=socket_manager, thread_id=thread_id,
                                        minio_client=minio_db).build()
    return await runNullRemoval(null_removal_tool, config, socket_manager, chroma_db.add_to_db, state)


async def column_analysis_node_runner(state: GlobalState, config: RunnableConfig):
    thread_id = config['metadata']['thread_id']
    column_analysis_tool = ColumnSmartParser(llm=llm_server.llm_ollama, thread_id=thread_id,
                                             minio_client=minio_db).build()
    return await runColumnParsing(column_analysis_tool, config, socket_manager, minio_db, chroma_db.add_to_db, state)


async def missingness_analysis_node_runner(state: GlobalState, config: RunnableConfig):
    return await runMissingnessNode(socket_manager, config, minio_db, chroma_db.add_to_db, state)


async def univariate_analysis_node_runner(state: GlobalState, config: RunnableConfig):
    return await runUnivariateNode(socket_manager, config, minio_db, chroma_db.add_to_db, state)


async def multivariate_analysis_node_runner(state: GlobalState, config: RunnableConfig):
    return await runMultivariateNode(socket_manager, config, minio_db, chroma_db.add_to_db, state)


async def agentic_analysis_node_runner(state: GlobalState, config: RunnableConfig):
    thread_id = config['metadata']['thread_id']

    try:
        await sandbox_connector.setup(thread_id)
    except Exception as e:
        return {"errors": f"Workspace Initialization Error: {str(e)}"}

    agentic_analysis_tool = AgenticAnalystNode(llm_server.llm_ollama, llm_server.llm_gemini, chroma_db,
                                               sandbox_connector, socket_manager, thread_id).build()
    state_update = await runAgenticAnalysisNode(agentic_analysis_tool, config, socket_manager, postgre_db, state)

    await sandbox_connector.clean(thread_id)
    return state_update


workflow = StateGraph(GlobalState)
workflow.add_node("data_ingestion", file_node_runner)
workflow.add_node("null_removal", null_node_runner)
workflow.add_node("column_analysis", column_analysis_node_runner)
workflow.add_node("missingness_analysis", missingness_analysis_node_runner)
workflow.add_node("univariate_analysis", univariate_analysis_node_runner)
workflow.add_node("multivariate_analysis", multivariate_analysis_node_runner)
workflow.add_node("agentic_analysis", agentic_analysis_node_runner)

workflow.add_edge(START, "data_ingestion")
workflow.add_edge("data_ingestion", "null_removal")
workflow.add_edge("null_removal", "column_analysis")
workflow.add_edge("column_analysis", "missingness_analysis")
workflow.add_edge("missingness_analysis", "univariate_analysis")
workflow.add_edge("univariate_analysis", "multivariate_analysis")
workflow.add_edge("multivariate_analysis", "agentic_analysis")
workflow.add_edge("agentic_analysis", END)
orchestrator = workflow.compile(checkpointer=MemorySaver())

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_graph_tasks: dict[str, asyncio.Task] = {}


async def background_pipeline_worker(thread_id: str, config: dict, fresh_start: bool):
    try:
        if not fresh_start:
            print(f"Resuming thread {thread_id}...")
            await orchestrator.ainvoke(None, config)
        else:
            print(f"Starting thread {thread_id}...")
            await orchestrator.ainvoke({
                "current_info": None,
                "user_input": ""
            }, config)
    except Exception as e:
        print(f"Fatal on thread {thread_id}: {str(e)}")
    finally:
        if thread_id in active_graph_tasks:
            del active_graph_tasks[thread_id]


@app.get("/page-data/")
async def get_page_data(thread_id: str, page: int):
    config = {"configurable": {"thread_id": thread_id}}
    current_info = orchestrator.get_state(config).values
    current_info = current_info.get("current_data_info", {})
    if current_info:
        progress = current_info.current_progress
        current_page_info = current_info.page_data
        page_data = current_page_info.get(page, {})
        ready = True
    else:
        progress = 0
        page_data = {}
        ready = False

    print(f"Ready: {ready}")
    return {"ready": ready, "progress": progress, "page_data": page_data}


@app.post("/reset/{thread_id}")
async def reset_pipeline_config(thread_id: str):
    try:
        if thread_id in active_graph_tasks:
            active_graph_tasks[thread_id].cancel()
            del active_graph_tasks[thread_id]
            print(f"Canceled task for thread: {thread_id}")

        if hasattr(orchestrator, "checkpointer") and orchestrator.checkpointer:
            storage = orchestrator.checkpointer.storage
            if thread_id in storage:
                del storage[thread_id]
                print(f"Cleared memory for thread: {thread_id}")

        minio_db.delete_folder(thread_id)
        postgre_db.reset_session_data(thread_id)

        try:
            await socket_manager.push(
                {"status": "pipeline_reset", "message": "State reset requested."},
                thread_id,
                msg_type="relay"
            )
        except Exception:
            pass
        return {
            "status": "success",
            "message": f"Successfully cleared all graph checkpoints and historical states for thread {thread_id}."
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute complete state purge for config: {str(e)}"
        )


@app.post("/upload/{thread_id}")
async def upload_file(thread_id: str, file: UploadFile = File(...)):
    file_content = await file.read()

    if file.filename.endswith(('.csv', '.txt')):
        df = pd.read_csv(io.BytesIO(file_content))
    elif file.filename.endswith('.json'):
        df = pd.read_json(io.BytesIO(file_content))
    elif file.filename.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(file_content))
    else:
        raise Exception("Unsupported file format.")

    file_base_name = os.path.splitext(file.filename)[0]
    minio_path = MINIO_PARQUET_ORIGINAL_PATH.format(thread_id=thread_id, file_base_name=file_base_name)

    minio_db.write_df(df, minio_path, thread_id)
    postgre_db.add_upload(thread_id, minio_path)

    await socket_manager.push({"rows": len(df), "cols": len(df.columns)}, thread_id, "dashboard_data")
    await socket_manager.push({
        "current_info": {
            "file_path": minio_path,
            "all_columns": df.columns.tolist(),
            "rows": len(df)
        },
        "user_input": "input state is changed, parse again."
    }, thread_id, "relay")
    await socket_manager.push({}, thread_id, "unlock")
    return {"type": "dashboard_data", "rows": len(df), "cols": len(df.columns), "minio_path": minio_path}


@app.delete("/upload/{thread_id}")
async def reset_upload_state(thread_id: str):
    await socket_manager.push({}, thread_id, "lock")
    files = postgre_db.get_uploads(thread_id)

    for m_path in files:
        minio_db.delete_df(m_path)

    postgre_db.delete_uploads(thread_id)

    await socket_manager.push({}, thread_id, "dashboard_data")
    await socket_manager.push({
        "current_info": {"file_path": ""},
        "user_input": "Empty input list try again"
    }, thread_id, "relay")

    return {"status": "success", "deleted_count": len(files)}


@app.get("/plot-data/")
async def get_plot_data(var1: str, var2: str, thread_id: str):
    minio_path = MINIO_PARQUET_PROCESSED_PATH.format(thread_id=thread_id)
    df = minio_db.get_df(minio_path, thread_id)
    if var1 not in df.columns or var2 not in df.columns:
        raise HTTPException(status_code=400, detail="One or both columns missing")

    plot_df = df[[var1, var2]].dropna()
    if len(plot_df) > 500:
        plot_df = plot_df.sample(500)

    chart_data = [
        {"x": row[var1], "y": row[var2]}
        for _, row in plot_df.iterrows()
    ]
    return chart_data


@app.get("/insite-data/")
async def get_insite_data(name: str, thread_id: str):
    output = postgre_db.get_insight(thread_id, name)

    if output is None:
        raise HTTPException(status_code=404, detail="Insight not found")

    plots = []
    for plot in output["plots"]:
        path = minio_db.return_plot_path(plot)
        if path is not None:
            plots.append(path)

    output["plots"] = plots
    return output


@app.get("/api/artifacts/{thread_id}/insights/{insight_id}/{filename}")
async def proxy_analysis_artifact(thread_id: str, insight_id: str, filename: str):
    object_key = f"{thread_id}/INSIGHTS/{insight_id}/{filename}"

    try:
        response = minio_db.minio_client.get_object("analysis-artifacts", object_key)
        return Response(content=response.read(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=404, detail="Visualization artifact not found")


def transform_report_payload(data_prime, df, insight_data):
    page0 = data_prime["current_data_info"].page_data[0]
    file_name = page0["file"][-30:]
    rows = page0["rows"]
    cols = page0["cols"]
    targets = page0["targets"]

    page1 = data_prime["current_data_info"].page_data[1]
    anomalies = page1["null_analysis"]["discovery_results"]

    missing = []
    for item in page1["col_analysis"]:
        if item.get('missingness_severity', 0) > 0:
            item_copy = item.copy()
            fmt_dict = item_copy.get("formatting", {})
            fmt_str = ", ".join([f"{k}: {v}" for k, v in fmt_dict.items()])
            item_copy["formatting"] = fmt_str if fmt_str else "N.A."
            missing.append(item_copy)

    page2 = data_prime["current_data_info"].page_data[2]
    page3 = data_prime["current_data_info"].page_data[3]

    relevant = {k: v for k, v in page3.items() if v.get('effect_val', 0) > 0.2 and v.get('rejected')}
    final_data = []

    for k, v in relevant.items():
        var1, var2 = [i.strip("()' ") for i in k.split(",")]

        multi_data = {
            "pair_id": k,
            "vars": [var1, var2],
            "pair_type": v["pair_type"],
            "rejected": v["rejected"],
            "test_used": v["test_used"],
            "p_adj": v["p_adj"],
            "effect_val": v["effect_val"],
            "magnitude": v["magnitude"],
        }

        if v["pair_type"] == "Num-Num":
            plot_df = df[[var1, var2]].dropna()
            if len(plot_df) > 500:
                plot_df = plot_df.sample(n=500, random_state=42)

            multi_data["scatter_points"] = [
                {"x": r[var1], "y": r[var2]}
                for r in plot_df.to_dict(orient="records")
            ]

        final_data.append(multi_data)

    return {
        "filename": "..." + file_name,
        "filesize": "UNK MB",
        "row_count": rows,
        "col_count": cols,
        "targets": targets,
        "columns": ["hello", "world"],
        "null_config": page1["null_analysis"]["search_config"],
        "anomalies": anomalies,
        "missingness_table": missing,
        "univariate_data": page2,
        "multivariate_meta": {
            "total_pairs": len(page3),
            "significant_hits": len(relevant)
        },
        "multivariate_data": final_data,
        "insights": insight_data
    }


@app.get("/generate-report")
async def get_report(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = orchestrator.get_state(config)

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Pipeline data not initialized for this thread.")

    data_prime = snapshot.values

    minio_path = MINIO_PARQUET_PROCESSED_PATH.format(thread_id=thread_id)
    df = minio_db.get_df(minio_path, thread_id)
    insight_data = postgre_db.get_report_insights(thread_id) or []

    for data in insight_data:
        plots = []
        for plot in data.get("plots", []):
            path = minio_db.return_plot_path(plot)
            if path is not None:
                plots.append(path)
        data["plots"] = plots

    report_data = await anyio.to_thread.run_sync(
        transform_report_payload, data_prime, df, insight_data
    )

    pdf_path = await generate_report_pdf(report_data)

    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=500, detail="Error generating final report PDF binary artifact.")

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename="Analysis_Report.pdf"
    )


@app.websocket("/ws/pipeline/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, thread_id: str):
    await socket_manager.connect(thread_id, websocket)
    config = {"configurable": {"thread_id": thread_id}}
    state_snapshot = orchestrator.get_state(config)

    fresh_start = not bool(state_snapshot.values)

    if thread_id in active_graph_tasks:
        active_graph_tasks[thread_id].cancel()

    task = asyncio.create_task(background_pipeline_worker(thread_id, config, fresh_start))
    active_graph_tasks[thread_id] = task

    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print(f"Disconnect for thread {thread_id}")
        socket_manager.disconnect(thread_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=APP_HOST, port=APP_PORT)