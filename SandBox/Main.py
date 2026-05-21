import os
import shutil
import sys
import uuid
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from Minio import DataBaseMinio

app = FastAPI(title="Secure Code Execution Sandbox with MinIO Support")

BASE_WORKSPACE = "/workspace/sandboxes"
minio_db = DataBaseMinio()


class ExecutionRequest(BaseModel):
    code: str
    thread_id: str
    minio_input_path: Optional[str] = None


class ExecutionResponse(BaseModel):
    stdout: str
    stderr: str
    generated_files_minio: List[str]
    exit_code: int

@app.post("/upload/{thread_id}")
async def upload_file(thread_id: str):
    session_dir = os.path.abspath(os.path.join(BASE_WORKSPACE, thread_id))
    output_dir = os.path.join(session_dir, "Output")
    os.makedirs(output_dir, exist_ok=True)

    df = minio_db.get_df(f"{thread_id}/MISSINGNESS_IMPUTED/Processed.parquet", thread_id)
    df.to_parquet(os.path.join(session_dir, "Input.parquet"), index=False, engine='pyarrow', compression='snappy')
    return "Environment created successfully"


@app.post("/execute", response_model=ExecutionResponse)
async def execute_code(request: ExecutionRequest):
    session_dir = os.path.abspath(os.path.join(BASE_WORKSPACE, request.thread_id))
    output_dir = os.path.join(session_dir, "Output")
    script_filename = f"script_{request.thread_id}.py"
    script_path = os.path.join(session_dir, script_filename)

    setup_code = "import matplotlib\nmatplotlib.use('Agg')\n"
    full_code = setup_code + request.code

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(full_code)

    try:
        process = subprocess.run(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=session_dir,
            timeout=15.0
        )

        uploaded_minio_paths = []
        if os.path.exists(output_dir) and process.returncode == 0:
            for file in os.listdir(output_dir):
                local_file_path = os.path.join(output_dir, file)

                if os.path.isfile(local_file_path):
                    remote_minio_path = f"{request.thread_id}/INSIGHTS/{uuid.uuid4()}/{file}"
                    minio_db.write_image(local_file_path, remote_minio_path)
                    uploaded_minio_paths.append(remote_minio_path)
        return ExecutionResponse(
            stdout=process.stdout,
            stderr=process.stderr,
            generated_files_minio=uploaded_minio_paths,
            exit_code=process.returncode
        )
    except subprocess.TimeoutExpired:
        return ExecutionResponse(
            stdout="",
            stderr="Execution Error: Code execution timed out after 15 seconds.",
            generated_files_minio=[],
            exit_code=124
        )
    except Exception as e:
        return ExecutionResponse(
            stdout="",
            stderr=f"System Error running sandbox: {str(e)}",
            generated_files_minio=[],
            exit_code=500
        )
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


@app.post("/cleanup/{thread_id}")
async def cleanup_folder(thread_id: str):
    session_dir = os.path.abspath(os.path.join(BASE_WORKSPACE, thread_id))

    try:
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
            return f"Environment for thread {thread_id} completely removed successfully."
        else:
            return f"No environment folder found for thread {thread_id}. Nothing to delete."

    except Exception as e:
        return f"Error cleaning up environment for thread {thread_id}: {str(e)}"