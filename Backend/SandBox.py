import os
import httpx
from typing import Optional
from URLs import SANDBOX_URL

ENDPOINT_UPLOAD = "/upload/{thread_id}"
ENDPOINT_EXECUTE = "/execute"
ENDPOINT_CLEANUP = "/cleanup/{thread_id}"


class SandBox:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("SANDBOX_SERVICE_URL", SANDBOX_URL)

    async def setup(self, thread_id: str):
        url = f"{self.base_url}{ENDPOINT_UPLOAD.format(thread_id=thread_id)}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url)
                if response.status_code != 200:
                    raise RuntimeError(f"Sandbox setup failed ({response.status_code}): {response.text}")
                return response.json()
            except httpx.RequestError as e:
                raise RuntimeError(f"Failed to connect to sandbox server during setup: {e}")

    async def execute(self, thread_id: str, code: str, minio_input_path: Optional[str] = None):
        url = f"{self.base_url}{ENDPOINT_EXECUTE}"
        payload = {
            "code": code,
            "thread_id": thread_id,
            "minio_input_path": minio_input_path
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    return {
                        "stdout": "",
                        "stderr": f"Sandbox Execution Service HTTP Error {response.status_code}: {response.text}",
                        "generated_files_minio": [],
                        "exit_code": response.status_code
                    }
                return response.json()
            except httpx.RequestError as e:
                return {
                    "stdout": "",
                    "stderr": f"Failed to connect to sandbox server during code execution: {str(e)}",
                    "generated_files_minio": [],
                    "exit_code": 500
                }

    async def clean(self, thread_id: str) -> str:
        url = f"{self.base_url}{ENDPOINT_CLEANUP.format(thread_id=thread_id)}"

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.post(url)
                if response.status_code != 200:
                    raise RuntimeError(f"Sandbox cleanup failed ({response.status_code}): {response.text}")
                return response.json()
            except httpx.RequestError as e:
                raise RuntimeError(f"Failed to connect to sandbox server during cleanup: {e}")