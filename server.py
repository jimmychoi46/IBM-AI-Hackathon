import traceback
import httpx
import logging
import os
import uvicorn
import re
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional

TIMEOUT_SEC = 60.0
# 타임아웃 기준 설정(접속 10초, 데이터 전송 50초, 총 60초)
timeout = httpx.Timeout(TIMEOUT_SEC, connect=10.0, read=50.0)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("오케스트레이트-백엔드")

# .env 로드 및 검증
load_dotenv()
IBM_API_KEY = os.getenv('IBM_API_KEY', '').strip()
BASE_URL = os.getenv('BASE_URL', '').strip().rstrip('/')
INSTANCE_ID = os.getenv('INSTANCE_ID', '').strip()
AGENT_ID = os.getenv("AGENT_ID", "").strip()
AGENT_ENVIRONMENT_ID = os.getenv("AGENT_ENVIRONMENT_ID", "").strip()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=timeout)
    yield
    await app.state.client.aclose()

app = FastAPI(
    title="Watsonx 오케스트레이트 연동 서버",
    description="Watsonx 오케스트레이트 데이터를 Fetch하여 프론트엔드에 전달하는 백엔드 API",
    debug=True,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_query: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    status: str
    answer: str
    data: Dict[str, Any]

async def get_ibm_token(client: httpx.AsyncClient):
    if not IBM_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="환경 변수(IBM_API_KEY) 확인 필요"
        )
    url: str = "https://iam.cloud.ibm.com/identity/token"
    payload: Dict[str, str] = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": IBM_API_KEY
    }
    try:
        response = await client.post(url, data=payload, timeout=5.0)
        response.raise_for_status()
        return response.json().get("access_token", "")
    except httpx.HTTPStatusError as e:
        logger.error(f"[Auth Error] 인증 실패: {e.response.text}")
        raise HTTPException(status_code=401, detail="인증 실패")



@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(request_data: ChatRequest, request: Request):
    client = request.app.state.client
    try:
        token = await get_ibm_token(client)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        endpoint = f"{BASE_URL}/instances/{INSTANCE_ID}/v1/orchestrate/runs"

        payload = {
            "message": {
                "role": "user",
                "content": request_data.user_query
            },
            "agent_id": AGENT_ID,
            "context": {},
            "additional_properties": {}
        }

        if request_data.thread_id and re.match(r'^[0-9a-fA-F-]{36}$', request_data.thread_id):
            payload["thread_id"] = request_data.thread_id

        response = await client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()

        return {
            "status": "success", 
            "answer": "요청 접수 완료", 
            "data": response.json()
        }

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/chat/status/{run_id}")
async def get_run_status(run_id: str, request: Request):
    client = request.app.state.client
    try:
        token = await get_ibm_token(client)
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        endpoint = f"{BASE_URL}/instances/{INSTANCE_ID}/v1/orchestrate/runs/{run_id}"
        response = await client.get(endpoint, headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
