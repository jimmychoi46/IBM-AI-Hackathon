import traceback
import httpx
import logging
import os
import uvicorn
import re
import json
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional


# 에이전트 연산 시간을 고려한 read timeout 확장
TIMEOUT_SEC = 60.0
timeout = httpx.Timeout(TIMEOUT_SEC, connect=10.0, read=50.0)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend-orchestrate")

# 환경 설정 검증
load_dotenv()
IBM_API_KEY = os.getenv('IBM_API_KEY', '').strip()
BASE_URL = os.getenv('BASE_URL', '').strip().rstrip('/')
INSTANCE_ID = os.getenv('INSTANCE_ID', '').strip()
AGENT_ID = os.getenv("AGENT_ID", "").strip()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 가동 시 Async HTTP Client pool 초기화 (성능 최적화)"""
    app.state.client = httpx.AsyncClient(timeout=timeout)
    yield
    await app.state.client.aclose()

app = FastAPI(title="Watsonx Real-time API", lifespan=lifespan)

# CORS -> Flutter Web 및 외부 기기 접속 허용을 위함
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

async def get_ibm_token(client: httpx.AsyncClient) -> str:
    """IBM Cloud IAM 기반 OAuth 2.0 Access Token 수신"""
    url = "https://iam.cloud.ibm.com/identity/token"
    payload = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": IBM_API_KEY}
    response = await client.post(url, data=payload, timeout=5.0)
    response.raise_for_status()
    return response.json().get("access_token", "")

@app.post("/api/chat")
async def chat_with_agent(request_data: ChatRequest, request: Request):
    """
    [POST] Orchestrate Run 초기화

    - 에이전트에게 작업을 명령하고 비동기 Job ID(run_id)를 수신함.
    - Flutter는 수신한 run_id로 결과 폴링을 수행할 것.
    """
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
            "message": {"role": "user", "content": request_data.user_query},
            "agent_id": AGENT_ID,
            "context": {},
            "additional_properties": {}
        }
        
        # 유효한 UUID 형식일 경우에만 thread_id 세션을 유지하도록 설정함
        if request_data.thread_id and re.match(r'^[0-9a-fA-F-]{36}$', request_data.thread_id):
            payload["thread_id"] = request_data.thread_id

        response = await client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        res_json = response.json()

        # IBM Response Schema 대응 (버전별 key name 차이 고려)
        run_id = res_json.get("id") or res_json.get("run_id") or res_json.get("data", {}).get("id")

        return {
            "status": "success", 
            "run_id": run_id, 
            "thread_id": res_json.get("thread_id")
        }
    except Exception:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="IBM Orchestrate 초기화 실패")

@app.get("/api/chat/status/{run_id}")
async def get_run_status(run_id: str, request: Request):
    """
    [GET] 작업 상태 폴링 및 결과 정제

    - status: 'running' (진행중), 'completed' (완료)
    - 완료 시 IBM의 중첩된 JSON을 Flattening하여 Flutter 친화적인 구조로 변환 수행
    """
    if not run_id or run_id == "null":
        return {"status": "error", "message": "Invalid run_id"}

    client = request.app.state.client
    try:
        token = await get_ibm_token(client)
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        endpoint = f"{BASE_URL}/instances/{INSTANCE_ID}/v1/orchestrate/runs/{run_id}"
        
        response = await client.get(endpoint, headers=headers)
        response.raise_for_status()
        raw_data = response.json()
        
        run_status = raw_data.get("status")
        answer_text = ""
        itineraries = []

        if run_status == "completed":
            # 1) 답변 텍스트 파싱
            content = raw_data.get("result", {}).get("data", {}).get("message", {}).get("content", [])
            if content:
                answer_text = content[0].get("text", "")

            # 2) tool_response 내부에 String으로 인코딩된 JSON 데이터(OTP 결과) 추출
            try:
                history = raw_data.get("result", {}).get("data", {}).get("message", {}).get("step_history", [])
                for step in history:
                    for detail in step.get("step_details", []):
                        if detail.get("type") == "tool_response":
                            content_str = detail.get("content", "")
                            # tripPatterns 키워드 포함 시 상세 경로로 간주함.
                            if "tripPatterns" in content_str:
                                # Double JSON Decode -> 문자열을 실제 JSON 객체로 변환
                                parsed_inner = json.loads(content_str)
                                itineraries = parsed_inner.get("data", {}).get("trip", {}).get("tripPatterns", [])
            except Exception as parse_err:
                logger.warning(f"Payload parsing failed: {parse_err}")

        return {
            "status": run_status,
            "answer": answer_text,
            "itineraries": itineraries
        }
    except Exception:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="상태 조회 실패")

if __name__ == "__main__":
    # Local Network 내 타 기기(Flutter)의 인바운드 허용을 위해 0.0.0.0 바인딩 수행
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
