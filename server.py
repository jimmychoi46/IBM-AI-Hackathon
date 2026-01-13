import httpx
import logging
import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional

TIMEOUT_SEC: float = 20.0  #타임아웃 기준 설정(20초)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("오케스트레이션-백엔드")

load_dotenv()
IBM_API_KEY = os.getenv('IBM_API_KEY')
INSTANCE_ID = os.getenv('INSTANCE_ID')
REGION = os.getenv("REGION", "us-south") #기본값은 댈러스로 설정함
AGENT_ID = os.getenv("AGENT_ID")

app = FastAPI(
    title="Watsonx 오케스트레이트 연동 서버",
    description="Watsonx 오케스트레이트 데이터를 Fetch하여 프론트엔드에 전달하는 백엔드 API",
    debug=True
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_URL: str = f"https://api.{REGION}.watson-orchestrate.ibm.com/instances/{INSTANCE_ID}/v1/orchestrate"


class ChatRequest(BaseModel):
    user_query: str


class ChatResponse(BaseModel):
    status: str
    answer: str
    data: Dict[str, Any]


#전역 예외 처리기
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"[System Error] 예기치 못한 시스템 오류 발생: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "백엔드 서버 내부 로직 오류가 발생하였습니다.",
            "detail": str(exc)
        }
    )


# IBM Cloud IAM 토큰 발급
async def get_ibm_token() -> str:
    if not IBM_API_KEY:
        logger.critical("[Config Error] IBM_API_KEY가 환경 변수에 설정되지 않았습니다.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="[서버 설정 오류] API 키가 누락되었습니다. .env 파일을 확인해 주세요."
        )

    url: str = "https://iam.cloud.ibm.com/identity/token"
    payload: Dict[str, str] = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": IBM_API_KEY
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, data=payload, timeout=5.0)
            response.raise_for_status()
            token = response.json().get("access_token", "")
            return token
        except httpx.HTTPStatusError as e:
            logger.error(f"[Auth Error] IBM IAM 인증 실패: {e.response.text}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="[인증 오류] IBM API Key가 유효하지 않거나 만료되었습니다. .env 설정을 확인해 주세요."
            )


#메인 API -> Watsonx 오케스트레이션 챗봇 데이터 fetch
@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(request_data: ChatRequest) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        try:
            # 1) IBM Cloud IAM 토큰 확보
            token = await get_ibm_token()

            # 2) Watsonx 오케스트레이션 API 요청 설정
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }

            # 2-1) 엔드포인트 설정
            endpoint = f"{BASE_URL}/chat/completions"
            payload = {
                "input": request_data.user_query,
                "context": {}
            }

            logger.info(f"[Request] Watsonx 오케스트레이트 데이터 Fetch 시도: {request_data.user_query}")

            # 3) Watsonx 오케스트레이트 서버 통신
            response = await client.post(endpoint, json=payload, headers=headers, timeout=20.0)

            # 3-1) 응답 상태 확인 및 예외 발생
            response.raise_for_status()

            # 4) 데이터 가공
            result: Dict[str, Any] = response.json()
            answer: str = result.get("output", "Watsonx 오케스트레이트 에이전트로부터 답변을 받지 못했습니다.")

            logger.info("[Success] Watsonx 오케스트레이트 데이터 Fetch 완료")

            return {
                "status": "success",
                "answer": answer,
                "data": result
            }

        # 5) 상세 예외 처리
        except httpx.ConnectError:
            logger.error("[Connect Error] Watsonx 오케스트레이트 서버 연결 불가.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="[연결 오류] IBM Cloud 서버에 접속할 수 없습니다. 네트워크를 확인하고 다시 시도해 주세요."
            )

        except httpx.TimeoutException:
            logger.error(f"[Timeout Error] Watsonx 오케스트레이트 서버 응답 시간 초과(설정 기간: {TIMEOUT_SEC}s).")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="[시간 초과] Watsonx 오케스트레이트 서버의 응답이 지연되고 있습니다. 잠시 후 다시 시도해 주세요."
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"[Status Error] Watsonx 오케스트레이트 API 오류 코드: {e.response.status_code}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"[서비스 오류] Watsonx 오케스트레이트 시스템 에러가 발생했습니다. (오류 코드: {e.response.status_code})"
            )

        except Exception as e:
            logger.error(f"[Unexpected Error] 비즈니스 로직 처리 중 예상치 못한 오류 발생: {str(e)}")
            raise e


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)