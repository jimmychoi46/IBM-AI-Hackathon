# Watsonx 오케스트레이트 연동 

이 파일은 백엔드 서버의 설정, 실행 및 프론트엔드 협업을 위한 통합 가이드입니다. 

에이전트의 비동기 연산 특성에 따라 엔드포인트로는 POST 와 GET을 사용하였습니다. 

-> 각 엔드포인트에 대한 설명은 '4. 프론트엔드 API 연동 명세'를 참고 바랍니다.

---

## 1. 환경 구축 및 라이브러리 설치

프로젝트 구동을 위해서는 파이썬(Python) 환경에서 아래 의존성 패키지를 설치해야 합니다.

```bash
pip install fastapi==0.115.6 uvicorn==0.34.0 httpx==0.28.1 python-dotenv==1.0.1
```

## 2. 환경 변수 설정 (.env)

프로젝트 루트 디렉토리에 .env 파일을 생성하고, 아래 항목에 정보를 입력하세요.

※ 해당 부분은 해당 에이전트를 구축한 담당자의 정보를 입력해주시길 바랍니다. (특히 API_key는 에이전트 구축자의 것을 사용하는 것을 권장함)


### IBM Cloud 인증 및 인스턴스 정보

IBM_API_KEY=당신의_API_key

INSTANCE_ID=당신의_인스턴스_ID

### 서비스 리전 (기본값: us-south)
REGION=us-south

### Watsonx 오케스트레이트 에이전트 정보 (Supervisor Agent의 Agent ID 사용 권장)
AGENT_ID=당신의_에이전트_id

AGENT_ENVIRONMENT_ID=당신의_에이전트_environment_id 

## 3. 서버 실행 방법

서버 메인 파일명이 server.py인 경우, 아래 명령어로 서버를 시작합니다.

```python
python server.py
```

Base URL: http://localhost:8000

대화형 API 문서 (Swagger): http://localhost:8000/docs

→ 여기서 직접 테스트 가능

## 4. 프론트엔드 API 연동 명세 (API Guide)


## 1. [POST] /api/chat

사용자의 질문을 백엔드를 거쳐 Watsonx 오케스트레이트에 전달, 응답을 받아옵니다.

### Request Body (JSON)

1) JSON 필드 (JSON 요청을 위함)

   - user_query (String) [필수]: 챗봇에게 전달할 사용자 질문 메시지
  
   - thread_id (string) [선택] 대화 문맥 유지를 위한 세션 ID(첫 질문 일 경우에는 생략 혹은 예시처럼 string으로 표기)-> 응답으로 받은 ID를 이후 요청에 재사용

2) JSON 요청 예시

```json
   {
     "user_query": "노들섬에서 국립중앙박물관 가는 법 알려줘"
     "thread_id": "string"
   }
```
### Response Body (JSON)

1) JSON 반환 필드

   - status (String): 성공 여부 ("success")

   - run_id: 작업 상태 조회를 위한 Job ID (GET 요청에 필요한 필드)

   - thread_id: 현재 대화 세션 ID (문맥 유지를 위해서는 프론트엔드에서 저장 필수)

2) JSON 응답 예시


```json
{
   "status": "success",
  "run_id": "8309bf4a-b8fd-4120-80d5-8a7df40dffd1",
  "thread_id": "f4b984f8-4ab6-4147-9663-05a21301e9f4"
}
```

## 2. [GET] /api/chat/status/{run_id}

POST 응답을 통해 반환된된 run_id를 사용하여 에이전트의 작업 완료 여부 확인 및 정제된 데이터 수신을 수행합니다.

### 1) Path Parameter (필수 인자)

   - run_id (String): POST /api/chat의 응답으로 받은 고유 식별자

### 2) Response Body (JSON 반환 필드 설명)

   - status (String): 현재 상태 ("running": 작업 중, "completed": 작업 완료)

   - answer (String): 완료 시 에이전트가 작성한 최종 텍스트 답변

   - itineraries (Array): 지도 UI 구성을 위한 정제된 경로 상세 데이터 배열
 
### 3) 응답 예시 (예시의 경우 Completed)
```json
{
  "status": "completed",
  "answer": "노들섬에서 국립중앙박물관까지의 경로 안내입니다.",
  "itineraries": [ { "duration": 1707, "legs": [...] } ]
}
```


## 5. Flutter 연동 가이드

아래의 폴링(Polling) 로직을 적용하십시오.

### 1. 연동 워크플로우
   1) POST /api/chat으로 run_id 획득

   2) GET /api/chat/status/{run_id}를 3~5초 간격으로 호출 (Polling)

   3) 응답의 status가 "completed"가 될 때까지 로딩 화면 유지

### 2. Dart 데이터 모델 (GET /api/chat/status/{run_id}을 통해 반환된 응답을 바탕으로 작성함)

```dart
class AgentResponse {
  final String status;
  final String answer;
  final List<Itinerary> itineraries;

  AgentResponse({required this.status, required this.answer, required this.itineraries});

  factory AgentResponse.fromJson(Map<String, dynamic> json) {
    return AgentResponse(
      status: json['status'] ?? '',
      answer: json['answer'] ?? '',
      itineraries: (json['itineraries'] as List)
          .map((i) => Itinerary.fromJson(i))
          .toList(),
    );
  }
}

class Itinerary {
  final int duration;
  final double distance;
  final List<dynamic> legs;

  Itinerary({required this.duration, required this.distance, required this.legs});

  factory Itinerary.fromJson(Map<String, dynamic> json) {
    return Itinerary(
      duration: json['duration'] ?? 0,
      distance: (json['distance'] ?? 0).toDouble(),
      legs: json['legs'] ?? [],
    );
  }
}
```

## 6.상세 예외 처리 가이드

통신 중 에러 발생 시 HTTP 상태 코드와 함께 아래와 같은 상세 메시지가 반환됩니다.

- 401 Unauthorized: 인증 오류 -> IBM API Key 설정 확인 필요.

- 503 Service Unavailable: 연결 오류(IBM Cloud 서버 접속 불가) -> 네트워크 확인 필요.

- 504 Gateway Timeout: 시간 초과(Watsonx 오케스트레이트 응답 지연 (설정 시간:총 60초[접속 10초 + 데이터 전송 50초]))
 
  -> 잠시 후 재시도 권장.

- run_id: "null" ->  요청 중 에이전트 할당 실패. 작업 생성이 안 된 경우이므로 재시도 필요


---
