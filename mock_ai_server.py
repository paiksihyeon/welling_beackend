# mock_ai_server.py

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Mock AI Summary Server")

class SummaryRequest(BaseModel):
    region_name: str
    topic: str
    text: str

@app.post("/api/generate_summary")
def generate_summary(request: SummaryRequest):
    """
    AI 모델 없이 테스트용으로 동작하는 요약 응답 API
    """
    # 요청 로그 출력
    print(f"[Mock AI] 요청 수신 → 지역: {request.region_name}, 주제: {request.topic}")

    # 가짜 요약 생성
    fake_summary = f"{request.region_name} 지역의 {request.topic} 관련 정책은 전반적으로 긍정적이며, 복지 효율성이 높게 평가됨."

    return {"summary": fake_summary}


if __name__ == "__main__":
    import uvicorn
    print("[mock_ai_server.py] 테스트용 AI 서버가 실행 중입니다...")
    uvicorn.run(app, host="127.0.0.1", port=5000)
