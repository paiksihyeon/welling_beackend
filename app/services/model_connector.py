# app/services/model_connector.py

import requests
from datetime import datetime
from sqlalchemy.orm import Session
from app.services.rag_service import save_rag_summary

"""
model_connector.py
외부 AI 모델(예: HuggingFace, OpenAI, KoBERT 기반 API 등)에 요청을 보내
정책 요약 결과를 받아오는 모듈입니다.
"""

# 예시용 AI 서버 엔드포인트 (실제 API URL로 교체 필요)
AI_API_URL = "http://127.0.0.1:5000/api/generate_summary"


def request_summary_from_model(region_name: str, topic: str, text: str) -> str:
    """
    외부 AI 서버에 요약 요청을 보내고 결과를 반환
    - region_name: 지역 이름
    - topic: 주제 (예: 청년정책, 의료서비스 등)
    - text: 원문 (AI가 요약할 텍스트)
    """
    try:
        payload = {
            "region_name": region_name,
            "topic": topic,
            "text": text
        }

        response = requests.post(AI_API_URL, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        summary = data.get("summary", "")
        print(f"[model_connector] 요약 결과 수신 완료: {summary}")
        return summary

    except Exception as e:
        print(f"[model_connector] 모델 요청 중 오류 발생: {e}")
        return "요약 생성 실패"


def generate_and_save_summary(db: Session, region_name: str, topic: str, text: str):
    """
    외부 AI 모델에서 요약을 생성하고 RAG Summary 테이블에 저장
    """
    summary = request_summary_from_model(region_name, topic, text)
    save_rag_summary(db, region_name, topic, summary)
    print(f"[model_connector] {region_name} 지역의 '{topic}' 요약이 DB에 저장되었습니다.")


# 실행 확인용
if __name__ == "__main__":
    print("[model_connector.py] 모듈이 정상적으로 로드되었습니다.")
