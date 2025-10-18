from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.utils.database import get_db
from app.services.model_connector import generate_and_save_summary
from datetime import datetime

"""
rag_router.py
외부 AI 모델과 연동하여 요약을 생성하고 DB에 저장하는 라우터입니다.
"""

router = APIRouter(prefix="/rag", tags=["RAG"])


class RagRequest(BaseModel):
    region_name: str
    topic: str
    text: str


@router.post("/generate/")
def generate_rag_summary(request: RagRequest, db: Session = Depends(get_db)):
    """
    외부 AI 모델을 호출하여 정책 요약(RAG 결과)을 생성하고 DB에 저장
    """
    try:
        generate_and_save_summary(
            db=db,
            region_name=request.region_name,
            topic=request.topic,
            text=request.text
        )

        print(f"[rag_router] {request.region_name} 지역 '{request.topic}' RAG 요약 생성 완료")
        return {
            "status": "success",
            "region": request.region_name,
            "topic": request.topic,
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        print(f"[rag_router] 오류 발생: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    print("[rag_router.py] 모듈이 정상적으로 로드되었습니다.")
