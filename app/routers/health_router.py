from fastapi import APIRouter
from datetime import datetime
from app.utils.database import engine

router = APIRouter()

"""
health_router.py
서버 및 DB 상태 확인용 API 라우터
- /api/health/
"""

@router.get("/health/")
def health_check():
    return {
        "db": str(engine.url),
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
    }

print("[health_router.py] 헬스체크 라우터가 성공적으로 로드되었습니다.")
