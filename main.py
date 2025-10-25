# app/main.py
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from app.utils.database import engine, SessionLocal
from app.utils import models
from app.utils.schemas import RegionResponse
from app.routers import (
    region_router,
    health_router,
    analysis_router,
    analytics_router,
    rag_router,
    rag_query_router,
    rag_pipeline_router,
    analysis_diagnosis_router,
    rag_action_router,
)
from app.services.sentiment_service import save_sentiment_result
from app.services.rag_service import save_rag_summary
from app.services.gap_calculator import update_all_gap_scores

# ============================================================
# 🚀 FastAPI 애플리케이션 설정
# ============================================================
app = FastAPI(
    title="Welling Backend API",
    version="1.2.0",
    description=(
        "AI 기반 복지정책 불균형 지도 Welling 백엔드 API입니다.\n\n"
        "- ✅ `/api/analysis/diagnosis/{region}` : 지역 여론 기반 문제 진단\n"
        "- ✅ `/api/rag/action/{region}` : 지역별 정책 개선 방향 제안 (RAG 기반)"
    ),
)

# ============================================================
# 🌐 CORS 설정 (React 프론트엔드 연동 허용)
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용 (개발 환경용)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)
print("[main.py] ✅ CORS 설정이 적용되었습니다.")

# ============================================================
# 🧱 데이터베이스 테이블 생성
# ============================================================
models.Base.metadata.create_all(bind=engine)
print("[main.py] ✅ 데이터베이스 테이블이 생성되었습니다.")

# ============================================================
# 🔗 라우터 등록
# ============================================================
app.include_router(region_router.router, prefix="/api", tags=["Region"])
app.include_router(health_router.router, prefix="/api", tags=["Health"])
app.include_router(analysis_router.router, prefix="/api", tags=["Analysis"])
app.include_router(analytics_router.router, prefix="/api", tags=["Analytics"])
app.include_router(rag_router.router, prefix="/api", tags=["RAG"])
app.include_router(rag_query_router.router, prefix="/api", tags=["RAG Query"])
app.include_router(rag_pipeline_router.router, prefix="/api", tags=["RAG Pipeline"])

# 🧠 새로 추가된 AI 분석 라우터
app.include_router(analysis_diagnosis_router.router, prefix="/api", tags=["Analysis - Diagnosis"])
app.include_router(rag_action_router.router, prefix="/api", tags=["RAG - Policy Action"])

# ============================================================
# 🔍 라우터 등록 로그 출력
# ============================================================
print("[main.py] 🔗 등록된 라우터 목록:")
for route in app.routes:
    if hasattr(route, "path"):
        print(f"  → {route.path}")

# ============================================================
# 🌱 기본 루트 엔드포인트
# ============================================================
@app.get("/")
async def root():
    return {
        "message": "🌍 Welling Backend is running.",
        "version": "1.2.0",
        "available_endpoints": [
            "/api/analysis/diagnosis/{region_name}",
            "/api/rag/action/{region_name}",
            "/api/regions/",
        ],
    }

# ============================================================
# 🧪 테스트 엔드포인트 1: Pydantic 스키마 검증
# ============================================================
@app.get("/test-schema")
async def test_schema():
    region = RegionResponse(
        id=1,
        region_name="서울",
        policy_score=82.5,
        sentiment_score=32.1,
        gap_score=50.4,
        updated_at=datetime.utcnow(),
    )
    return region

# ============================================================
# 🧩 테스트 엔드포인트 2: 주요 서비스 로직 동작 확인
# ============================================================
@app.get("/test-services")
async def test_services():
    """
    주요 서비스 로직 검증용 (DB 쓰기/읽기, 감정 저장, RAG 요약, Gap 계산 등)
    """
    db = SessionLocal()
    try:
        save_sentiment_result(db, "서울", "복지 서비스가 부족해요", 30.5, "kobert")
        save_rag_summary(db, "서울", "청년복지정책", "서울은 청년 지원 정책을 확대 중임")
        update_all_gap_scores(db)
        return {"message": "서비스 로직 테스트 완료 ✅"}
    finally:
        db.close()
