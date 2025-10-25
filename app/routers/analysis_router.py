from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.utils.database import get_db
from app.services.sentiment_service import save_sentiment_result
from app.services.rag_service import save_rag_summary
from app.services.gap_calculator import update_all_gap_scores
from app.utils.models import RegionData
from datetime import datetime
import random
import json
import os

router = APIRouter(prefix="/api/analysis", tags=["Analysis"])

# 요청 body용 Pydantic 모델
class AnalysisRequest(BaseModel):
    region_name: str
    topic: str
    summary: str


@router.post("/run_analysis/")
def run_analysis(request: AnalysisRequest, db: Session = Depends(get_db)):
    """
    특정 지역(region_name)에 대해 AI 분석 파이프라인 실행
    1. 감정분석 점수 저장
    2. 정책 요약(RAG) 저장
    3. Gap Score 재계산
    """
    try:
        # 감정 분석 점수 계산 (테스트용 랜덤값)
        sentiment_score = round(random.uniform(0, 100), 2)

        save_sentiment_result(
            db=db,
            region_name=request.region_name,
            text=request.summary,
            score=sentiment_score,
            model="kobert"
        )

        # RAG 요약 저장
        save_rag_summary(
            db=db,
            region_name=request.region_name,
            topic=request.topic,
            summary=request.summary
        )

        # Gap Score 전체 업데이트
        update_all_gap_scores(db)

        print(f"[analysis_router] {request.region_name} 지역의 분석 완료")
        return {
            "status": "success",
            "region": request.region_name,
            "sentiment_score": sentiment_score,
            "topic": request.topic,
            "summary": request.summary,
            "updated_at": datetime.utcnow()
        }

    except Exception as e:
        db.rollback()
        print(f"[analysis_router] 오류 발생: {e}")
        return {"status": "error", "message": str(e)}


# -----------------------------------------------------------
# 지도용 파이프라인 자동 실행 API 추가
# -----------------------------------------------------------

@router.post("/run-map/")
def run_map_pipeline(db: Session = Depends(get_db)):
    """
    지도용 파이프라인 실행 API
    (정책-여론 점수 계산 → Gap 계산 → 결과 반환 및 파일 저장)
    """
    try:
        # 1️⃣ 지역 데이터 확인 (없으면 샘플 삽입)
        regions = db.query(RegionData).all()
        if not regions:
            print("[run-map] 지역 데이터 없음 → 샘플 데이터 추가 중...")
            sample_data = [
                {"region_name": "서울", "policy_score": 82.5, "sentiment_score": 40.2},
                {"region_name": "부산", "policy_score": 71.3, "sentiment_score": 61.7},
                {"region_name": "대전", "policy_score": 76.0, "sentiment_score": 45.5},
                {"region_name": "광주", "policy_score": 68.9, "sentiment_score": 59.1},
                {"region_name": "제주", "policy_score": 74.2, "sentiment_score": 66.3},
            ]
            for s in sample_data:
                region = RegionData(
                    region_name=s["region_name"],
                    policy_score=s["policy_score"],
                    sentiment_score=s["sentiment_score"],
                    gap_score=abs(s["policy_score"] - s["sentiment_score"]),
                    updated_at=datetime.utcnow(),
                )
                db.add(region)
            db.commit()
            print("[run-map] 샘플 데이터 삽입 완료")

        # 2️⃣ Gap Score 계산
        update_all_gap_scores(db)

        # 3️⃣ 최신 데이터 조회
        updated_regions = db.query(RegionData).all()
        result = [
            {
                "region_name": r.region_name,
                "policy_score": r.policy_score,
                "sentiment_score": r.sentiment_score,
                "gap_score": r.gap_score,
                "infra_sentiment": getattr(r, "infra_sentiment", None),
                "housing_sentiment": getattr(r, "housing_sentiment", None),
                "health_sentiment": getattr(r, "health_sentiment", None),
                "economy_sentiment": getattr(r, "economy_sentiment", None),
                "policy_efficiency": getattr(r, "policy_efficiency", None),
                "updated_at": r.updated_at.isoformat() if r.updated_at else None
            }
            for r in updated_regions
        ]

        # 4️⃣ 결과 파일로 저장
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", "map_pipeline_result.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        print(f"[run-map] 지도 파이프라인 완료 → 결과 저장: {output_path}")
        return {
            "status": "success",
            "count": len(result),
            "updated_at": datetime.utcnow(),
            "data": result
        }

    except Exception as e:
        db.rollback()
        print(f"[analysis_router/run-map] 오류 발생: {e}")
        return {"status": "error", "message": str(e)}


# 실행 확인용
if __name__ == "__main__":
    print("[analysis_router.py] 모듈이 정상적으로 로드되었습니다.")
