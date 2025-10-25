"""
run_map_pipeline.py
------------------------------------------
지도용 백엔드 파이프라인 전체 실행 스크립트
(정책-여론 괴리 계산 → DB 업데이트 → 결과 요약)
"""

from app.utils.database import SessionLocal, engine
from app.utils import models
from app.services.gap_calculator import update_all_gap_scores
from datetime import datetime
import json
import os

# 실행 로그용
print("[run_map_pipeline] 지도 파이프라인 실행 시작")

# 1️⃣ DB 초기화 및 테이블 생성 확인
models.Base.metadata.create_all(bind=engine)
db = SessionLocal()
print("[1/4] 데이터베이스 초기화 완료")

# 2️⃣ 샘플 데이터 존재 확인 (없으면 자동 삽입)
existing_regions = db.query(models.RegionData).count()
if existing_regions == 0:
    print("[2/4] 샘플 데이터가 없습니다 → 기본 데이터 삽입 중...")
    sample_data = [
        {"region_name": "서울", "policy_score": 82.5, "sentiment_score": 40.2},
        {"region_name": "부산", "policy_score": 71.3, "sentiment_score": 61.7},
        {"region_name": "대전", "policy_score": 76.0, "sentiment_score": 45.5},
        {"region_name": "광주", "policy_score": 68.9, "sentiment_score": 59.1},
        {"region_name": "제주", "policy_score": 74.2, "sentiment_score": 66.3},
    ]
    for s in sample_data:
        region = models.RegionData(
            region_name=s["region_name"],
            policy_score=s["policy_score"],
            sentiment_score=s["sentiment_score"],
            gap_score=abs(s["policy_score"] - s["sentiment_score"]),
            updated_at=datetime.utcnow(),
        )
        db.add(region)
    db.commit()
    print("[2/4] 샘플 데이터 삽입 완료")
else:
    print(f"[2/4] 기존 지역 데이터 {existing_regions}개 존재")

# 3️⃣ Gap Score 업데이트 (정책 - 감정 괴리 계산)
update_all_gap_scores(db)
print("[3/4] Gap Score 업데이트 완료")

# 4️⃣ 지도 시각화용 요약 데이터 생성
regions = db.query(models.RegionData).all()
summary = []
for r in regions:
    summary.append({
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
    })

db.close()

# 5️⃣ 결과 JSON 파일로 저장
os.makedirs("output", exist_ok=True)
output_path = os.path.join("output", "map_pipeline_result.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=4)

print(f"[4/4] 결과 파일 저장 완료 → {output_path}")
print("✅ 지도 파이프라인 실행 완료.")
