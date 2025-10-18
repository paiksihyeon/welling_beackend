# app/services/gap_calculator.py

from sqlalchemy.orm import Session
from app.utils.models import RegionData
from datetime import datetime

"""
gap_calculator.py
정책 점수(policy_score)와 심리 점수(sentiment_score)의 차이를 기반으로
불균형 점수(gap_score)를 계산하고 DB에 반영하는 서비스 로직입니다.
"""


def calculate_gap(policy_score: float, sentiment_score: float) -> float:
    """단일 지역의 gap_score 계산"""
    try:
        if policy_score is None or sentiment_score is None:
            return 0.0
        return round(abs(policy_score - sentiment_score), 2)
    except Exception as e:
        print(f"[gap_calculator] gap 계산 중 오류: {e}")
        return 0.0


def update_all_gap_scores(db: Session):
    """DB 내 모든 지역의 gap_score를 일괄 계산 및 업데이트"""
    try:
        regions = db.query(RegionData).all()
        if not regions:
            print("[gap_calculator] 업데이트할 지역 데이터가 없습니다.")
            return {"status": "empty"}

        for region in regions:
            region.gap_score = calculate_gap(region.policy_score, region.sentiment_score)
            region.updated_at = datetime.now()
            db.add(region)

        db.commit()
        print(f"[gap_calculator] 모든 지역의 gap_score가 업데이트되었습니다. (총 {len(regions)}개)")
        return {"status": "success", "updated_regions": len(regions)}

    except Exception as e:
        db.rollback()
        print(f"[gap_calculator] DB 업데이트 중 오류 발생: {e}")
        return {"status": "error", "message": str(e)}


# 실행 확인용
if __name__ == "__main__":
    print("[gap_calculator.py] 모듈이 정상적으로 로드되었습니다.")
