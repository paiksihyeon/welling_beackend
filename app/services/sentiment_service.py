# app/services/sentiment_service.py

from sqlalchemy.orm import Session
from app.utils.models import RegionData, SentimentAnalysisLog
from app.services.gap_calculator import calculate_gap
from datetime import datetime

"""
sentiment_service.py
감정 분석 결과를 DB에 저장하고, 해당 지역의 sentiment_score 및 gap_score를 갱신합니다.
"""

def save_sentiment_result(
    db: Session,
    region_name: str,
    text: str,
    score: float,
    model: str = "unknown"
):
    """감정 분석 결과 저장 및 지역 점수 갱신"""
    try:
        # 지역 데이터 조회 또는 신규 생성
        region = db.query(RegionData).filter(RegionData.region_name == region_name).first()
        if not region:
            print(f"[sentiment_service] {region_name} 지역이 존재하지 않아 새로 생성합니다.")
            region = RegionData(
                region_name=region_name,
                policy_score=0.0,
                sentiment_score=0.0,
                gap_score=0.0,
                updated_at=datetime.utcnow()
            )
            db.add(region)
            db.commit()
            db.refresh(region)

        # 감정 분석 로그 추가
        log = SentimentAnalysisLog(
            region_id=region.id,
            text=text,
            sentiment_score=round(score, 2),
            model=model,
            created_at=datetime.utcnow()
        )
        db.add(log)

        # 지역 심리·불균형 점수 갱신
        region.sentiment_score = round(score, 2)
        region.gap_score = calculate_gap(region.policy_score, region.sentiment_score)
        region.updated_at = datetime.utcnow()

        # 커밋 및 로그 출력
        db.commit()
        print(f"[sentiment_service] '{region_name}' 감정 점수({score})가 저장되었습니다.")
        return {"status": "success", "region": region_name, "score": score}

    except Exception as e:
        db.rollback()
        print(f"[sentiment_service] 오류 발생: {e}")
        return {"status": "error", "message": str(e)}
