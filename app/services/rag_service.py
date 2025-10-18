from sqlalchemy.orm import Session
from app.utils.models import RegionData, RagSummary
from datetime import datetime

"""
rag_service.py
AI 모델로부터 받은 정책 요약(RAG 결과)을 저장하는 서비스 로직입니다.
"""


def save_rag_summary(db: Session, region_name: str, topic: str, summary: str):
    """RAG 요약 결과 저장 및 지역 정보 자동 연동"""
    try:
        # 지역 정보 조회 또는 신규 생성
        region = db.query(RegionData).filter(RegionData.region_name == region_name).first()
        if not region:
            print(f"[rag_service] {region_name} 지역이 존재하지 않아 새로 생성합니다.")
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

        # 동일 주제의 이전 요약 삭제 (중복 방지)
        db.query(RagSummary).filter(
            RagSummary.region_id == region.id,
            RagSummary.topic == topic
        ).delete()

        # 새로운 요약 데이터 삽입
        new_summary = RagSummary(
            region_id=region.id,
            topic=topic,
            summary=summary,
            created_at=datetime.utcnow()
        )
        db.add(new_summary)

        # 지역 갱신일자 업데이트 및 커밋
        region.updated_at = datetime.utcnow()
        db.commit()

        print(f"[rag_service] '{region_name}' 지역의 '{topic}' 요약이 저장되었습니다.")
        return {"status": "success", "region": region_name, "topic": topic}

    except Exception as e:
        db.rollback()
        print(f"[rag_service] 오류 발생: {e}")
        return {"status": "error", "message": str(e)}


# 실행 확인용
if __name__ == "__main__":
    print("[rag_service.py] 모듈이 정상적으로 로드되었습니다.")
