from app.utils.database import SessionLocal
from app.utils.models import RegionData, RagSummary, SentimentAnalysisLog
from datetime import datetime

# DB 세션 생성
db = SessionLocal()

# ✅ 기존 데이터 전체 삭제 (중복 방지)
db.query(SentimentAnalysisLog).delete()
db.query(RagSummary).delete()
db.query(RegionData).delete()
db.commit()
print("[insert_sample_data] 기존 데이터 모두 삭제 완료.")

# ------------------------------------------------------
# 1️⃣ RegionData 샘플 데이터
# ------------------------------------------------------
sample_regions = [
    {
        "region_name": "서울",
        "policy_score": 82.5,
        "sentiment_score": 40.2,
        "infra_sentiment": 55.0,
        "housing_sentiment": 47.8,
        "health_sentiment": 52.3,
        "economy_sentiment": 44.5,
        "policy_efficiency": 49.1,
    },
    {
        "region_name": "부산",
        "policy_score": 71.3,
        "sentiment_score": 61.7,
        "infra_sentiment": 58.1,
        "housing_sentiment": 65.0,
        "health_sentiment": 60.2,
        "economy_sentiment": 55.4,
        "policy_efficiency": 59.8,
    },
    {
        "region_name": "대전",
        "policy_score": 76.0,
        "sentiment_score": 45.5,
        "infra_sentiment": 49.5,
        "housing_sentiment": 53.3,
        "health_sentiment": 48.7,
        "economy_sentiment": 51.2,
        "policy_efficiency": 50.4,
    },
    {
        "region_name": "광주",
        "policy_score": 68.9,
        "sentiment_score": 59.1,
        "infra_sentiment": 62.5,
        "housing_sentiment": 60.4,
        "health_sentiment": 57.8,
        "economy_sentiment": 61.2,
        "policy_efficiency": 59.0,
    },
    {
        "region_name": "제주",
        "policy_score": 74.2,
        "sentiment_score": 66.3,
        "infra_sentiment": 69.2,
        "housing_sentiment": 71.4,
        "health_sentiment": 65.8,
        "economy_sentiment": 63.7,
        "policy_efficiency": 68.5,
    },
]

for item in sample_regions:
    region = RegionData(
        region_name=item["region_name"],
        policy_score=item["policy_score"],
        sentiment_score=item["sentiment_score"],
        gap_score=abs(item["policy_score"] - item["sentiment_score"]),
        infra_sentiment=item["infra_sentiment"],
        housing_sentiment=item["housing_sentiment"],
        health_sentiment=item["health_sentiment"],
        economy_sentiment=item["economy_sentiment"],
        policy_efficiency=item["policy_efficiency"],
        updated_at=datetime.now().replace(microsecond=0),
    )
    db.add(region)
db.commit()
print("[insert_sample_data] RegionData 샘플 데이터 추가 완료.")

# ------------------------------------------------------
# 2️⃣ RagSummary (정책 요약 및 제안 리스트)
# ------------------------------------------------------
sample_summaries = [
    {
        "region_name": "서울",
        "topic": "청년복지정책",
        "summary": "서울시는 청년 주거와 일자리 정책을 통합해 지원 중입니다.",
        "proposal_list": "청년 월세 지원 확대, 일자리 매칭 플랫폼 강화",
    },
    {
        "region_name": "부산",
        "topic": "노인복지정책",
        "summary": "부산은 노인 돌봄 서비스와 의료 접근성 개선을 추진 중입니다.",
        "proposal_list": "노인 요양 인프라 확충, 방문의료 서비스 확대",
    },
]

for s in sample_summaries:
    region = db.query(RegionData).filter_by(region_name=s["region_name"]).first()
    if region:
        rag = RagSummary(
            region_id=region.id,
            topic=s["topic"],
            summary=s["summary"],
            proposal_list=s["proposal_list"],
            created_at=datetime.now().replace(microsecond=0),
        )
        db.add(rag)
db.commit()
print("[insert_sample_data] RagSummary 샘플 데이터 추가 완료.")

# ------------------------------------------------------
# 3️⃣ SentimentAnalysisLog (감정 분석 로그)
# ------------------------------------------------------
sample_logs = [
    {
        "region_name": "서울",
        "text": "서울 청년정책에 대한 긍정적인 반응이 증가하고 있습니다.",
        "sentiment_score": 0.78,
        "model": "gpt-4-turbo",
    },
    {
        "region_name": "부산",
        "text": "부산 노인복지 관련 정책 만족도가 상승했습니다.",
        "sentiment_score": 0.82,
        "model": "gpt-4-turbo",
    },
]

for log in sample_logs:
    region = db.query(RegionData).filter_by(region_name=log["region_name"]).first()
    if region:
        entry = SentimentAnalysisLog(
            region_id=region.id,
            text=log["text"],
            sentiment_score=log["sentiment_score"],
            model=log["model"],
            created_at=datetime.now().replace(microsecond=0),
        )
        db.add(entry)
db.commit()
print("[insert_sample_data] SentimentAnalysisLog 샘플 데이터 추가 완료.")

# ------------------------------------------------------
# 종료
# ------------------------------------------------------
db.close()
print("[insert_sample_data] 샘플 데이터가 모두 성공적으로 추가되었습니다.")
