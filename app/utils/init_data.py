from app.utils.database import SessionLocal
from app.utils.models import RegionData

REGIONS = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
]


def insert_initial_regions():
    """17개 광역자치단체를 RegionData 테이블에 삽입"""
    db = SessionLocal()
    try:
        existing_regions = {r.region_name for r in db.query(RegionData.region_name).all()}
        inserted = 0

        for region in REGIONS:
            if region not in existing_regions:
                new_region = RegionData(
                    region_name=region,
                    policy_score=0.0,
                    sentiment_score=0.0,
                    gap_score=0.0
                )
                db.add(new_region)
                inserted += 1

        db.commit()
        print(f"[init_data] {inserted}개 지역이 새로 추가되었습니다.")
        print(f"[init_data] 총 지역 수: {len(REGIONS)}개")
    except Exception as e:
        print(f"[init_data] 오류 발생: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    print("[init_data.py] 초기 데이터 삽입을 시작합니다...")
    insert_initial_regions()
    print("[init_data.py] 작업 완료.")
