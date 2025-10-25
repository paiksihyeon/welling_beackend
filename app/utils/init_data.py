import os
import pandas as pd
from datetime import datetime, timezone
from app.utils.database import SessionLocal, get_engine
from app.utils.models import Base, RegionData


def create_tables_if_not_exist():
    engine = get_engine()
    print("[init_data] 테이블 존재 여부 확인 중...")
    Base.metadata.create_all(bind=engine)
    print("[init_data] 테이블 확인/생성 완료")


def insert_real_dataset():
    db = SessionLocal()
    try:
        print("[init_data] 실제 데이터셋 삽입을 시작합니다...")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.normpath(os.path.join(base_dir, "../files/Welling_Master_dataset.csv"))
        print(f"[init_data] CSV 경로: {csv_path}")

        if not os.path.exists(csv_path):
            print(f"[init_data] ⚠️ CSV 파일을 찾을 수 없습니다. 실제 경로: {csv_path}")
            return

        df = pd.read_csv(csv_path, encoding="utf-8")

        # NaN 값 0.0으로 대체
        df = df.fillna(0.0)

        db.query(RegionData).delete()

        for _, row in df.iterrows():
            new_region = RegionData(
                region_name=row["region"],

                policy_avg_score=float(row["policy_avg_score"]),
                transport_infra_policy_score=float(row["transport_infra_policy_score"]),
                labor_economy_policy_score=float(row["labor_economy_policy_score"]),
                healthcare_policy_score=float(row["healthcare_policy_score"]),
                policy_efficiency_score=float(row["policy_efficiency_score"]),
                housing_environment_policy_score=float(row["housing_environment_policy_score"]),

                sentiment_avg_score=float(row["sentiment_avg_score"]),
                sentiment_transport_infra_score=float(row["sentiment_transport_infra_score"]),
                sentiment_labor_economy_score=float(row["sentiment_labor_economy_score"]),
                sentiment_healthcare_score=float(row["sentiment_healthcare_score"]),
                sentiment_policy_efficiency_score=float(row["sentiment_policy_efficiency_score"]),
                sentiment_housing_environment_score=float(row["sentiment_housing_environment_score"]),

                gap_score=float(row["gap_score"]),
                updated_at=datetime.now(timezone.utc),
            )
            db.add(new_region)

        db.commit()
        print(f"[init_data] ✅ {len(df)}개 지역 데이터 삽입 완료")

    except Exception as e:
        db.rollback()
        print(f"[init_data] ❌ 오류 발생: {e}")
    finally:
        db.close()
        print("[init_data] DB 세션이 종료되었습니다.")


if __name__ == "__main__":
    print("[init_data.py] Welling 실제 데이터셋 초기화를 시작합니다...")
    create_tables_if_not_exist()
    insert_real_dataset()
    print("[init_data.py] 작업 완료 ✅")
