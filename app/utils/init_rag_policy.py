import pandas as pd
import os
from app.utils.database import SessionLocal
from app.utils.models import RagPolicy


def insert_rag_policy_data():
    db = SessionLocal()
    print("[init_rag_policy] RAG 정책 데이터 삽입을 시작합니다...")

    try:
        # ✅ CSV 경로
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "../files/rag_policy_dataset.csv")
        csv_path = os.path.normpath(csv_path)

        if not os.path.exists(csv_path):
            print(f"[init_rag_policy] ❌ CSV 파일이 존재하지 않습니다: {csv_path}")
            return

        # ✅ CSV 로드
        df = pd.read_csv(csv_path)
        print(f"[init_rag_policy] CSV 로드 완료: {len(df)}개 행")

        # ✅ 기존 데이터 초기화
        db.query(RagPolicy).delete()

        # ✅ 데이터 삽입
        for _, row in df.iterrows():
            record = RagPolicy(
                region=row["region"],
                policy=row["policy"]
            )
            db.add(record)

        db.commit()
        print(f"[init_rag_policy] ✅ {len(df)}개 행 삽입 완료")
    except Exception as e:
        db.rollback()
        print(f"[init_rag_policy] ❌ 오류 발생: {e}")
    finally:
        db.close()
        print("[init_rag_policy] DB 세션이 종료되었습니다.")


if __name__ == "__main__":
    insert_rag_policy_data()
