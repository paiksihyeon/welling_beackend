import pandas as pd
from datetime import datetime
from app.utils.database import SessionLocal
from app.utils.models import SentimentAnalysisLog
import os


def insert_sentiment_dataset():
    """여론 데이터셋(senti_dataset.csv)을 sentiment_analysis_log 테이블에 삽입"""
    db = SessionLocal()
    print("[init_sentiment_data] Welling 여론 데이터셋 초기화를 시작합니다...")

    try:
        # ✅ CSV 경로 지정
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "../files/sentiment_dataset.csv")
        csv_path = os.path.normpath(csv_path)
        print(f"[init_sentiment_data] CSV 경로: {csv_path}")

        # ✅ 파일 존재 확인
        if not os.path.exists(csv_path):
            print("[init_sentiment_data] ❌ CSV 파일을 찾을 수 없습니다.")
            return

        # ✅ CSV 로드
        df = pd.read_csv(csv_path)
        print(f"[init_sentiment_data] CSV 로드 완료: {len(df)}개 행")

        # ✅ 기존 데이터 초기화
        db.query(SentimentAnalysisLog).delete()
        print("[init_sentiment_data] 기존 데이터 초기화 완료")

        # ✅ 데이터 삽입
        inserted = 0
        for _, row in df.iterrows():
            entry = SentimentAnalysisLog(
                region=row["region"],
                topic=row["topic"],
                text=row["text"],
                label=int(row["label"]),
            )
            db.add(entry)
            inserted += 1

        db.commit()
        print(f"[init_sentiment_data] ✅ {inserted}개 행이 성공적으로 삽입되었습니다.")
        print(f"[init_sentiment_data] 완료 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        db.rollback()
        print(f"[init_sentiment_data] ❌ 오류 발생: {e}")
    finally:
        db.close()
        print("[init_sentiment_data] DB 세션이 종료되었습니다.")
        print("[init_sentiment_data.py] 작업 완료 ✅")


if __name__ == "__main__":
    insert_sentiment_dataset()
