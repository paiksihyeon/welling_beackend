from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from openai import OpenAI
import json, os, numpy as np
import pathlib
from app.utils.database import get_db
from app.utils.models import RegionData, RagSummary

router = APIRouter(prefix="/api/rag", tags=["RAG Pipeline"])
client = OpenAI()


# 🔹 코사인 유사도 계산
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# 🔹 JSON 파일 로더
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.post("/run-pipeline/")
def run_rag_pipeline(db: Session = Depends(get_db)):
    """
    RAG 파이프라인 전체 자동 실행
    1️⃣ 괴리 큰 지역 탐색
    2️⃣ 여론 벡터 → 시민 불만 요약
    3️⃣ 정책 벡터 → 유사 정책 검색
    4️⃣ LLM 종합 요약 생성
    5️⃣ JSON 파일 저장 및 응답
    """
    try:
        print("[RAG Pipeline] 시작")

        # ✅ 실제 경로 (files 폴더)
        current_dir = pathlib.Path(__file__).resolve()
        project_root = current_dir.parents[2]  # welling_backend/
        files_dir = project_root / "app" / "files"

        sentiment_path = files_dir / "sentiment_vectors.json"
        policy_path = files_dir / "policy_vectors.json"

        # 1️⃣ gap이 큰 지역 3곳 선택
        regions = db.query(RegionData).order_by(RegionData.gap_score.desc()).limit(3).all()
        if not regions:
            raise ValueError("데이터베이스에 지역 정보가 없습니다.")

        print(f"[RAG Pipeline] → {len(regions)}개 지역 로드 완료")

        # 2️⃣ 벡터 파일 로드
        sentiment_vectors = load_json(sentiment_path) if os.path.exists(sentiment_path) else {}
        policy_vectors = load_json(policy_path) if os.path.exists(policy_path) else []

        results = []

        for region in regions:
            print(f"[RAG Pipeline] ▶ {region.region_name} 지역 분석 시작")

            # 주제별 순회
            for topic in ["주거/환경", "인프라/교통", "의료/보건", "정책효능감", "노동/경제"]:
                region_vec = sentiment_vectors.get(region.region_name, {}).get(topic)
                if region_vec is None:
                    continue

                # 시민 불만 요약 요청
                prompt_opinion = (
                    f"지역 '{region.region_name}'의 '{topic}' 주제 관련 시민 여론을 분석하여, "
                    f"주요 불만 사항을 2~3문장으로 요약하세요."
                )
                citizen_summary = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "당신은 사회정책 분석 전문가입니다."},
                        {"role": "user", "content": prompt_opinion},
                    ],
                    max_tokens=250,
                ).choices[0].message.content.strip()

                # 정책 벡터 유사도 계산
                scored = []
                for p in policy_vectors:
                    try:
                        sim = cosine_similarity(region_vec, p["vector"])
                        scored.append((p["policy_name"], sim, p["description"]))
                    except Exception:
                        continue

                # 상위 3개 정책 선택
                top_policies = sorted(scored, key=lambda x: x[1], reverse=True)[:3]

                # 최종 정책 제안 생성
                prompt_final = (
                    f"'{region.region_name}'의 '{topic}' 관련 시민 불만:\n{citizen_summary}\n\n"
                    f"유사 정책 사례:\n"
                    + "\n".join([f"• {p[0]}: {p[2]}" for p in top_policies])
                    + "\n\n이를 기반으로 정책 개선 방향을 제안하세요."
                )
                final_summary = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "사회정책 전문가로서 종합 제안을 작성하세요."},
                        {"role": "user", "content": prompt_final},
                    ],
                    max_tokens=400,
                ).choices[0].message.content.strip()

                result_item = {
                    "region": region.region_name,
                    "topic": topic,
                    "citizen_summary": citizen_summary,
                    "policy_examples": [p[0] for p in top_policies],
                    "final_summary": final_summary,
                }
                results.append(result_item)

        # 결과 파일 저장
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", "rag_pipeline_result.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"[RAG Pipeline] 완료 → {output_path}")
        return {
            "status": "success",
            "count": len(results),
            "data": results,
            "saved_to": output_path,
            "updated_at": datetime.utcnow(),
        }

    except Exception as e:
        print(f"[RAG Pipeline] 오류 발생: {e}")
        return {"status": "error", "message": str(e)}
