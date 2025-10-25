from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from openai import OpenAI
import json, os, numpy as np
from pathlib import Path

from app.utils.database import get_db
from app.utils.models import RegionData

# ---------------------------------------------
# 라우터 기본 설정
# ---------------------------------------------
router = APIRouter(prefix="/api/rag", tags=["RAG Pipeline"])
client = OpenAI()

# 경로 설정
current_file = Path(__file__).resolve()
app_dir = current_file.parents[1]
project_root = app_dir.parent
files_dir = app_dir / "files"

sentiment_path = files_dir / "sentiment_vectors.json"
policy_path = files_dir / "policy_vectors.json"

# 파일 존재 검사
if not files_dir.exists():
    raise FileNotFoundError(f"[RAG Pipeline] files 폴더가 존재하지 않습니다: {files_dir}")
if not sentiment_path.exists():
    raise FileNotFoundError(f"[RAG Pipeline] sentiment_vectors.json 파일을 찾을 수 없습니다: {sentiment_path}")
if not policy_path.exists():
    raise FileNotFoundError(f"[RAG Pipeline] policy_vectors.json 파일을 찾을 수 없습니다: {policy_path}")

print(f"[RAG Pipeline] files 경로: {files_dir}")
print(f"[RAG Pipeline] sentiment_vectors.json 존재 여부: {sentiment_path.exists()}")
print(f"[RAG Pipeline] policy_vectors.json 존재 여부: {policy_path.exists()}")

# ---------------------------------------------
# 코사인 유사도 계산
# ---------------------------------------------
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ---------------------------------------------
# JSON 파일 로더
# ---------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------
# RAG 전체 파이프라인
# ---------------------------------------------
@router.post("/run-pipeline/")
def run_rag_pipeline(db: Session = Depends(get_db)):
    """
    RAG 파이프라인 전체 자동 실행
    1. Gap이 큰 지역 3곳 탐색
    2. 여론 벡터 기반 시민 불만 요약
    3. 정책 벡터 기반 유사 정책 검색
    4. LLM 기반 종합 정책 제안 생성
    5. 결과를 JSON 파일로 저장
    """
    try:
        print("[RAG Pipeline] 시작")

        # 1. Gap이 큰 지역 3곳 선택
        regions = db.query(RegionData).order_by(RegionData.gap_score.desc()).limit(3).all()
        if not regions:
            raise ValueError("데이터베이스에 지역 정보가 없습니다.")
        print(f"[RAG Pipeline] {len(regions)}개 지역 로드 완료")

        # 2. 벡터 파일 로드
        sentiment_vectors = load_json(sentiment_path)
        policy_vectors = load_json(policy_path)
        results = []

        # 3. 지역별 분석
        for region in regions:
            print(f"[RAG Pipeline] {region.region_name} 지역 분석 시작")

            for topic in ["주거/환경", "인프라/교통", "의료/보건", "정책효능감", "노동/경제"]:
                region_vec = None

                # dict 구조 처리
                if isinstance(sentiment_vectors, dict):
                    region_vec = sentiment_vectors.get(region.region_name, {}).get(topic)
                # list 구조 처리
                elif isinstance(sentiment_vectors, list):
                    for item in sentiment_vectors:
                        if (
                            item.get("region") == region.region_name
                            and item.get("topic") == topic
                        ):
                            region_vec = item.get("vector")
                            break

                if region_vec is None:
                    print(f"[RAG Pipeline] {region.region_name} - {topic} 벡터를 찾을 수 없습니다.")
                    continue

                # 시민 불만 요약
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
                top_policies = sorted(scored, key=lambda x: x[1], reverse=True)[:3]

                # 최종 정책 제안 생성
                prompt_final = (
                    f"'{region.region_name}'의 '{topic}' 관련 시민 불만:\n{citizen_summary}\n\n"
                    f"유사 정책 사례:\n"
                    + "\n".join([f"- {p[0]}: {p[2]}" for p in top_policies])
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

                # 결과 누적
                result_item = {
                    "region": region.region_name,
                    "topic": topic,
                    "citizen_summary": citizen_summary,
                    "policy_examples": [p[0] for p in top_policies],
                    "final_summary": final_summary,
                }
                results.append(result_item)

        # 4. 결과 저장
        output_dir = project_root / "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / "rag_pipeline_result.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"[RAG Pipeline] 완료 - 결과 저장: {output_path}")
        return {
            "status": "success",
            "count": len(results),
            "data": results,
            "saved_to": str(output_path),
            "updated_at": datetime.utcnow(),
        }

    except Exception as e:
        print(f"[RAG Pipeline] 오류 발생: {e}")
        return {"status": "error", "message": str(e)}
