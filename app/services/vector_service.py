import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict

BASE_PATH = "app/files"
GAP_CSV_PATH = os.path.join(BASE_PATH, "gap_score.csv")

# ✅ 표준 주제 매핑 (CSV 컬럼명 → (한글명, 영문명))
topic_map = {
    "gap_transport_infra": ("교통인프라", "transport_infra"),
    "gap_labor_economy": ("노동경제", "labor_economy"),
    "gap_healthcare": ("의료보건", "healthcare"),
    "gap_policy_efficiency": ("정책효능감", "policy_efficiency"),
    "gap_housing_environment": ("주거환경", "housing_environment"),
}


# -------------------------------
# ✅ 기본 유틸
# -------------------------------
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(v1, v2):
    """코사인 유사도 계산"""
    v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom != 0 else 0.0


# -------------------------------
# ✅ 데이터 로드
# -------------------------------
def load_policy_vectors():
    """정책 제안 문서 벡터 로드"""
    file_path = os.path.join(BASE_PATH, "policy_vectors.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ policy_vectors.json 파일이 없습니다: {file_path}")
    print(f"[vector_service] ✅ 정책 벡터 로드 완료: {file_path}")
    return load_json(file_path)


def load_region_vectors(region_name: str):
    """지역별 여론 벡터 로드 (예: 서울_vectors_e5.json)"""
    file_path = os.path.join(BASE_PATH, f"{region_name}_vectors_e5.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ {region_name}_vectors_e5.json 파일이 없습니다: {file_path}")
    print(f"[vector_service] ✅ 지역 벡터 로드 완료: {file_path}")
    return load_json(file_path)


# -------------------------------
# ✅ 주제별 평균 벡터 집계
# -------------------------------
def aggregate_topic_vectors(region_vectors):
    """JSON이 list 또는 dict 형태든 대응하여 평균 벡터 집계"""
    if isinstance(region_vectors, dict):
        return region_vectors  # 이미 주제별로 집계된 경우 그대로 반환

    topic_dict = defaultdict(list)
    for item in region_vectors:
        topic = item.get("topic")
        vec = item.get("vector")
        if topic and vec:
            topic_dict[topic].append(np.array(vec, dtype=float))

    topic_avg_vectors = {}
    for topic, vecs in topic_dict.items():
        topic_avg_vectors[topic] = {
            "vector": np.mean(vecs, axis=0).tolist(),
            "gap_score": float(len(vecs))
        }

    return topic_avg_vectors


# -------------------------------
# ✅ 갭이 큰 주제 찾기 (CSV 기반)
# -------------------------------
def find_top_gap_topics(region_vectors=None, region_name: str = None, top_k: int = 3):
    """
    ✅ app/files/gap_score.csv에서 지역별 gap 값을 불러와
       표준 topic_map을 기준으로 상위 K개 주제 반환
    """
    if not region_name:
        raise ValueError("⚠️ region_name이 필요합니다.")
    if not os.path.exists(GAP_CSV_PATH):
        raise FileNotFoundError(f"⚠️ gap_score.csv 파일이 없습니다: {GAP_CSV_PATH}")

    df = pd.read_csv(GAP_CSV_PATH)
    if "region" not in df.columns:
        raise ValueError("⚠️ gap_score.csv에 'region' 컬럼이 없습니다.")

    # ✅ 지역 행 찾기
    region_row = df[df["region"] == region_name]
    if region_row.empty:
        raise ValueError(f"⚠️ {region_name} 지역 데이터가 gap_score.csv에 없습니다.")
    region_row = region_row.iloc[0]

    # ✅ topic_map 기준으로 gap 데이터 구성
    topic_info = []
    for csv_col, (topic_kr, topic_en) in topic_map.items():
        if csv_col not in df.columns:
            print(f"⚠️ CSV에 {csv_col} 컬럼이 없습니다. 건너뜀.")
            continue

        gap_value = float(region_row[csv_col])
        topic_info.append({
            "topic": topic_kr,
            "topic_en": topic_en,
            "gap": round(gap_value, 2),
            # 필요 시 확장 가능
            "policy_score": None,
            "sentiment_score": None
        })

    # ✅ gap 기준 정렬 및 상위 K개 반환
    topic_info = sorted(topic_info, key=lambda x: x["gap"], reverse=True)[:top_k]

    return topic_info


# -------------------------------
# ✅ 정책 벡터 유사도 계산
# -------------------------------
def find_similar_policies(region_name: str, topic: str, top_k: int = 3):
    """특정 지역의 특정 주제 벡터와 정책 문서 간 유사도 계산"""
    region_vectors_raw = load_region_vectors(region_name)

    region_vectors = (
        aggregate_topic_vectors(region_vectors_raw)
        if isinstance(region_vectors_raw, list)
        else region_vectors_raw
    )

    if topic not in region_vectors:
        raise ValueError(f"{region_name} 지역 데이터에 '{topic}' 주제가 없습니다.")

    topic_vec = np.array(region_vectors[topic]["vector"], dtype=float)
    topic_vec /= np.linalg.norm(topic_vec) + 1e-8

    policy_vectors = load_policy_vectors()
    similarities = []

    for policy_topic, vec in policy_vectors.items():
        arr = np.array(vec, dtype=float)
        arr /= np.linalg.norm(arr) + 1e-8
        score = float(np.dot(topic_vec, arr))
        similarities.append((policy_topic, score))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
