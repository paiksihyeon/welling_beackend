import json
import os
import numpy as np
from collections import defaultdict

BASE_PATH = "app/files"


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
    """지역별 여론 벡터 로드 (예: 서울_vectors.json)"""
    file_path = os.path.join(BASE_PATH, f"{region_name}_vectors_e5.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ {region_name}_vectors.json 파일이 없습니다: {file_path}")
    print(f"[vector_service] ✅ 지역 벡터 로드 완료: {file_path}")
    return load_json(file_path)


# -------------------------------
# ✅ 주제별 평균 벡터 집계
# -------------------------------
def aggregate_topic_vectors(region_vectors):
    """
    주제(topic)별로 여러 의견 벡터를 평균 벡터로 집계.
    JSON이 list 또는 dict 어떤 형태든 대응.
    """
    if isinstance(region_vectors, dict):
        # 이미 주제별로 집계된 경우 그대로 반환
        return region_vectors

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
            "gap_score": float(len(vecs))  # 단순히 샘플 수 기반 가중치
        }

    return topic_avg_vectors


# -------------------------------
# ✅ 갭이 큰 주제 찾기
# -------------------------------
def find_top_gap_topics(region_vectors, top_k: int = 3):
    """
    갭 스코어가 큰 주제 찾기
    - JSON이 list 형태인 경우 label 기반 계산
    - JSON이 dict 형태인 경우 gap_score 기반 계산
    """
    if isinstance(region_vectors, dict):
        topic_gaps = {
            topic: data.get("gap_score", 0)
            for topic, data in region_vectors.items()
            if isinstance(data, dict)
        }
    else:
        topic_stats = defaultdict(lambda: {"pos": 0, "neg": 0})
        for item in region_vectors:
            topic = item.get("topic")
            label = item.get("label")
            if topic and label is not None:
                if label > 0:
                    topic_stats[topic]["pos"] += 1
                elif label < 0:
                    topic_stats[topic]["neg"] += 1

        topic_gaps = {
            t: stats["neg"] - stats["pos"] for t, stats in topic_stats.items()
        }

    sorted_topics = sorted(topic_gaps.items(), key=lambda x: x[1], reverse=True)
    return [t[0] for t in sorted_topics[:top_k]]


# -------------------------------
# ✅ 정책 벡터 유사도 계산
# -------------------------------
def find_similar_policies(region_name: str, topic: str, top_k: int = 3):
    region_vectors_raw = load_region_vectors(region_name)

    # 형태 변환
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
