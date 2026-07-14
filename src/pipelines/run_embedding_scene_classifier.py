from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.clients.ollama_embedding_client import OllamaEmbeddingClient
from src.clients.ollama_client import OllamaClient
from src.utils.io import read_test_cases, write_csv


SCENE_DESCRIPTIONS = {
    "日常闲聊": (
        "用户没有明确任务，主要是轻松聊天、分享日常、吐槽小事、找话题、放松陪聊。"
        "重点是自然回应和延续聊天。"
    ),
    "情绪陪伴": (
        "用户主要表达难过、焦虑、失落、压力、孤独、烦躁、低落等情绪，"
        "需要被理解、安慰、陪伴或情绪支持。"
    ),
    "知识解释": (
        "用户主要询问概念、原因、区别、原理或事实说明，"
        "希望知道是什么、为什么、有什么不同。"
    ),
    "生活建议": (
        "用户提出具体生活问题，希望获得可直接执行的通用建议，"
        "例如作息、学习、饮食、收纳、效率、放松、日常安排。"
    ),
    "追问澄清": (
        "用户要求推荐、选择、规划、安排、方案或个性化建议，"
        "但缺少目标、预算、时间、地点、偏好、限制条件、当前水平等关键条件，"
        "需要先追问再回答。"
    ),
    "安全敏感": (
        "用户涉及自伤轻生、违法违规、侵犯隐私、危险行为、医疗用药、极端节食、"
        "投资买卖建议、制作危险物品或其他需要谨慎拒绝或安全引导的内容。"
    ),
}


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("vectors 不能为空")
    size = len(vectors[0])
    return [sum(vector[i] for vector in vectors) / len(vectors) for i in range(size)]


def embed_rows(
    client: OllamaEmbeddingClient,
    rows: list[dict[str, str]],
    *,
    batch_size: int,
    name: str,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    total = len(rows)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = rows[start:end]
        texts = [row["user_query"] for row in batch]
        result = client.embed(texts)
        embeddings.extend(l2_normalize(vector) for vector in result.embeddings)
        print(f"[embedding] {name} {end}/{total} latency={result.latency_ms}ms")
    return embeddings


def classify_by_centroid(
    reference_rows: list[dict[str, str]],
    reference_embeddings: list[list[float]],
    test_rows: list[dict[str, str]],
    test_embeddings: list[list[float]],
    *,
    model: str,
    rerank_client: OllamaClient | None = None,
    rerank_threshold: float = 0.0,
    rerank_candidates: int = 3,
) -> list[dict[str, object]]:
    scene_vectors: dict[str, list[list[float]]] = defaultdict(list)
    for row, embedding in zip(reference_rows, reference_embeddings):
        scene_vectors[row["scene"]].append(embedding)

    centroids = {
        scene: l2_normalize(mean_vector(vectors))
        for scene, vectors in scene_vectors.items()
    }

    results: list[dict[str, object]] = []
    for row, embedding in zip(test_rows, test_embeddings):
        scores = sorted(
            ((scene, cosine(embedding, centroid)) for scene, centroid in centroids.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        top1_scene, top1_score = scores[0]
        top2_scene, top2_score = scores[1] if len(scores) > 1 else ("", 0.0)
        candidate_scenes = [scene for scene, _ in scores[:rerank_candidates]]
        predicted_scene = top1_scene
        rerank_used = False
        rerank_reason = ""
        rerank_raw_response = ""
        margin = top1_score - top2_score

        if rerank_client is not None and margin <= rerank_threshold:
            rerank_used = True
            system_prompt = build_rerank_prompt(candidate_scenes)
            try:
                rerank_result = rerank_client.chat(
                    system_prompt=system_prompt,
                    user_query=row["user_query"],
                    temperature=0.0,
                    max_tokens=128,
                )
                rerank_raw_response = rerank_result.answer
                predicted_scene, rerank_reason = parse_rerank_response(
                    rerank_result.answer,
                    allowed_scenes=set(candidate_scenes),
                )
            except Exception as exc:
                predicted_scene = top1_scene
                rerank_reason = f"复判失败，保留 embedding 结果: {exc}"

        results.append(
            {
                "id": row["id"],
                "gold_scene": row["scene"],
                "user_query": row["user_query"],
                "predicted_scene": predicted_scene,
                "is_correct": predicted_scene == row["scene"],
                "top1_score": round(top1_score, 6),
                "top2_scene": top2_scene,
                "top2_score": round(top2_score, 6),
                "margin": round(margin, 6),
                "strategy": "centroid",
                "embedding_model": model,
                "nearest_id": "",
                "nearest_query": "",
                "candidate_scenes": "|".join(candidate_scenes),
                "rerank_used": rerank_used,
                "rerank_reason": rerank_reason,
                "rerank_raw_response": rerank_raw_response,
            }
        )
    return results


def extract_json_object(text: str, required_key: str | None = None) -> dict[str, object]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, object]] = []
    for match in re.finditer(r"\{", text.strip()):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and (required_key is None or required_key in value):
            candidates.append(value)
    if not candidates:
        raise ValueError(f"未找到有效 JSON 对象: {text}")
    return candidates[-1]


def build_rerank_prompt(candidate_scenes: list[str]) -> str:
    descriptions = "\n".join(
        f"- {scene}：{SCENE_DESCRIPTIONS[scene]}"
        for scene in candidate_scenes
    )
    return f"""你是语音助手场景复判器。

只能从候选场景中选择一个，不能输出候选之外的场景。

候选场景：
{descriptions}

判断要求：
1. 如果涉及自伤轻生、违法违规、侵犯隐私、危险物品、医疗用药、极端减肥、借贷风险、投资买卖建议、保证收益、内幕消息、炒币股票基金等风险，优先选择安全敏感。
2. 如果主要是表达难过、焦虑、低落、压力，优先选择情绪陪伴。
3. 如果是在问概念、原因、区别、原理，选择知识解释。
4. 如果是在要推荐、规划、选择、个性化方案，但缺少关键条件，选择追问澄清。
5. 如果可以直接给通用可执行建议，选择生活建议。
6. 如果只是轻松聊天、分享日常、找话题，选择日常闲聊。

只输出一行 JSON：
{{"scene":"场景名","confidence":0.0,"reason":"简短原因"}}"""


def parse_rerank_response(text: str, *, allowed_scenes: set[str]) -> tuple[str, str]:
    data = extract_json_object(text, required_key="scene")
    scene = str(data.get("scene", "")).strip()
    if scene not in allowed_scenes:
        raise ValueError(f"复判输出不在候选范围内: {scene}")
    reason = str(data.get("reason", "")).strip()
    return scene, reason


def build_description_rows(weight: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for scene, description in SCENE_DESCRIPTIONS.items():
        for index in range(weight):
            rows.append(
                {
                    "id": f"description_{scene}_{index + 1}",
                    "scene": scene,
                    "user_query": f"{scene}：{description}",
                }
            )
    return rows


def classify_by_knn(
    reference_rows: list[dict[str, str]],
    reference_embeddings: list[list[float]],
    test_rows: list[dict[str, str]],
    test_embeddings: list[list[float]],
    *,
    model: str,
    k: int,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for row, embedding in zip(test_rows, test_embeddings):
        neighbors = sorted(
            (
                (cosine(embedding, ref_embedding), ref_row)
                for ref_row, ref_embedding in zip(reference_rows, reference_embeddings)
            ),
            key=lambda item: item[0],
            reverse=True,
        )[:k]

        score_by_scene: dict[str, float] = defaultdict(float)
        count_by_scene: Counter[str] = Counter()
        for score, ref_row in neighbors:
            score_by_scene[ref_row["scene"]] += score
            count_by_scene[ref_row["scene"]] += 1

        scene_scores = sorted(
            score_by_scene.items(),
            key=lambda item: (count_by_scene[item[0]], item[1]),
            reverse=True,
        )
        top1_scene, top1_score = scene_scores[0]
        top2_scene, top2_score = scene_scores[1] if len(scene_scores) > 1 else ("", 0.0)
        nearest_score, nearest_row = neighbors[0]

        results.append(
            {
                "id": row["id"],
                "gold_scene": row["scene"],
                "user_query": row["user_query"],
                "predicted_scene": top1_scene,
                "is_correct": top1_scene == row["scene"],
                "top1_score": round(top1_score / k, 6),
                "top2_scene": top2_scene,
                "top2_score": round(top2_score / k, 6),
                "margin": round((top1_score - top2_score) / k, 6),
                "strategy": f"knn_k{k}",
                "embedding_model": model,
                "nearest_id": nearest_row["id"],
                "nearest_query": nearest_row["user_query"],
                "nearest_score": round(nearest_score, 6),
                "candidate_scenes": "",
                "rerank_used": False,
                "rerank_reason": "",
                "rerank_raw_response": "",
            }
        )
    return results


def print_summary(rows: list[dict[str, object]], *, show_wrong: bool) -> None:
    total = len(rows)
    correct = sum(1 for row in rows if row["is_correct"])
    print(f"\n总准确率：{correct}/{total} accuracy={correct / total:.3f}")

    by_scene: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["gold_scene"])].append(row)

    print("\n=== by scene ===")
    for scene, scene_rows in by_scene.items():
        scene_correct = sum(1 for row in scene_rows if row["is_correct"])
        print(f"{scene}: {scene_correct}/{len(scene_rows)} accuracy={scene_correct / len(scene_rows):.3f}")

    if show_wrong:
        wrong_rows = [row for row in rows if not row["is_correct"]]
        print("\n=== wrong cases ===")
        for row in wrong_rows:
            print(
                f"{row['id']} gold={row['gold_scene']} pred={row['predicted_scene']} "
                f"margin={row['margin']} query={row['user_query']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用本地 embedding 模型测试语音助手场景分类")
    parser.add_argument("--model", default="bge-m3", help="Ollama embedding 模型名称")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama 服务地址")
    parser.add_argument("--reference-cases", default="data/test_cases/scene_reference_centroid_v1.csv")
    parser.add_argument("--test-cases", default="data/test_cases/scene_test_generated_v2.csv")
    parser.add_argument("--output", default="outputs/classifier/embedding_scene_classifier_bge_m3_centroid.csv")
    parser.add_argument(
        "--strategy",
        choices=["centroid", "centroid-desc", "centroid-rerank", "knn"],
        default="centroid",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--description-weight", type=int, default=1)
    parser.add_argument("--rerank-model", default="qwen2.5:3b")
    parser.add_argument("--rerank-threshold", type=float, default=0.04)
    parser.add_argument("--rerank-candidates", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--show-wrong", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()

    reference_rows = read_test_cases(args.reference_cases)
    test_rows = read_test_cases(args.test_cases)
    if not reference_rows or not test_rows:
        raise RuntimeError("reference-cases 和 test-cases 都不能为空")
    if args.strategy == "knn" and args.k <= 0:
        raise RuntimeError("--k 必须大于 0")
    if args.strategy == "centroid-rerank" and args.rerank_candidates < 2:
        raise RuntimeError("--rerank-candidates 至少为 2")
    if args.description_weight <= 0:
        raise RuntimeError("--description-weight 必须大于 0")

    print(f"reference={args.reference_cases} rows={len(reference_rows)}")
    print(f"test={args.test_cases} rows={len(test_rows)}")
    print(f"model={args.model} strategy={args.strategy}")

    client = OllamaEmbeddingClient(model=args.model, base_url=args.base_url)
    if args.strategy == "centroid-desc":
        description_rows = build_description_rows(args.description_weight)
        reference_rows = reference_rows + description_rows
        print(f"description_rows={len(description_rows)} weight={args.description_weight}")

    reference_embeddings = embed_rows(client, reference_rows, batch_size=args.batch_size, name="reference")
    test_embeddings = embed_rows(client, test_rows, batch_size=args.batch_size, name="test")

    if args.strategy in {"centroid", "centroid-desc", "centroid-rerank"}:
        rerank_client = None
        if args.strategy == "centroid-rerank":
            rerank_client = OllamaClient(model=args.rerank_model, base_url=args.base_url)
            print(
                f"rerank_model={args.rerank_model} "
                f"threshold={args.rerank_threshold} candidates={args.rerank_candidates}"
            )
        result_rows = classify_by_centroid(
            reference_rows,
            reference_embeddings,
            test_rows,
            test_embeddings,
            model=args.model,
            rerank_client=rerank_client,
            rerank_threshold=args.rerank_threshold,
            rerank_candidates=args.rerank_candidates,
        )
        if args.strategy == "centroid-desc":
            for row in result_rows:
                row["strategy"] = f"centroid_desc_w{args.description_weight}"
        elif args.strategy == "centroid-rerank":
            for row in result_rows:
                row["strategy"] = f"centroid_rerank_t{args.rerank_threshold}"
    else:
        result_rows = classify_by_knn(
            reference_rows,
            reference_embeddings,
            test_rows,
            test_embeddings,
            model=args.model,
            k=args.k,
        )

    fieldnames = [
        "id",
        "gold_scene",
        "user_query",
        "predicted_scene",
        "is_correct",
        "top1_score",
        "top2_scene",
        "top2_score",
        "margin",
        "strategy",
        "embedding_model",
        "nearest_id",
        "nearest_query",
        "candidate_scenes",
        "rerank_used",
        "rerank_reason",
        "rerank_raw_response",
    ]
    if args.strategy == "knn":
        fieldnames.append("nearest_score")
    write_csv(Path(args.output), result_rows, fieldnames)
    print_summary(result_rows, show_wrong=args.show_wrong)
    print(f"\noutput={args.output}")
    print(f"elapsed_ms={int((time.perf_counter() - started_at) * 1000)}")


if __name__ == "__main__":
    main()
