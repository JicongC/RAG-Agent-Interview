from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.rerank_service import RerankService
from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf, rag_conf


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not item.get("question") or not item.get("expected_keywords"):
            raise ValueError(f"{path}:{line_number} 缺少 question 或 expected_keywords")
        cases.append(item)
    return cases


def doc_texts(docs) -> list[str]:
    return [(doc.page_content or "") for doc in docs]


def source_names(docs) -> list[str]:
    names = []
    for doc in docs:
        metadata = doc.metadata or {}
        source = metadata.get("source") or metadata.get("file_path") or "unknown"
        names.append(Path(str(source)).name)
    return names


def keyword_stats(text: str, keywords: list[str]) -> tuple[list[str], float, bool]:
    matched = [keyword for keyword in keywords if keyword.lower() in text.lower()]
    recall = len(matched) / len(keywords) if keywords else 0.0
    return matched, recall, len(matched) == len(keywords)


def evaluate_case(case: dict, retriever, reranker: RerankService | None, final_k: int) -> dict:
    question = case["question"]
    expected_keywords = case["expected_keywords"]

    vector_started_at = time.perf_counter()
    recall_docs = retriever.invoke(question)
    vector_elapsed_ms = (time.perf_counter() - vector_started_at) * 1000

    vector_top_docs = recall_docs[:final_k]
    recall_text = "\n".join(doc_texts(recall_docs))
    vector_top_text = "\n".join(doc_texts(vector_top_docs))

    recall_matched, recall_keyword_recall, recall_all_hit = keyword_stats(
        recall_text, expected_keywords
    )
    vector_top_matched, vector_top_keyword_recall, vector_top_all_hit = keyword_stats(
        vector_top_text, expected_keywords
    )

    rerank_docs = []
    rerank_status = "disabled"
    rerank_elapsed_ms = 0.0
    rerank_keyword_recall = None
    rerank_all_hit = None
    rerank_matched = []
    rerank_sources = []

    if reranker:
        rerank_started_at = time.perf_counter()
        rerank_docs, rerank_status = reranker.rerank_with_status(question, recall_docs)
        rerank_elapsed_ms = (time.perf_counter() - rerank_started_at) * 1000
        rerank_text = "\n".join(doc_texts(rerank_docs))
        rerank_matched, rerank_keyword_recall, rerank_all_hit = keyword_stats(
            rerank_text, expected_keywords
        )
        rerank_sources = source_names(rerank_docs)

    return {
        "id": case.get("id", ""),
        "question": question,
        "expected_keywords": "、".join(expected_keywords),
        "recall_count": len(recall_docs),
        "final_k": final_k,
        "vector_elapsed_ms": round(vector_elapsed_ms, 2),
        "rerank_elapsed_ms": round(rerank_elapsed_ms, 2),
        "rerank_status": rerank_status,
        "recall_keyword_recall": round(recall_keyword_recall, 4),
        "recall_all_hit": recall_all_hit,
        "recall_matched": "、".join(recall_matched),
        "vector_top_keyword_recall": round(vector_top_keyword_recall, 4),
        "vector_top_all_hit": vector_top_all_hit,
        "vector_top_matched": "、".join(vector_top_matched),
        "rerank_keyword_recall": (
            round(rerank_keyword_recall, 4) if rerank_keyword_recall is not None else ""
        ),
        "rerank_all_hit": "" if rerank_all_hit is None else rerank_all_hit,
        "rerank_matched": "、".join(rerank_matched),
        "vector_top_sources": " | ".join(source_names(vector_top_docs)),
        "rerank_sources": " | ".join(rerank_sources),
    }


def average(rows: list[dict], key: str):
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return round(mean(values), 4) if values else None


def hit_rate(rows: list[dict], key: str):
    values = [row[key] for row in rows if isinstance(row.get(key), bool)]
    return round(sum(values) / len(values), 4) if values else None


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 RAG 召回与 Rerank 命中效果")
    parser.add_argument("--cases", default="evaluation/rag_eval_set.jsonl")
    parser.add_argument("--output-dir", default="evaluation/reports")
    parser.add_argument("--disable-rerank", action="store_true")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.is_absolute():
        cases_path = PROJECT_ROOT / cases_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    cases = load_cases(cases_path)
    enable_rerank = bool(rag_conf.get("enable_rerank", False)) and not args.disable_rerank
    recall_k = int(rag_conf.get("rerank_recall_k", 12)) if enable_rerank else int(chroma_conf.get("k", 3))
    final_k = int(chroma_conf.get("k", 3))

    vector_store = VectorStoreService()
    retriever = vector_store.get_retriever(k=recall_k)
    reranker = RerankService() if enable_rerank else None

    rows = []
    for index, case in enumerate(cases, 1):
        row = evaluate_case(case, retriever, reranker, final_k)
        rows.append(row)
        print(
            f"[{index}/{len(cases)}] {case['question']} | "
            f"召回关键词覆盖={row['recall_keyword_recall']} | "
            f"向量Top-{final_k}={row['vector_top_keyword_recall']} | "
            f"RerankTop-{final_k}={row['rerank_keyword_recall']}"
        )

    summary = {
        "case_count": len(rows),
        "recall_k": recall_k,
        "final_k": final_k,
        "rerank_enabled": enable_rerank,
        "metrics": {
            "recall_keyword_recall_avg": average(rows, "recall_keyword_recall"),
            "recall_all_hit_rate": hit_rate(rows, "recall_all_hit"),
            "vector_top_keyword_recall_avg": average(rows, "vector_top_keyword_recall"),
            "vector_top_all_hit_rate": hit_rate(rows, "vector_top_all_hit"),
            "rerank_keyword_recall_avg": average(rows, "rerank_keyword_recall"),
            "rerank_all_hit_rate": hit_rate(rows, "rerank_all_hit"),
            "vector_elapsed_ms_avg": average(rows, "vector_elapsed_ms"),
            "rerank_elapsed_ms_avg": average(rows, "rerank_elapsed_ms"),
        },
    }

    write_csv(output_dir / "rag_eval_detail.csv", rows)
    (output_dir / "rag_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

