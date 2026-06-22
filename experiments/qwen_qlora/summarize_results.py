from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import yaml

from experiment_paths import DEFAULT_CONFIG, project_path, resolve_config_path


SCORE_COLUMNS = [
    "correctness_1_5",
    "completeness_1_5",
    "structure_1_5",
    "interview_fit_1_5",
]


def average(rows: list[dict], key: str):
    values = []
    for row in rows:
        value = row.get(key, "")
        if value not in ("", None):
            values.append(float(value))
    return round(mean(values), 4) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 QLoRA 实验结果")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    with resolve_config_path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    result_dir = project_path(config["evaluation"]["output_dir"])
    with (result_dir / "human_review.csv").open("r", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    metrics_path = (
        project_path(config["training"]["output_dir"]) / "experiment_metrics.json"
    )
    training_metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else {}
    )

    summary = {
        "sample_count": len(rows),
        "automatic_metrics": {
            "base_char_f1": average(rows, "base_char_f1"),
            "finetuned_char_f1": average(rows, "finetuned_char_f1"),
            "base_rouge_l": average(rows, "base_rouge_l"),
            "finetuned_rouge_l": average(rows, "finetuned_rouge_l"),
        },
        "human_scores": {},
        "training": training_metrics,
    }
    for score in SCORE_COLUMNS:
        summary["human_scores"][score] = {
            "base": average(rows, f"base_{score}"),
            "finetuned": average(rows, f"finetuned_{score}"),
        }

    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
