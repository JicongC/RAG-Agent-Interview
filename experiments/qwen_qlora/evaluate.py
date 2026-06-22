from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from experiment_paths import DEFAULT_CONFIG, project_path, resolve_config_path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_unique_questions(samples: list[dict], max_samples: int) -> list[dict]:
    """每个原始问题只评测一个表达变体，避免重复题目放大指标。"""
    selected = []
    seen_groups = set()
    for sample in samples:
        group_id = sample.get("group_id", sample["id"])
        if group_id in seen_groups:
            continue
        seen_groups.add(group_id)
        selected.append(sample)
        if len(selected) >= max_samples:
            break
    return selected


def normalize_chars(text: str) -> list[str]:
    return list(re.sub(r"\s+", "", text).lower())


def lcs_length(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for item in left:
        current = [0]
        for index, other in enumerate(right, 1):
            current.append(
                previous[index - 1] + 1
                if item == other
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def char_f1(prediction: str, reference: str) -> float:
    from collections import Counter

    prediction_chars = normalize_chars(prediction)
    reference_chars = normalize_chars(reference)
    if not prediction_chars or not reference_chars:
        return 0.0
    overlap = sum((Counter(prediction_chars) & Counter(reference_chars)).values())
    precision = overlap / len(prediction_chars)
    recall = overlap / len(reference_chars)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def rouge_l_f1(prediction: str, reference: str) -> float:
    prediction_chars = normalize_chars(prediction)
    reference_chars = normalize_chars(reference)
    if not prediction_chars or not reference_chars:
        return 0.0
    lcs = lcs_length(prediction_chars, reference_chars)
    precision = lcs / len(prediction_chars)
    recall = lcs / len(reference_chars)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0


def load_model(model_name: str, adapter_path: str | None):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    tokenizer_path = adapter_path if adapter_path else model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, sample: dict, evaluation_config: dict) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": sample["system"]},
        {"role": "user", "content": sample["instruction"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    started_at = time.perf_counter()
    do_sample = float(evaluation_config["temperature"]) > 0
    generation_kwargs = {
        "max_new_tokens": int(evaluation_config["max_new_tokens"]),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": float(evaluation_config["temperature"]),
                "top_p": float(evaluation_config["top_p"]),
            }
        )
    output = model.generate(**inputs, **generation_kwargs)
    elapsed = time.perf_counter() - started_at
    generated_ids = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip(), elapsed


def write_review_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "id",
        "instruction",
        "reference",
        "base_answer",
        "finetuned_answer",
        "base_char_f1",
        "finetuned_char_f1",
        "base_rouge_l",
        "finetuned_rouge_l",
        "base_correctness_1_5",
        "finetuned_correctness_1_5",
        "base_completeness_1_5",
        "finetuned_completeness_1_5",
        "base_structure_1_5",
        "finetuned_structure_1_5",
        "base_interview_fit_1_5",
        "finetuned_interview_fit_1_5",
        "reviewer_notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="对比 Qwen 微调前后的面试回答质量")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--skip-base", action="store_true", help="只评测微调模型")
    args = parser.parse_args()
    config = load_config(resolve_config_path(args.config))
    evaluation_config = config["evaluation"]
    test_path = project_path(config["data"]["output_dir"]) / "test.jsonl"
    samples = select_unique_questions(
        load_jsonl(test_path), int(evaluation_config["max_samples"])
    )
    output_dir = project_path(evaluation_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = str(project_path(config["training"]["output_dir"]) / "adapter")
    base_model_name = config["model"]["name_or_path"]
    rows: list[dict] = []

    base_outputs = {}
    if not args.skip_base:
        print(f"[1/2] 正在评测基座模型，共 {len(samples)} 个独立测试问题。")
        base_model, base_tokenizer = load_model(base_model_name, None)
        for index, sample in enumerate(samples, 1):
            answer, elapsed = generate(base_model, base_tokenizer, sample, evaluation_config)
            base_outputs[sample["id"]] = {"answer": answer, "seconds": elapsed}
            print(
                f"  基座模型 [{index}/{len(samples)}] "
                f"{elapsed:.1f}s：{sample['instruction'][:28]}"
            )
        del base_model
        torch.cuda.empty_cache()

    print(f"[2/2] 正在评测微调模型，共 {len(samples)} 个独立测试问题。")
    tuned_model, tuned_tokenizer = load_model(base_model_name, adapter_path)
    for index, sample in enumerate(samples, 1):
        tuned_answer, tuned_elapsed = generate(
            tuned_model, tuned_tokenizer, sample, evaluation_config
        )
        base_answer = base_outputs.get(sample["id"], {}).get("answer", "")
        rows.append(
            {
                "id": sample["id"],
                "instruction": sample["instruction"],
                "reference": sample["output"],
                "base_answer": base_answer,
                "finetuned_answer": tuned_answer,
                "base_char_f1": char_f1(base_answer, sample["output"]) if base_answer else "",
                "finetuned_char_f1": char_f1(tuned_answer, sample["output"]),
                "base_rouge_l": rouge_l_f1(base_answer, sample["output"]) if base_answer else "",
                "finetuned_rouge_l": rouge_l_f1(tuned_answer, sample["output"]),
                "base_seconds": base_outputs.get(sample["id"], {}).get("seconds", ""),
                "finetuned_seconds": tuned_elapsed,
            }
        )
        print(
            f"  微调模型 [{index}/{len(samples)}] "
            f"{tuned_elapsed:.1f}s：{sample['instruction'][:28]}"
        )

    (output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    write_review_csv(output_dir / "human_review.csv", rows)
    print(f"已生成 {len(rows)} 组回答：{output_dir / 'predictions.jsonl'}")
    print(f"人工盲评表：{output_dir / 'human_review.csv'}")


if __name__ == "__main__":
    main()
