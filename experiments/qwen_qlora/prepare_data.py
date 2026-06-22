from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable

import yaml

from experiment_paths import DEFAULT_CONFIG, project_path, resolve_config_path


SYSTEM_PROMPT = (
    "你是一位专业、严谨的中文技术面试辅导助手。回答应准确、结构清晰、"
    "突出核心结论，并在适合时补充项目例子、常见误区或面试追问点。"
)

INSTRUCTION_TEMPLATES = [
    "{question}",
    "请回答这道面试题：{question}",
    "面试官问：{question}",
    "技术面试中被问到“{question}”时，应如何回答？",
    "请说明：{question}",
    "请解释下面的问题：{question}",
    "请给出这道题的参考回答：{question}",
    "请作答：{question}",
    "候选人应如何回答这个问题：{question}",
    "下面是一道面试题，请回答：{question}",
    "请提供“{question}”的面试答案。",
    "针对以下问题给出回答：{question}",
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).strip("：:。！？!?")


def stable_id(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()[:16]


def parse_numbered_qa(text: str, source: str) -> list[dict]:
    text = text.replace("\r\n", "\n")
    question_pattern = re.compile(
        r"(?m)^\s*(\d+)[.、]\s*(.+?[？?])\s*(?:\n|$)"
    )
    matches = list(question_pattern.finditer(text))
    records: list[dict] = []
    for index, match in enumerate(matches):
        question = match.group(2).strip()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end():end].strip()
        answer_match = re.search(r"(?:答案|答)\s*[：:]\s*(.+)", block, flags=re.S)
        if not answer_match:
            continue
        answer = answer_match.group(1).strip()
        answer = re.split(r"\n\s*(?:---+|[一二三四五六七八九十]+[、.])", answer)[0].strip()
        if len(question) >= 4 and len(answer) >= 8:
            records.append(
                {
                    "instruction": question,
                    "output": answer,
                    "category": Path(source).stem,
                    "source": source,
                }
            )
    return records


def read_seed_records(source_files: Iterable[str], extra_jsonl: str | None) -> list[dict]:
    records: list[dict] = []
    for source in source_files:
        path = project_path(source)
        if not path.exists():
            print(f"[warning] source file not found: {path}")
            continue
        records.extend(parse_numbered_qa(path.read_text(encoding="utf-8"), source))

    if extra_jsonl:
        path = project_path(extra_jsonl)
        if path.exists():
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not item.get("instruction") or not item.get("output"):
                    raise ValueError(f"{path}:{line_number} 缺少 instruction/output")
                records.append(
                    {
                        "instruction": item["instruction"].strip(),
                        "output": item["output"].strip(),
                        "category": item.get("category", "extra"),
                        "source": str(path),
                    }
                )

    deduplicated: dict[str, dict] = {}
    for record in records:
        key = normalize(record["instruction"])
        if key and key not in deduplicated:
            record["group_id"] = stable_id(record["instruction"])
            deduplicated[key] = record
    return list(deduplicated.values())


def split_groups(records: list[dict], train_ratio: float, validation_ratio: float, seed: int):
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = round(total * train_ratio)
    validation_end = train_end + round(total * validation_ratio)
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def allocate_counts(target_size: int, split_sizes: dict[str, int]) -> dict[str, int]:
    total = sum(split_sizes.values())
    if total == 0:
        raise ValueError("没有可用的种子问答数据")
    raw = {name: target_size * size / total for name, size in split_sizes.items()}
    counts = {name: int(value) for name, value in raw.items()}
    for name, _ in sorted(raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True):
        if sum(counts.values()) >= target_size:
            break
        counts[name] += 1
    return counts


def augment(records: list[dict], target_count: int, seed: int) -> list[dict]:
    if not records or target_count <= 0:
        return []
    rng = random.Random(seed)
    output: list[dict] = []
    order = records[:]
    rng.shuffle(order)
    for index in range(target_count):
        source = order[index % len(order)]
        template_index = (index // len(order)) % len(INSTRUCTION_TEMPLATES)
        instruction = INSTRUCTION_TEMPLATES[template_index].format(
            question=source["instruction"]
        )
        output.append(
            {
                "id": f"{source['group_id']}-{index:04d}",
                "group_id": source["group_id"],
                "system": SYSTEM_PROMPT,
                "instruction": instruction,
                "output": source["output"],
                "category": source["category"],
                "source": source["source"],
            }
        )
    rng.shuffle(output)
    return output


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 Qwen QLoRA 面试问答数据")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config = load_config(resolve_config_path(args.config))
    data_config = config["data"]
    seed = int(config["seed"])
    records = read_seed_records(
        data_config["source_files"], data_config.get("extra_jsonl")
    )
    if len(records) < 30:
        raise ValueError(
            f"仅解析到 {len(records)} 条唯一问答，至少需要 30 条种子数据才能可靠划分。"
        )

    splits = split_groups(
        records,
        float(data_config["train_ratio"]),
        float(data_config["validation_ratio"]),
        seed,
    )
    targets = allocate_counts(
        int(data_config["target_size"]),
        {name: len(items) for name, items in splits.items()},
    )
    output_dir = project_path(data_config["output_dir"])
    manifest = {
        "seed": seed,
        "target_size": int(data_config["target_size"]),
        "unique_seed_questions": len(records),
        "split_strategy": "按原始问题 group_id 分组后划分，防止同题变体跨集合泄漏",
        "splits": {},
    }
    for offset, (name, items) in enumerate(splits.items()):
        augmented = augment(items, targets[name], seed + offset)
        write_jsonl(output_dir / f"{name}.jsonl", augmented)
        manifest["splits"][name] = {
            "seed_questions": len(items),
            "samples": len(augmented),
        }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
