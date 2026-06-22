from __future__ import annotations

import argparse
import json
import os
import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from experiment_paths import DEFAULT_CONFIG, project_path, resolve_config_path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


@dataclass
class GpuSnapshot:
    name: str = "unavailable"
    total_mib: float = 0.0
    peak_used_mib: float = 0.0
    peak_torch_allocated_mib: float = 0.0
    peak_torch_reserved_mib: float = 0.0


class GpuMonitor:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.snapshot = GpuSnapshot()
        self._pynvml = None
        self._handle = None

    def start(self) -> None:
        if not torch.cuda.is_available():
            return
        self.snapshot.name = torch.cuda.get_device_name(0)
        self.snapshot.total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
        torch.cuda.reset_peak_memory_stats(0)
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._pynvml = None
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def _poll(self) -> None:
        while not self.stop_event.wait(self.interval):
            if self._pynvml and self._handle:
                info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self.snapshot.peak_used_mib = max(
                    self.snapshot.peak_used_mib, info.used / 2**20
                )

    def stop(self) -> GpuSnapshot:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=self.interval * 2)
        if torch.cuda.is_available():
            self.snapshot.peak_torch_allocated_mib = (
                torch.cuda.max_memory_allocated(0) / 2**20
            )
            self.snapshot.peak_torch_reserved_mib = (
                torch.cuda.max_memory_reserved(0) / 2**20
            )
        if self._pynvml:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        return self.snapshot


def build_preprocess(tokenizer, max_length: int):
    def preprocess(example):
        prompt_messages = [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["instruction"]},
        ]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": example["output"]}
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tokenizer(
            prompt_text, add_special_tokens=False, truncation=True, max_length=max_length
        )["input_ids"]
        encoded = tokenizer(
            full_text, add_special_tokens=False, truncation=True, max_length=max_length
        )
        labels = encoded["input_ids"][:]
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        encoded["labels"] = labels
        return encoded

    return preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 QLoRA 微调 Qwen 面试问答模型")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    set_seed(int(config["seed"]))

    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA 训练需要 CUDA 版 PyTorch，但当前环境未启用 CUDA。\n"
            f"检测结果：torch={torch.__version__}, "
            f"torch.version.cuda={torch.version.cuda!r}。\n"
            "如果版本号包含 '+cpu' 或 CUDA 为 None，请卸载 CPU 版 torch，"
            "再从 PyTorch 官方 CUDA wheel 源安装 GPU 版。"
        )

    model_config = config["model"]
    train_config = config["training"]
    qlora_config = config["qlora"]
    data_dir = project_path(config["data"]["output_dir"])
    output_dir = project_path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = torch.bfloat16 if train_config.get("bf16", True) else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(train_config.get("gradient_checkpointing", True)),
    )
    lora = LoraConfig(
        r=int(qlora_config["rank"]),
        lora_alpha=int(qlora_config["alpha"]),
        lora_dropout=float(qlora_config["dropout"]),
        target_modules=list(qlora_config["target_modules"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "validation.jsonl"),
        },
    )
    preprocess = build_preprocess(tokenizer, int(model_config["max_length"]))
    tokenized = dataset.map(
        preprocess,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing interview data",
    )

    bf16_enabled = bool(train_config.get("bf16", True)) and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(train_config["epochs"]),
        learning_rate=float(train_config["learning_rate"]),
        per_device_train_batch_size=int(train_config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_config["gradient_accumulation_steps"]),
        warmup_ratio=float(train_config["warmup_ratio"]),
        weight_decay=float(train_config["weight_decay"]),
        logging_steps=int(train_config["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(train_config["eval_steps"]),
        save_strategy="steps",
        save_steps=int(train_config["save_steps"]),
        save_total_limit=int(train_config["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=bool(train_config.get("gradient_checkpointing", True)),
        bf16=bf16_enabled,
        fp16=bool(train_config.get("fp16", False)) and not bf16_enabled,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        seed=int(config["seed"]),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100
        ),
    )

    monitor = GpuMonitor()
    monitor.start()
    started_at = time.time()
    try:
        train_result = trainer.train()
        eval_metrics = trainer.evaluate()
        trainer.save_model(str(output_dir / "adapter"))
        tokenizer.save_pretrained(str(output_dir / "adapter"))
    finally:
        gpu = monitor.stop()
    elapsed = time.time() - started_at

    metrics = {
        "experiment": {
            "base_model": model_config["name_or_path"],
            "lora_rank": int(qlora_config["rank"]),
            "lora_alpha": int(qlora_config["alpha"]),
            "learning_rate": float(train_config["learning_rate"]),
            "epochs": float(train_config["epochs"]),
            "effective_batch_size": (
                int(train_config["per_device_train_batch_size"])
                * int(train_config["gradient_accumulation_steps"])
                * int(os.environ.get("WORLD_SIZE", "1"))
            ),
        },
        "runtime": {
            "wall_clock_seconds": elapsed,
            "wall_clock_minutes": elapsed / 60,
            "trainer_metrics": train_result.metrics if "train_result" in locals() else {},
            "evaluation_metrics": eval_metrics if "eval_metrics" in locals() else {},
        },
        "hardware": {
            "gpu_name": gpu.name,
            "gpu_total_mib": gpu.total_mib,
            "peak_gpu_used_mib": gpu.peak_used_mib,
            "peak_torch_allocated_mib": gpu.peak_torch_allocated_mib,
            "peak_torch_reserved_mib": gpu.peak_torch_reserved_mib,
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "python_platform": platform.platform(),
        },
        "config_file": str(config_path),
    }
    (output_dir / "experiment_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
