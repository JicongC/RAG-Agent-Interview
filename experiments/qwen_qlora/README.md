# Qwen 面试问答 QLoRA 微调实验

本实验使用小型中文指令模型 `Qwen/Qwen2.5-1.5B-Instruct`，在面试问答数据上进行 4-bit QLoRA 微调，并对比微调前后的回答质量、显存占用与训练时间。

> 仓库当前提供完整的实验代码、配置和 1,200 条处理后数据。未实际完成训练时，请将其表述为“实现了 QLoRA 微调实验流程”，不要填写未经实测的质量提升、显存占用或训练时间。

## 1. 实验设置

| 项目 | 默认值 |
|---|---|
| 基座模型 | Qwen2.5-1.5B-Instruct |
| 指令数据量 | 1,200 条 |
| 数据划分 | 训练集 80% / 验证集 10% / 测试集 10% |
| 量化 | NF4 4-bit + Double Quant |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| 学习率 | 2e-4 |
| Epoch | 3 |
| 最大序列长度 | 1,024 |
| 有效 Batch Size | 2 × 8 = 16（单卡） |

所有参数均集中在 [config.yaml](./config.yaml)，实际训练参数会写入 `experiment_metrics.json`。

## 2. 数据准备

数据脚本会解析项目已有的：

- `data/hr_questions.txt`
- `data/tech_knowledge.txt`
- `data/raw/interview_qa.jsonl` 中的补充样本

然后执行问题级去重、模板化指令增强和分组划分。划分发生在增强之前，同一原始问题的不同表达只会出现在一个集合中，避免训练/测试泄漏。

补充数据采用 JSONL，每行格式如下：

```json
{"instruction":"什么是 Redis 缓存击穿？","output":"缓存击穿是……","category":"redis"}
```

生成默认 1,200 条数据：

```powershell
# 在项目根目录运行
python experiments/qwen_qlora/prepare_data.py
```

也可以进入实验目录运行，脚本不会再依赖当前终端位置：

```powershell
cd experiments/qwen_qlora
python prepare_data.py
```

产物：

```text
data/processed/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── manifest.json
```

`manifest.json` 记录种子问题数和三个集合的实际样本数。若希望使用 1,000～3,000 之间的其他规模，修改 `config.yaml` 中的 `target_size`。

> 注意：模板增强用于统一回答风格，不能替代真实、多样且经过审核的面试问答。正式实验建议继续向 `interview_qa.jsonl` 添加真实数据，并检查事实准确性、隐私和版权。

## 3. 环境与训练

建议使用 Linux/WSL2、Python 3.10 或 3.11、支持 CUDA 的 NVIDIA 显卡。Windows + RTX 4060 可先安装 CUDA 12.8 版 PyTorch：

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

验证 PyTorch 已识别显卡：

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

其中 `torch.cuda.is_available()` 应输出 `True`，版本号不能包含 `+cpu`。然后安装其余实验依赖：

```powershell
pip install -r experiments/qwen_qlora/requirements.txt
```

开始训练：

```powershell
python experiments/qwen_qlora/train_qlora.py
```

模型适配器和实验记录位于：

```text
outputs/qwen2.5-1.5b-qlora/
├── adapter/
└── experiment_metrics.json
```

`experiment_metrics.json` 自动记录：

- LoRA Rank、Alpha、学习率、Epoch、有效 Batch Size
- GPU 型号和总显存
- NVML 观测到的进程训练期峰值显存
- PyTorch 峰值 allocated/reserved 显存
- Trainer 训练时间与墙钟时间
- 训练损失和验证损失

显存不足时，优先将 `per_device_train_batch_size` 改为 1，或把 `max_length` 改为 768/512；保持梯度累积后有效 Batch Size 接近 16。

## 4. 微调前后质量对比

运行基座模型和 QLoRA 模型的成对测试：

```powershell
python experiments/qwen_qlora/evaluate.py
```

输出：

- `results/predictions.jsonl`：参考答案、微调前回答、微调后回答、生成耗时和自动指标
- `results/human_review.csv`：人工评分表

评测脚本按 `group_id` 去重，每个未见过的原始问题只保留一个表达变体，避免重复题目虚高指标。默认数据下约有 14 个独立测试问题；正式报告建议扩充真实种子问答，使测试集至少包含 50～100 个独立问题。

自动指标包括字符级 F1 与 ROUGE-L，仅用于辅助观察。开放式面试答案可能存在多种正确表达，因此最终结论应以人工评审为主。若用于论文或正式报告，建议另行隐藏模型身份并随机化回答顺序，进行盲评。

人工评分建议由至少 2 名评审独立完成，每项 1～5 分：

| 维度 | 评分重点 |
|---|---|
| 正确性 | 是否存在事实错误、概念混淆 |
| 完整性 | 是否覆盖问题关键点 |
| 结构性 | 结论是否清晰，层次是否合理 |
| 面试适用性 | 是否适合口头回答，是否有例子、误区或追问意识 |

填写 `human_review.csv` 后汇总：

```powershell
python experiments/qwen_qlora/summarize_results.py
```

汇总结果写入 `results/summary.json`。

## 5. 实验报告填写模板

| 指标 | 微调前 | 微调后 | 变化 |
|---|---:|---:|---:|
| 字符 F1 | 待运行 | 待运行 | 待运行 |
| ROUGE-L | 待运行 | 待运行 | 待运行 |
| 人工正确性（1～5） | 待评审 | 待评审 | 待评审 |
| 人工完整性（1～5） | 待评审 | 待评审 | 待评审 |
| 人工结构性（1～5） | 待评审 | 待评审 | 待评审 |
| 面试适用性（1～5） | 待评审 | 待评审 | 待评审 |

| 资源项 | 实测值 |
|---|---|
| 显卡型号 | 由脚本记录 |
| 显卡总显存 | 由脚本记录 |
| 峰值显存占用 | 由脚本记录 |
| 训练时间 | 由脚本记录 |
| 最优验证集 Loss | 由脚本记录 |

报告结论应同时说明优势和边界，例如：微调是否让回答更结构化、更贴近面试表达；是否出现答案模板化、过拟合或事实能力下降；测试集是否足以覆盖 Java、数据库、Redis、系统设计和行为面试等类别。
