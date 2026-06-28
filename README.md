# 基于 RAG 与 Agent 的智能面试辅导系统

> 一个面向求职面试场景的智能辅导系统，融合 **RAG（检索增强生成）** 与 **Agent（工具调用）**，支持模拟面试、知识问答、用户状态保存与面试报告生成。

## 项目亮点

- **双模式交互**：支持 `问答模式` 和 `模拟面试模式`
- **RAG 知识增强**：从本地知识库检索相关内容，提高回答的准确性与专业性
- **Agent 工具调用**：集成城市定位、天气查询、用户信息读取等工具能力
- **面试报告生成**：自动汇总对话过程并输出复盘报告
- **用户状态持久化**：按用户 ID 保存历史记录，支持继续上次会话
- **Streamlit 快速展示**：界面简洁，适合演示和原型展示

## 在线效果

本项目适合用于：

- 面试前知识复习
- 模拟真实面试提问
- 生成面试复盘报告
- 展示 RAG + Agent 的应用实践

## 功能介绍

### 1. 问答模式
用户可以直接输入问题，系统会先检索知识库，再结合大模型生成回答，适合快速查阅知识点或岗位面试内容。

### 2. 模拟面试模式
系统会扮演面试官，根据知识库内容连续提问，用户回答后系统继续追问，用于训练表达能力和应答节奏。

### 3. 面试报告
面试结束后，系统可根据完整对话记录和知识库参考内容生成面试报告，便于用户复盘与改进。

### 4. 工具调用
系统支持以下工具能力：
- 获取当前城市
- 查询天气
- 获取当前用户 ID
- 获取当前月份

## 技术架构

```text
用户输入
   ↓
Streamlit 前端
   ↓
InterviewAssistantService
   ├─ Agent 工具调用
   ├─ RAG 检索增强
   └─ 面试报告生成
   ↓
向量数据库 / 外部工具 / 用户状态存储
```

## 项目结构

```text
.
├─ app.py                          # Streamlit 主入口（扁平版界面）
├─ agent/
│  ├─ agent_tools.py               # Agent 工具函数
│  └─ interview_assistant_service.py # 面试助手核心服务
├─ rag/
│  ├─ rag_service.py               # RAG 检索与总结服务
│  ├─ vector_store.py              # 向量库构建与检索
│  └─ rerank_service.py            # 重排序服务
├─ utils/
│  ├─ user_history_store.py        # 用户状态持久化
│  ├─ prompt_loader.py             # Prompt 加载
│  └─ ...
├─ config/
│  ├─ agent.yml                    # Agent 与外部接口配置
│  ├─ rag.yml                      # 向量库与模型配置
│  └─ prompts.yml                  # 提示词配置
├─ data/
│  ├─ user_histories/              # 用户会话持久化数据
│  └─ ...                          # 知识库文档等
├─ README.md
├─ LICENSE
└─ requirements.txt
```

## 技术栈

- **Python**
- **Streamlit**
- **LangChain / LangGraph**
- **ChromaDB**
- **RAG**
- **大模型与嵌入模型**
- **高德开放平台 API**

## 快速开始

### 1. 克隆仓库
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. 创建虚拟环境（推荐）
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置 API 与模型参数
请根据你的实际环境修改：

#### `config/agent.yml`
- `amap_key`：高德地图 API Key
- `external_data_path`：外部用户记录文件路径

#### `config/rag.yml`
- `chat_model_name`：对话模型名称
- `embedding_model_name`：嵌入模型名称
- `rerank_model_name`：重排序模型名称

如果你使用的是本地模型或其他云服务，还需要补充相应的环境变量或连接配置。

### 5. 启动项目
```bash
streamlit run app.py
```

如果你想使用其它入口，也可以：

```bash
streamlit run streamlit_app.py
```

或

```bash
streamlit run streamlit_app_flat.py
```

## 使用说明

### 第一步：加载知识库
首次运行建议在左侧点击 **加载/更新知识库**，系统会扫描 `data` 中符合条件的文档并构建向量索引。

也可以在 **知识库管理** 页面通过 **上传知识文件（txt/pdf）** 直接添加新的知识文档。上传文件会保存到 `data/uploads/`，随后点击 **保存上传文件并更新知识库** 即可写入向量库。该目录已加入 `.gitignore`，避免把个人资料或临时文档提交到 GitHub。

知识库更新采用文件级增量策略：系统会记录每个文件的 MD5、片段数和来源路径。文件未变化时跳过；文件内容变化时先删除该文件旧向量，再重新切分入库；文件从知识库目录移除后，也会清理对应旧向量。

**知识库管理** 页面还可查看当前已入库文件、chunk 数、MD5 摘要和文件状态。对于 `data/uploads/` 中的上传文件，可直接在页面删除原文件并同步清理向量索引；对于项目内置知识文件，页面只移除索引，保留原文件，避免误删项目数据。

### 第二步：切换用户
输入用户 ID 后点击 **切换/加载用户**，系统会自动恢复该用户的历史对话、面试状态与报告内容。

### 第三步：选择模式
- **问答模式**：适合单轮问题解答
- **模拟面试模式**：适合连续追问与面试训练

### 第四步：生成报告
在模拟面试结束后，可勾选生成报告并一键生成本次面试总结。

## 配置说明

### `config/agent.yml`
- `external_data_path`：外部数据文件路径
- `amap_key`：高德 API Key
- `amap_ip_api`：IP 定位接口
- `amap_weather_api`：天气查询接口

### `config/rag.yml`
- `chat_model_name`：对话模型名称
- `embedding_model_name`：向量嵌入模型名称
- `enable_rerank`：是否启用重排序
- `rerank_model_name`：重排序模型名称
- `rerank_recall_k`：召回数量

## 数据说明
-本项目所用到的API KEY均已设置为环境变量
- `data/user_histories/`：保存用户状态数据
- `data/` 下的知识库文件：用于 RAG 检索
- 向量库与索引目录：由配置文件中的 `persist_directory` 决定

## 后续扩展方向

- 增加语音输入与语音播报
- 加入简历解析与岗位匹配
- 支持更多岗位题库
- 增加面试评分维度
- 扩展图片/截图等多模态输入能力

## 免责声明

本项目仅用于学习、研究与演示。实际部署与使用时，请根据自身环境配置 API Key、模型服务及数据安全策略。

## Qwen QLoRA 微调实验

项目已补充基于 `Qwen2.5-1.5B-Instruct` 的面试问答 QLoRA 实验模块，包含 1,200 条指令数据准备、训练/验证/测试集分组划分、4-bit NF4 量化训练、微调前后回答评测，以及 GPU 显存和训练耗时记录。

- [QLoRA 实验说明](experiments/qwen_qlora/README.md)
- [项目经历与简历描述](docs/项目经历描述.md)

## RAG 检索评测

项目提供轻量级 RAG 检索评测脚本，可对比向量召回 Top-K、向量 Top-3 与 Rerank Top-3 的关键词覆盖情况和耗时：

```bash
python evaluation/rag_eval.py
```

评测集位于 `evaluation/rag_eval_set.jsonl`，报告输出到 `evaluation/reports/`。
