# 多模态个人知识库 RAG 问答系统

一个面向个人资料管理的本地 RAG 工程项目，支持 **PDF、Markdown、图片、扫描页**。从解析和索引到混合检索、重排、问答、引用核查、资料更新与评测，保留可复核的来源与执行记录。

## 能力与运行模式

| 能力 | 默认模式 | 可选增强 |
|---|---|---|
| 文档解析 | PyMuPDF 原生 PDF、Markdown 结构解析 | 扫描页与图片使用真实 RapidOCR |
| 视觉理解 | 未配置时标记 `not_configured` | OpenAI 兼容 VLM，直接传入图片 |
| 向量索引 | 真实 TF-IDF 向量写入 Chroma | 本地 SentenceTransformer 或远程 Embedding |
| 关键词召回 | 中文分词 + BM25 | 可替换分词器 |
| 融合与重排 | RRF + 词汇覆盖重排 | 本地 CrossEncoder 神经重排、可选 MMR |
| 回答 | 原文抽取 + 真实引用 | 模型生成 + 引用编号检查 + 支持性复核 |
| 更新 | 文档 Hash、有效版本切换、缓存失效 | 语义 Embedding 按内容缓存复用 |
| 评测 | 固定测试集的 Hit Rate、MRR、Recall | RAGAS 忠实度、相关性与上下文精度 |

**TF-IDF 是词汇统计基线，不是语义 Embedding；词汇重排也不是神经 Reranker。** 两种模式使用同一套存储和问答流程，页面、报告中明确标记所用后端。项目不使用 Hash 随机向量代替语义模型，也不生成模拟 OCR、VLM 或 RAGAS 分数。

## 快速启动

要求 Python 3.11+。以下命令在项目根目录执行。

Windows：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
rag-workbench demo
rag-workbench serve
```

Linux / macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
rag-workbench demo
rag-workbench serve
```

打开 **http://127.0.0.1:8000**。在页面中选择 `demo` 知识库，或点击“载入原创示例”。

首次运行 OCR 可能需要下载模型文件；之后可以复用本地模型。无大模型 API Key 时仍可完成原文解析、OCR、真实索引、混合检索与抽取式问答。若暂不运行 OCR，设置 `RAG_OCR_ENABLED=false`，再通过页面上传 Markdown 和原生 PDF。

若 Windows 中文路径创建 venv 时 `ensurepip` 失败，可以让已有的 pip 为不含 pip 的环境安装依赖：

```powershell
python -m venv --without-pip .venv
python -m pip --python .venv\Scripts\python.exe install -e ".[dev]"
```

Docker：

```bash
docker compose up --build
```

## 五类原创演示资料

`rag-workbench demo` 自动生成并导入：

1. `atlas.md`：虚构 Atlas A1/A2 产品条款，检验型号精确匹配。
2. `retrieval.md`：检索技术笔记，检验术语查询和跨概念比较。
3. `atlas-spec.pdf`：原生 PDF，检验页码与区域坐标。
4. `maintenance.png`：固件升级流程图，检验 OCR 和图片来源。
5. `scanned-maintenance.pdf`：仅含图片的 PDF，检验扫描页 OCR 回退。

示例为本项目原创虚构资料，不包含企业数据，也不公开参考课程 PDF。

可以提问：

- **Atlas A1 的保修期是多久？**
- **Atlas A1 和 Atlas A2 有什么区别？**
- **BM25 和向量检索有什么区别？**
- **Atlas A1 firmware update**
- **它的额定功率呢？**（先问过 Atlas A1 后）
- **这个产品呢？**（新会话中触发澄清）

## 使用自己的资料

```powershell
rag-workbench ingest "D:\notes\资料.pdf" --kb personal
rag-workbench ask "文档中规定的保修范围是什么？" --kb personal --session alice
```

同一知识库内相同文件名默认视为同一份资料，修改后导入会更新版本；同名但不同资料建议重命名，Python 服务接口也支持显式 `document_id`。

CLI 导入 Markdown 时可解析位于文档目录内的相对路径图片。网页采用单文件上传，不自动建立多个上传文件之间的相对路径关系；网页使用 PDF 或单独图片更方便。

## 本地语义 Embedding 与神经重排

```powershell
python -m pip install -e ".[semantic]"
$env:RAG_EMBEDDING="local"
$env:RAG_LOCAL_EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
$env:RAG_RERANKER="neural"
$env:RAG_LOCAL_RERANK_MODEL="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
rag-workbench demo
```

模型路径也可以指向预下载的本地目录。此模式使用 CPU，首次下载和初始化时间与网络、硬件有关。更换模型后索引采用新配置，旧检索缓存不会复用；不支持多个进程以不同配置同时写同一运行目录。

## 远程模型与 VLM

编辑 `.env`，或设置同名环境变量：

```dotenv
RAG_EMBEDDING=remote
RAG_GENERATOR=remote
RAG_BASE_URL=https://your-provider.example/v1
RAG_API_KEY=your-api-key
RAG_CHAT_MODEL=your-chat-model
RAG_EMBEDDING_MODEL=your-embedding-model
RAG_VLM_MODEL=your-vision-model
```

模型服务需要支持相应的 `/embeddings`、`/chat/completions` 和图片输入协议。VLM 解析结果会保存为 `vision` 内容块，与 OCR 和关联正文一起进入索引，并保留原图。

未配置 VLM 时明确记录 `not_configured`；请求失败时记录 `failed`。它们不会被描述为“已完成图像理解”。

模型生成答案后先检查引用编号，再调用模型检查事实支持性。不通过或请求失败时回退到原文抽取，并在 Trace 中记录原因。模型复核仍可能出错，用户可以查看原始引用进行复核。

## 解析与来源定位

- **原生 PDF：** 使用文本块、页码和 bbox；没有可用原生文字时整页渲染后 OCR。
- **PDF 图片：** 抽取内嵌图片后执行 OCR，以及配置后的 VLM 理解。
- **Markdown：** 保留标题路径与行号；识别代码围栏，避免把代码注释误当标题。
- **图片：** 转为 PNG，保存内容 Hash，绑定 OCR 和视觉描述。
- **切分：** 使用 LangChain `Document` 与递归切分器，保留每个块的来源信息。

内容块包含 `document_id`、`version`、`chunk_id`、类型、页码、标题、行号、图片 ID 和区域坐标。复杂双栏阅读顺序、跨页表格和超长代码块不保证完美还原，原文件始终保留供复核。

## 混合检索与问答

链路依次完成 **问题路由、指代改写、BM25/Chroma 召回、RRF 融合、重排、上下文组装、回答与引用核查**。

- BM25 保留术语与型号匹配能力。
- 向量检索按实际配置使用统计或语义向量。
- RRF 融合排名，不直接相加尺度不同的原始分数。
- 比较问题最多拆出三个子查询，独立检索后去重。
- 图片问题优先展示已召回的 OCR/视觉内容块。
- 证据不足时最多尝试一次规则扩展检索；无法找到依据则拒答。
- 指代问题只读取当前知识库、当前会话近期历史；无可用历史时要求澄清。

默认上下文最多五个内容块，每块长度受切分配置约束。最终证据编号从 `E1` 开始，只能引用当前有效版本中的实际召回内容。

## 版本更新与一致性

资料身份与内容版本分开维护：文件名产生稳定文档 ID，文件内容、Markdown 关联图片、解析配置和切分参数共同确定版本。

更新过程：

1. 检查 Hash，完全不变则跳过解析与建库。
2. 仅解析受影响的文档，复用可用解析缓存。
3. 基于新内容和其他有效文档构建候选 Chroma 集合。
4. 候选索引成功后，SQLite 事务同时切换文档清单、集合名和知识库 epoch。
5. 读请求在同一进程锁内读取有效快照，旧版本不参与检索。
6. 失败时保留原有效版本，并记录失败信息。

**这是文档解析与 Embedding 缓存层面的增量更新。** 为保证小型知识库一致性，每次变更会重建 BM25 统计结构和候选向量集合；语义模式复用未变化片段的 Embedding。TF-IDF 的词汇表依赖语料，因此基线模式重新计算全部统计向量。

有效 Chroma 集合和 SQLite 清单持久化保存。重启后恢复有效版本，不把“每次全部重新解析”称为增量索引。

删除会立即移除有效文档与检索证据。原文件、解析资产和旧缓存仍保留在本地运行目录用于排查；需要物理清除历史数据时，可停服后清理对应运行目录并重新导入。

## 分层缓存

| 层级 | 缓存键的重要组成 |
|---|---|
| 解析 | 文件 Hash、关联图片、解析器版本、OCR/VLM 配置 |
| 语义 Embedding | 内容、模型和服务地址 |
| 检索与重排结果 | 知识库 epoch、查询、模式、Top K、模型和重排配置 |
| 回答 | epoch、问题、改写结果、生成与检索模型配置 |

更新或删除资料后 epoch 增加，旧检索和回答缓存不再命中。缓存保存在 SQLite，页面展示回答缓存命中；完整过程保存到 `runtime/reports/`。

## 评测

```powershell
rag-workbench evaluate examples/evaluation.jsonl --kb demo
```

四组实验固定问题集与 Top K，分别执行 **BM25、向量检索、混合检索、混合检索+重排**。报告包含：

- Hit Rate@K、MRR@K、Recall@K。
- 每个问题的实际召回来源、命中排名和耗时。
- 知识库版本、Embedding 后端和重排后端。

当前标签单位是来源文档，MRR 按内容块排序中首个相关文档位置计算；不是句子级或 chunk 级相关性标注。示例集规模较小，只用于验证流程，不代表生产准确率。对照实验不预设提升幅度。

对照实验先加载索引，检索计时绕过结果缓存；BM25 单路实验不调用向量查询。运行报告标记 `cache=bypassed`，避免把缓存耗时当作真实检索性能。

RAGAS：

```powershell
python -m pip install -e ".[eval]"
rag-workbench evaluate examples/evaluation.jsonl --kb demo --ragas
```

需要配置评测模型与 Embedding 接口。指标包括 Faithfulness、Answer Relevancy、Context Precision。返回状态严格区分 `not_run`、`executed` 与 `failed`；未配置模型不会输出模拟评分。

## HTTP API

| 接口 | 用途 |
|---|---|
| `GET /health` | 查看运行后端 |
| `GET /api/documents?kb=demo` | 当前文档与版本 |
| `POST /api/documents?kb=personal` | multipart 上传资料 |
| `DELETE /api/documents/{id}?kb=personal` | 删除资料并更新索引 |
| `GET /api/documents/{id}/original?kb=personal` | 打开原资料，PDF 支持页码链接 |
| `GET /api/assets/{image_id}` | 查看引用图片 |
| `POST /api/ask` | 返回回答、证据与 Trace |
| `POST /api/ask/stream` | SSE：运行状态、核查后的答案分段、最终结果 |
| `POST /api/demo` | 生成并载入原创示例 |
| `POST /api/evaluate?kb=demo` | 固定示例集四组检索对照，`ragas=true` 可执行模型评测 |

SSE 为引用核查完成后的应用层分段发送，不宣称未经实现的模型原生 Token 流式输出。默认页面使用普通回答接口以便一起展示完整证据。

```json
{"question":"Atlas A1 的保修期是多久？","kb":"demo","session":"alice"}
```

## 项目结构

```text
src/rag_workbench/
  config.py       配置与 .env
  models.py       解析块、内容块、证据与答案
  parsing.py      PDF / Markdown / OCR / 图片解析与 LangChain 切分
  providers.py    语义模型、VLM、神经重排与生成链
  retrieval.py    Chroma、BM25、RRF、MMR 与重排
  tfidf.py        可检查的 NumPy 字符 n-gram TF-IDF 基线
  storage.py      SQLite 文档清单、版本、缓存与会话
  service.py      入库、更新、问答与引用核查
  evaluation.py   检索指标与 RAGAS
  demo.py         原创 PDF、图片与扫描页生成
  api.py          FastAPI 与 SSE
  cli.py          命令行
  web/            中文工作台
examples/         原创文本资料与固定评测问题
tests/            解析、版本、缓存、隔离、引用、指标与 API 测试
```

## 自动化验证

```powershell
python -m ruff check src tests
python -m pytest -p no:cacheprovider
```

测试覆盖重复导入、修改替换、删除失效、解析失败保留旧版本、重启恢复、多知识库隔离、会话隔离、PDF 页码、代码围栏、关联图片变更、无效引用回退、RRF 与指标计算，以及 HTTP 上传、SSE、原文件访问和删除。

本地验证环境为 Windows / Python 3.12，**30 项测试通过**；五类原创资料成功入库，图片和扫描 PDF 的真实 OCR 状态为 `executed`，中文页面已验证回答、来源和页码链接。原始运行报告位于 `runtime/evaluation.json`，可通过演示命令重新生成。

固定示例集只有 8 个问题，标签以来源文档为单位，不能证明答案事实准确率或生产效果。此次四组实验 Hit Rate@5 与文档 Recall@5 均为 1.0，前三组 MRR 为 1.0，词汇重排组 MRR 为 0.875；**重排没有在这份小测试集上带来提升**。语义模型、VLM、远程生成及 RAGAS 的真实模型调用尚未在本地验证，HTTP 协议与回退路径由自动化测试覆盖。当前机器未安装 Docker，容器启动未实测。

## 项目经历对应

| 技术亮点 | 对应模块 |
|---|---|
| 多模态离线索引与来源元数据 | `parsing.py`、`providers.py`、`models.py` |
| 混合检索与重排 | `retrieval.py` |
| 路由、改写、补充检索与引用核查 | `service.py` |
| RAGAS、MRR 与 Hit Rate | `evaluation.py`、`examples/evaluation.jsonl` |
| 文档 Hash 与有效版本更新 | `service.py`、`storage.py` |
| 模块化和分层缓存 | 各 Provider、Parser、Snapshot、Store 与服务接口 |

OCR/VLM、语义模型和 RAGAS 的能力由对应真实后端提供；默认基线演示不代表这些增强后端全部已执行。

## 部署边界

面向个人本地单实例使用，默认绑定 127.0.0.1。知识库 ID 与图片 ID 有路径校验，上传限制文件类型、大小及 PDF 页数；不执行文档内指令。没有账户认证，勿直接作为公网多人服务部署。公开服务需要另外增加认证、上传扫描、隔离运行与权限校验。

参考资料仅用于理解工程思路。项目不收集企业资料、不包含密钥。
