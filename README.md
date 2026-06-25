# LangGraph 学习项目

从零开始、循序渐进学 LangGraph 的代码 + 笔记。LLM 用阿里百炼 qwen-plus（兼容 OpenAI 接口）。

---

## 🚀 换电脑 / 新电脑快速启动

> `.venv/`（虚拟环境）和 `.env`（含 API Key）**不在仓库里**，新电脑必须重新创建。
> 其余代码、笔记 git 一拉就全有。

```bash
# 1. 拉代码
git clone git@github.com:zhougang0574-code/LangGraph.git
cd LangGraph

# 2. 建虚拟环境（Python 3.11）
python -m venv .venv

# 3. 激活虚拟环境
#   Windows PowerShell:
.venv\Scripts\Activate.ps1
#   Windows CMD:
#   .venv\Scripts\activate.bat
#   macOS / Linux:
#   source .venv/bin/activate

# 4. 装依赖
pip install -r requirements.txt

# 5. 创建 .env（见下方格式），填入自己的 API Key

# 6. 跑任意一课验证
python "course_v2/01_基础/01_hello_graph.py"
```

### `.env` 格式（放在项目根目录，不要提交）

```
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
API_KEY=sk-你的密钥
MODEL=qwen-plus
```

> `.env` 只需放在**根目录**一份，所有子文件夹里的课都能自动找到（`load_dotenv()` 会向上查找）。

---

## 📁 目录结构

| 目录 / 文件 | 说明 |
|------------|------|
| **`course_v2/`** | **← 当前学习版本，从这里开始** |
| `course_v2/README.md` | 课程指南：10 个概念域 + 学习顺序 + 优化说明 |
| `course_v2/langgraph_notes.html` | 网页版图文笔记（浏览器直接打开） |
| `course_v2/01_基础 … 10_子图与多Agent/` | 36 课，按概念域分组 |
| `archive_v1/` | 旧版线性 23 课的完整留底（参考用，一般不动） |
| `requirements.txt` | Python 依赖 |
| `.env` | API Key 配置（需自建，不入库） |
| `.venv/` | 虚拟环境（需自建，不入库） |

---

## 📖 怎么学

课程组织方式：**按概念域分组（分类轴）+ 每个域内一文件一个新概念、小步递进（域内梯度）**。
域与域之间已按前置依赖排好顺序，从上往下学即可：

```
01 基础 → 02 状态State → 03 节点Node → 04 控制流Edge → 05 工具与Agent
→ 06 持久化 → 07 状态读写 → 08 中断 → 09 流式 → 10 子图与多Agent
```

- **想看代码逐课跑**：进 `course_v2/`，按文件夹序号 + 文件内序号顺序看，每个 `.py` 顶部 docstring 写了「本课新概念」和「与上一课的区别」。
- **想看图文讲解**：浏览器打开 `course_v2/langgraph_notes.html`，侧边栏按域折叠导航。
- **完整说明**：见 `course_v2/README.md`。

---

## ⚠️ 已知事项

- 部分课需要额外的库（均已写进 `requirements.txt`，旧 `.venv` 可能未装，跑前 `pip install -r requirements.txt`）：
  - `06_持久化与记忆/02_sqlite_saver.py` → `langgraph-checkpoint-sqlite`
  - `10_子图与多Agent/05_supervisor_lib.py` → `langgraph-supervisor`
  - `01_基础/03_visualize.py` 的 `print_ascii()` → `grandalf`（`draw_mermaid()` 不需要）
- Windows 下若用纯命令行（非 PyCharm/UTF-8 终端）跑课，遇到 `▶ ✓` 等字符的 `GBK` 编码报错，
  可先设 `set PYTHONIOENCODING=utf-8` 或直接在 PyCharm 里运行。
- 课程基于 qwen-plus 调试；换模型或网关时，token 级流式、并行工具调用等表现可能略有差异。
