# CLAUDE.md

给后续 Claude Code 会话的项目说明，打开即读。**先读完这份再动手。**

## 这是什么

LangGraph 的**中文教学项目**：一套循序渐进的代码课 + 一份网页图文笔记。
学习者 LangChain 基础不熟练 —— 讲解时不要预设已知 LangChain 概念，代码注释 / 文档 / 讲解一律用中文。
LLM 用阿里百炼 **qwen-plus**（兼容 OpenAI 接口）。

## 权威版本（最重要）

- **`course_v2/` 是唯一在维护的学习版本，一切以它为准。**
- `archive_v1/`（旧版线性 23 课）**已废弃**，只作回溯参考，**不要再改动它**。
- 课程的权威目录树和学习顺序见 **`course_v2/README.md`**（改了结构要同步更新它）。

## 课程结构

10 个「概念域」、36 个 `.py`，**目录结构即学习路径**，从上往下学：

```
01 基础 → 02 状态State → 03 节点Node → 04 控制流Edge → 05 工具与Agent
→ 06 持久化与记忆 → 07 状态读写与时间旅行 → 08 中断与人工干预 → 09 流式输出 → 10 子图与多Agent
```

设计轴：**分类轴（按概念域分组）+ 域内梯度（每域内 01→N 小步递进）**。
（02 状态 / 03 节点 / 04 控制流 对应 LangGraph 的三大支柱 State / Node / Edge。）

## 教学法（硬性原则）

1. **一个文件只引入一个新概念**；其余代码尽量和上一课保持一致，靠"对比上一课哪里不一样"来教。
2. **宁可多拆几课，也不要一课塞太多**（学习者明确这样要求过）。
3. **概念累积、不回退**：后面的课在前面的基础上加一层。
4. 标〔进阶〕的课可第一遍跳过，放在各域末尾。

## 文件内约定

每个 `.py` 顶部 docstring 包含：标题行 `【域名 / 序号】`、与上一课的区别、新概念说明；
文末用多行字符串写"执行流程 / 核心规律"。跨课引用统一写 **`〔域文件夹名/序号〕`**，例如 `〔05_工具与Agent/02〕`。

> ⚠️ 改了课程编号 / 顺序，必须同步更新所有 `【…】` 标题和 `〔…〕` 引用（`.py` 与 HTML 都有）。
> 批量改时用 Python 脚本做"全 token 替换"最稳（键含域名，避免串号）。

## 三处必须同步

新增 / 调整一课时，以下三处都要改，保持一致：

1. **`.py`** —— docstring 标题、区别、新概念、末尾流程图。
2. **`course_v2/langgraph_notes.html`** —— 侧边栏 nav 条目 + 正文 `<section>`（详见下）。
3. **`course_v2/README.md`** 与根 **`README.md`** —— 目录树、课程数、依赖说明。

## HTML 笔记说明（`course_v2/langgraph_notes.html`）

- 单文件、零依赖，浏览器直接打开；**暗色护眼主题**（CSS 变量在 `<style>` 顶部 `:root`）。
- 左侧 `<nav>` 手风琴折叠（domain 标题 + 每课 `nav-lesson` + 子锚点）；右侧 `<main>` 每课一个 `<section>`。
- **nav 顺序与正文 section 顺序必须严格一致**；每课 section 前有一个 `<div class="lesson-divider">`。
- 历史原因存在**两种 banner 风格**：富 `lesson-banner`+`lesson-tag`（前 7 课）、简 `lesson-header`+`lesson-num`（其余）。新增课用简风格即可（与多数一致）。
- 课号用**域内序号**：富风格在 `lesson-tag`，简风格在 `lesson-num` 和 `nav-lesson-num`。
- 改动量大时优先用 Python 脚本对 HTML 做切片 / 替换，改完务必校验：`<section>` 开闭数相等、`<div>` 开闭数相等、nav 顺序 == section 顺序。

## 运行与环境

- `.env` 只放在**仓库根目录**（`load_dotenv()` 向上查找），格式：
  ```
  BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
  API_KEY=sk-xxxx
  MODEL=qwen-plus
  ```
- 用项目根的 `.venv` 运行，例：`python "course_v2/05_工具与Agent/02_react_agent.py"`
- LangGraph 1.2.5。额外依赖（`.venv` 常未装，见 `requirements.txt`，不装跑对应课会 ImportError）：
  `langgraph-checkpoint-sqlite`（06/02）、`langgraph-supervisor`（10/05）、`grandalf`（01/03 的 `print_ascii`）。
- **Windows 编码坑**：代码里用了 `▶ ✓ ✗` 等字符，纯命令行跑会触发 GBK 报错。
  自测时用 `PYTHONIOENCODING=utf-8`；用户实际在 PyCharm（UTF-8）里跑没问题。

## 新增一课的步骤

1. 在对应域文件夹放 `NN_xxx.py`（序号紧接该域已有文件；若插在中间，后续文件和所有引用都要重编号）。
2. 写 docstring（标题 / 区别 / 新概念）+ 代码 + 末尾流程图，遵循"单概念"原则。
3. 在 HTML 的 nav 和 `<main>` 对应位置插入 section（简风格），更新域内序号。
4. 更新 `course_v2/README.md` 目录树 + 根 `README.md` 课程数。
5. 校验：`python -m py_compile` 全部通过；能离线跑的就实跑一遍；HTML 标签开闭与顺序校验。
