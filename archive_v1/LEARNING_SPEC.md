# LangGraph 学习项目规范

> 本文档是"继续教学"的完整说明书。换了新 AI、换了新电脑，读完这份文档就能无缝接手。

---

## 一、学习者背景

| 项目 | 说明 |
|------|------|
| LangChain 基础 | 不熟练，不要预设已知 LangChain 概念 |
| 开发环境 | PyCharm，macOS |
| LLM 服务 | 阿里百炼 qwen-plus（兼容 OpenAI 接口） |
| 语言偏好 | 代码注释用中文，文档用中文 |
| GitHub | zhougang0574-code |

---

## 二、核心教学原则（最重要）

### 原则1：每课只引入一个新概念

这是整个课程设计最核心的规则。

- ✅ 本课的新东西只有一个，其余和上一课完全相同
- ❌ 不能同时引入两个新概念，哪怕它们"相关"
- ❌ 不能"顺手"重构之前的代码

**判断标准**：学生看完这节课，能用一句话说出"这次新学了 X"。如果说不出来，或者要说"X 和 Y"，说明内容太多了。

### 原则2：宁愿多生成几课，也不要一课讲太多

学生明确说过：**"我宁愿多生成几个课时，也不要一下子讲太多内容，理解起来有困难。"**

遇到"这两个概念很相关，放一课讲完不就行了"的冲动，要拆成两课。

### 原则3：概念累积，不要回退

每节课都在上一节课的基础上加一层，之前的代码结构保留：

```
第一课：State + Node + Edge（llm.invoke("字符串")）
第二课：在第一课基础上 → 换成 HumanMessage + SystemMessage
第三课：在第二课基础上 → 加一个节点，展示数据流
第四课：在第三课基础上 → 加条件路由
...
```

学生通过对比"这次和上一课哪里不一样"来理解新概念。

### 原则4：Graph 结构改变是大步

以下改变属于"大步"，需要单独成课，不能和其他新概念合并：

- 从线性变分叉（引入 `add_conditional_edges`）
- 从无循环变有循环（引入 loop / ReAct）
- 从单 agent 变多 agent
- 引入 Checkpointer（引入持久化）

---

## 三、当前课程进度

### 已完成课程

| 文件 | 课题 | 新增概念 | Graph 结构 |
|------|------|---------|-----------|
| `01_hello_graph.py` | Hello Graph | State / Node / Edge / compile / invoke | 线性，2节点 |
| `02_messages.py` | 消息格式 | HumanMessage + SystemMessage，State 新增字段 | 线性，2节点 |
| `03_pipeline.py` | 节点链 | 节点间数据流动，节点可以是纯 Python | 线性，3节点 |
| `04_conditional.py` | 条件路由 | add_conditional_edges，路由函数，映射字典 | 分叉，2路径 |

### 当前 State 累积情况

每课 State 字段是在上一课基础上累积的：

```python
# 第一课
class State(TypedDict):
    question: str
    answer: str

# 第二课（新增 role）
class State(TypedDict):
    question: str
    role: str       # ← 新增
    answer: str

# 第三课（新增 word_count）
class State(TypedDict):
    question: str
    role: str
    answer: str
    word_count: int  # ← 新增

# 第四课（回到简洁，mode 替代 role）
class State(TypedDict):
    question: str
    mode: str        # ← 替代 role，用于路由
    answer: str
```

---

## 四、待完成课程规划

以下是建议的后续课程，每课只引入一个新概念，按顺序进行：

| 课次 | 建议课题 | 新增概念 | 前置依赖 |
|------|---------|---------|---------|
| 第五课 | 多分支汇聚 | 3条分支 → 汇入同一个节点（fan-in） | 第四课 |
| 第六课 | 消息列表（messages list） | `Annotated[list, add_messages]` Reducer，理解"追加"vs"覆盖" | 第三课 |
| 第七课 | 工具定义 | `@tool` 装饰器，`bind_tools()`，了解 tool_calls | 第六课 |
| 第八课 | ReAct Agent | `ToolNode`，`tools_condition`，循环结构 | 第七课 |
| 第九课 | 持久化记忆 | `MemorySaver`，`compile(checkpointer=...)`，`thread_id` | 第八课 |
| 第十课 | Human-in-the-Loop | `interrupt_before`，`invoke(None)`，`update_state` | 第九课 |
| 第十一课 | 多 Agent | Supervisor 模式，`next` 字段路由 | 第十课 |

> **注意**：上表是建议，不是约束。每次开始新课前，先问学生当前对哪个概念不清楚，根据反馈调整节奏。

---

## 五、每课交付流程（必须按序）

### 第一步：写 Python 代码文件

**命名规则**：`01_hello_graph.py`、`02_messages.py`……，序号严格递增。

**代码文件必须包含**：

1. 文件顶部 `"""docstring"""` 说明：
   - 本课课题
   - 与上一课的区别（只有这一点）
   - 新概念的说明

2. 代码内注释：说明"为什么"，而不只是"做了什么"

3. 文件末尾说明块（多行字符串）：
   - 执行流程图（ASCII 格式）
   - 本课核心规律（★ 标注）

**代码风格**：

```python
# ── 节标题分隔符风格（保持一致）──────────────────────
# 节点函数：接收 State，返回 dict（哪怕不改任何字段也要返回 {}）
def my_node(state: State) -> dict:
    ...
    return {}
```

### 第二步：Append HTML 笔记

**不要重新生成整个 HTML**，只在 `</main>` 前追加本课 section。

每课 HTML 必须包含（按顺序）：

1. `<div class="lesson-divider"></div>` 分隔线
2. `<section id="lessonN">` 课程主体
   - `lesson-banner`（带课号标签、标题、描述、objectives 标签）
   - 各小节（`sec-title` + 内容）
   - 完整代码块
   - 执行流程图（`<div class="flow">`）
   - 知识点总结（`<div class="kp-grid">`）
   - **踩坑记录**（`<div class="pitfall">`，必须有，至少 2 条）

**同时更新侧边栏 nav**：在 `</nav>` 前（其他 nav-lesson 之前）插入本课导航，顺序必须与正文一致。

---

## 六、HTML 排版硬性要求

| 项目 | 要求 |
|------|------|
| 字体大小 | `font-size: 17px` |
| 侧边栏高度 | `height: 100vh`（**不能用 `min-height`**） |
| 侧边栏滚动 | `overflow-y: auto`，必须可上下滑动 |
| 导航结构 | 手风琴折叠，每课一个父条目 + 子标题列表 |
| 子标题默认 | `max-height: 0`（收起） |
| 展开时 | 添加 `.open` class → `max-height: 400px`（或足够大的值） |
| 箭头图标 | `▶`，展开时旋转 90° |
| 课程顺序 | 侧边栏 nav 顺序必须与正文顺序严格一致（第1课在最上方） |
| Banner 颜色 | 每课不同：blue / teal / green / purple / orange 轮换 |
| 踩坑样式 | `.pitfall` 红色左边框，有 `.pitfall-title` 标题 |

---

## 七、已踩过的坑（避免重复）

| 问题 | 原因 | 正确做法 |
|------|------|----------|
| 侧边栏无法滚动，课程多了看不到 | 用了 `min-height: 100vh` | 改为 `height: 100vh; overflow-y: auto` |
| 课程导航顺序乱 | append nav 时插入位置错误 | 每次 append 检查 nav 与正文的课程顺序，**新课导航加在已有导航之后** |
| 路由函数返回值和映射字典 key 不一致 | 拼写不统一 | 路由函数返回什么，映射字典 key 就写什么 |
| 节点返回 None 导致 State 异常 | 忘记 return 语句 | 哪怕什么都不改，也要 `return {}` |
| 用了 `Graph` 而不是 `StateGraph` | 与 LangChain 内部类混淆 | 永远从 `langgraph.graph` 导入 `StateGraph` |
| 代码文件 SyntaxError | f-string 里用了中文引号 `"` `"` | 统一用 ASCII 单引号 `'` |

---

## 八、环境配置

### .env 文件（禁止提交 git）

```
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
MODEL=qwen-plus
```

### LLM 初始化（所有课通用写法）

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)
```

### .gitignore 必须包含

```
.env
.idea/
.venv/
venv/
__pycache__/
*.pyc
```

---

## 九、新 AI 接手时的操作步骤

1. 读本文档，理解教学原则和当前进度
2. 读最新一课的 `.py` 文件，了解当前 State 结构和已有概念
3. 浏览 `langgraph_notes.html` 的侧边栏，确认已完成到哪一课
4. 问学生："上节课有没有哪里不清楚？准备好了就继续第 X 课。"
5. 按"每课只引入一个新概念"的原则，参照《待完成课程规划》继续

---

## 十、新项目使用方式

1. 复制本文件到新项目根目录
2. 创建 `.env` 填入 API Key
3. 创建 `.gitignore`（参考第八节）
4. 告知 AI：**"请读 LEARNING_SPEC.md，按照规范帮我学习 xxx"**
