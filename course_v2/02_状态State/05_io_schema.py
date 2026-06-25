"""
【02 状态State / 05（进阶）】输入/输出 Schema —— 对外只暴露该暴露的字段
=========================================================
前面所有课，invoke() 的输入和返回的输出，用的都是同一个 State——
State 里有什么字段，调用方就要传什么、能看到什么。

但有时 State 里有很多"内部中转字段"，不该让调用方关心：
  - 输入：调用方只该传 question，不该被迫传一堆内部字段
  - 输出：调用方只想拿 answer，不想看到一堆中间变量

新概念（只有这一个）：
  StateGraph(整体State, input_schema=输入State, output_schema=输出State)
    - input_schema ：invoke() 只接收这些字段（多传的会被忽略/报错）
    - output_schema：invoke() 返回只包含这些字段
    - 整体State    ：图内部节点之间流转用的完整字段集（包含内部中转字段）

三个 Schema 的关系：
  输入字段 ─┐
            ├─► 整体State（内部流转，可有私有字段）─► 输出字段
  其它字段 ─┘
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ── 1. 三个 Schema ────────────────────────────────────────────────────────
class InputState(TypedDict):
    question: str            # 对外：调用方只需要传这个


class OutputState(TypedDict):
    answer: str              # 对外：调用方只会拿到这个


# 整体 State = 输入 + 输出 + 内部中转字段
class OverallState(TypedDict):
    question: str            # 来自输入
    draft: str               # ★ 内部中转字段，调用方看不到也不用传
    answer: str              # 给到输出


# ── 2. 节点：内部用到 draft 这个私有字段 ──────────────────────────────────
def draft_node(state: OverallState) -> dict:
    """先让 LLM 写个初稿，存到内部字段 draft"""
    response = llm.invoke([
        SystemMessage(content="用一句话简洁回答。"),
        HumanMessage(content=state["question"]),
    ])
    return {"draft": response.content}


def polish_node(state: OverallState) -> dict:
    """把初稿加工成最终 answer（这里只是加个前缀示意）"""
    return {"answer": f"【已校对】{state['draft']}"}


# ── 3. 构建：声明 input_schema / output_schema ────────────────────────────
builder = StateGraph(
    OverallState,
    input_schema=InputState,    # invoke 只收 question
    output_schema=OutputState,  # invoke 只返回 answer
)
builder.add_node("draft",  draft_node)
builder.add_node("polish", polish_node)
builder.add_edge(START,   "draft")
builder.add_edge("draft", "polish")
builder.add_edge("polish", END)

graph = builder.compile()


# ── 4. 运行 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 输入只给 question（符合 InputState）
    result = graph.invoke({"question": "什么是闭包？"})

    # 输出只有 answer（符合 OutputState），看不到 question / draft
    print("返回的字段：", list(result.keys()))
    print("answer：", result["answer"])


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
不声明 input/output schema（前面所有课的默认行为）：
  invoke 的输入、输出都用整体 State——
  调用方要传 question、draft、answer，返回也带着 draft 这种内部字段。

声明之后：
  输入按 InputState 过滤  → 只认 question
  内部按 OverallState 跑   → draft 正常流转
  输出按 OutputState 裁剪  → 只给 answer

★ 核心规律：
  input_schema / output_schema 是"对外接口"，OverallState 是"内部实现"。
  三者都是 TypedDict；OverallState 的字段必须是输入∪输出∪内部字段的全集。

典型用途：
  把一个图当成可复用组件 / 子图（〔10_子图与多Agent/01〕）对外发布时，
  用 input/output schema 约定清晰的"传什么、回什么"，隐藏内部细节。
"""
