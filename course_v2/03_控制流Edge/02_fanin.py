"""
【03 控制流Edge / 02】多路汇聚（Fan-in）
==========================
与〔03_控制流Edge/01〕的区别：
  〔03_控制流Edge/01〕  2条分支 → 各自直接走 END
  〔03_控制流Edge/02〕  3条分支 → 汇入同一个 print_node → 再走 END

新概念（只有这一个）：
  多个节点可以都连向同一个节点（fan-in）
  add_edge("brief",    "print_node")
  add_edge("detailed", "print_node")
  add_edge("bullet",   "print_node")
  三条路最终汇到同一个出口

用途：
  不管走哪条分支，最后都需要做同一件事（打印、保存、记录…）
  把这件事提取成一个公共节点，避免在每个分支里重复写。
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


# ── 1. State：和〔03_控制流Edge/01〕一样，mode 多了一个选项 "bullet" ─────────────────────
class State(TypedDict):
    question: str
    mode: str    # "brief" | "detailed" | "bullet"
    answer: str


# ── 2. 路由函数：多了一个分支 ──────────────────────────────────────────────
def route_by_mode(state: State) -> str:
    mode = state.get("mode", "brief")
    if mode == "brief":
        return "brief"
    elif mode == "detailed":
        return "detailed"
    return "bullet"


# ── 3. 三条分支的节点 ──────────────────────────────────────────────────────
def brief_node(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content="你是简洁的助手，用1-2句话回答，不展开细节。"),
        HumanMessage(content=state["question"]),
    ])
    return {"answer": response.content}


def detailed_node(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content="你是耐心的助手，详细解释，可以分点说明，越完整越好。"),
        HumanMessage(content=state["question"]),
    ])
    return {"answer": response.content}


def bullet_node(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content="你是结构化的助手，用5条以内的要点（bullet point）回答，每条一行，以 - 开头。"),
        HumanMessage(content=state["question"]),
    ])
    return {"answer": response.content}


# ── 4. 汇聚节点：三条路都会经过这里 ──────────────────────────────────────
def print_node(state: State) -> dict:
    print(f"\n【模式：{state['mode']}】")
    print(state["answer"])
    return {}


# ── 5. 构建 Graph ──────────────────────────────────────────────────────────
builder = StateGraph(State)
builder.add_node("brief",      brief_node)
builder.add_node("detailed",   detailed_node)
builder.add_node("bullet",     bullet_node)
builder.add_node("print_node", print_node)

builder.add_conditional_edges(
    START,
    route_by_mode,
    {
        "brief":    "brief",
        "detailed": "detailed",
        "bullet":   "bullet",
    }
)

# 三条分支都汇入 print_node（fan-in）
builder.add_edge("brief",    "print_node")
builder.add_edge("detailed", "print_node")
builder.add_edge("bullet",   "print_node")

builder.add_edge("print_node", END)

graph = builder.compile()


# ── 6. 运行 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    question = "什么是面向对象编程？"

    for mode in ["brief", "detailed", "bullet"]:
        graph.invoke({"question": question, "mode": mode})


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
Graph 结构：

                 ┌─ brief    ─┐
  START → (路由) ┼─ detailed ─┼→ print_node → END
                 └─ bullet   ─┘

fan-in 的价值：
  print_node 只写一次，三条路都能用到。
  如果没有 fan-in，就要在 brief / detailed / bullet 里各自写一遍打印逻辑。
"""
