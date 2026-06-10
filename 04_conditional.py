"""
第四课：条件路由 —— 根据 State 走不同的路
==========================================
前三课的 Graph 都是线性的：A → B → C，固定不变。
本课第一次让图"分叉"——根据 State 的值决定走哪条路。

新概念（只有这一个）：
  add_conditional_edges(源节点, 路由函数, 映射字典)
  - 路由函数：接收 State，返回一个字符串（下一个节点的名字）
  - 映射字典：把路由函数的返回值映射到实际节点名

与 add_edge 的区别：
  add_edge("A", "B")           → 固定：A 完成后永远去 B
  add_conditional_edges(...)   → 动态：A 完成后根据 State 决定去哪
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


# ── 1. State：新增 mode 字段，用来控制走哪条路 ─────────────────────────────
class State(TypedDict):
    question: str
    mode: str    # "brief"（简洁）或 "detailed"（详细），invoke 时传入
    answer: str


# ── 2. 路由函数：读取 State，返回下一个节点的名字 ──────────────────────────
# 规则：
#   - 接收 State 作为参数
#   - 返回值是字符串，代表下一个节点的名字
#   - 不修改 State，只读取
def route_by_mode(state: State) -> str:
    if state["mode"] == "brief":
        return "brief_node"
    return "detailed_node"


# ── 3. 两条路上的节点 ──────────────────────────────────────────────────────
def brief_node(state: State) -> dict:
    """简洁路径：用简短的语气回答"""
    response = llm.invoke([
        SystemMessage(content="你是简洁的助手，用1-2句话回答，不展开细节。"),
        HumanMessage(content=state["question"]),
    ])
    return {"answer": response.content}


def detailed_node(state: State) -> dict:
    """详细路径：用详尽的语气回答"""
    response = llm.invoke([
        SystemMessage(content="你是耐心的助手，详细解释，可以分点说明，越完整越好。"),
        HumanMessage(content=state["question"]),
    ])
    return {"answer": response.content}


# ── 4. 构建 Graph ──────────────────────────────────────────────────────────
builder = StateGraph(State)
builder.add_node("brief",    brief_node)
builder.add_node("detailed", detailed_node)

# add_conditional_edges：从 START 出发，由 route_by_mode 决定去哪
# 第三个参数（映射字典）：路由函数返回什么字符串 → 对应哪个节点
builder.add_conditional_edges(
    START,
    route_by_mode,
    {
        "brief_node":    "brief",
        "detailed_node": "detailed",
    }
)

# 两条路都通向 END
builder.add_edge("brief",    END)
builder.add_edge("detailed", END)

graph = builder.compile()


# ── 5. 运行：同一个问题，不同 mode，走不同的路 ────────────────────────────
if __name__ == "__main__":
    question = "什么是递归？"

    print("=" * 50)
    print("mode=brief → 走 brief_node")
    print("=" * 50)
    result1 = graph.invoke({"question": question, "mode": "brief"})
    print(result1["answer"])

    print("\n" + "=" * 50)
    print("mode=detailed → 走 detailed_node")
    print("=" * 50)
    result2 = graph.invoke({"question": question, "mode": "detailed"})
    print(result2["answer"])


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
执行路径对比：

  mode="brief"
    START → (route_by_mode 返回 "brief_node") → brief_node → END

  mode="detailed"
    START → (route_by_mode 返回 "detailed_node") → detailed_node → END

两次 invoke，同一个 Graph，走了完全不同的节点。

映射字典 {"brief_node": "brief_node", ...} 看起来多余，
但它的作用是：允许路由函数的返回值与节点名不同。
比如路由函数返回 "short"，映射到实际节点名 "brief_node"：
  add_conditional_edges(START, router, {"short": "brief_node", ...})
"""
