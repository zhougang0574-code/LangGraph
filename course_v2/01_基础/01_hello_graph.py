"""
【01 基础 / 01】Hello Graph —— 最小可运行的 LangGraph
==============================================
只有 4 个概念，先跑起来再说：

  State  —— 共享数据包（TypedDict）
  Node   —— 普通函数，读 State、返回更新字段
  Edge   —— 节点之间的有向连线
  Graph  —— 把以上三者组装起来，compile 后 invoke 运行
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ── 1. State：整个 Graph 的唯一数据载体 ────────────────────────────────────
class State(TypedDict):
    question: str
    answer: str


# ── 2. Node：普通函数，接收 State，返回要更新的字段 ─────────────────────────
def llm_node(state: State) -> dict:
    response = llm.invoke(state["question"])
    return {"answer": response.content}   # 只返回要改的字段，其余保持不变


def answer_node(state: State) -> dict:
    print(f"回答字数：{len(state['answer'])}")
    return {}   # 不更新任何字段，返回空字典


# ── 3. 构建 Graph ──────────────────────────────────────────────────────────
builder = StateGraph(State)

builder.add_node("llm_node", llm_node)
builder.add_node("answer_node", answer_node)

# ── 4. Edge：决定执行顺序 ──────────────────────────────────────────────────
builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", "answer_node")
builder.add_edge("answer_node", END)

graph = builder.compile()


# ── 5. 运行 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = graph.invoke({"question": "用一句话介绍 LangGraph"})
    print("问题：", result["question"])
    print("回答：", result["answer"])


# ── 执行流程 ───────────────────────────────────────────────────────────────
"""
初始 State: {question: "用一句话介绍 LangGraph", answer: ""}
      │
      ▼ START → llm_node
 [llm_node]     调用 LLM，把回答写入 answer 字段
      │
      ▼ llm_node → answer_node
 [answer_node]  打印字数，返回 {}（不改任何字段）
      │
      ▼ answer_node → END
     END

invoke() 返回最终 State，包含所有字段的最新值。

★ 3 个最重要的细节：

1. 节点只需返回"改了什么"，不需要复制整个 State
   return {"answer": "xxx"}  ← 只更新 answer，question 保持不变

2. add_edge(START, "llm_node") 设置入口节点
   LangGraph 没有 set_entry_point 也可以，START 就是入口标记

3. compile() 必须调用，之后才能 invoke()
"""
