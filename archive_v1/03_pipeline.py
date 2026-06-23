"""
第三课：节点链 —— 数据在节点间流动
=====================================
与第二课的区别：
  第二课  2 个节点，answer_node 只打印、不写入 State
  第三课  3 个节点，每个节点都有自己负责的字段

新概念（只有这一个）：
  节点链 = 数据流水线
  - 前面的节点把结果写入 State
  - 后面的节点从 State 读取，继续加工
  - 节点不一定要调 LLM，纯 Python 逻辑也是节点

数据流向：
  question + role
      ↓ llm_node 负责
    answer
      ↓ count_node 负责（纯 Python，不调 LLM）
    word_count
      ↓ print_node 负责
    （打印结果，不写入字段）
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


# ── 1. State：比第二课多了一个 word_count 字段 ─────────────────────────────
class State(TypedDict):
    question: str
    role: str
    answer: str
    word_count: int   # 新增：由 count_node 负责写入


# ── 2. 节点 ────────────────────────────────────────────────────────────────

def llm_node(state: State) -> dict:
    """节点1：调用 LLM，写入 answer（和第二课完全一样）"""
    response = llm.invoke([
        SystemMessage(content=state["role"]),
        HumanMessage(content=state["question"]),
    ])
    return {"answer": response.content}


def count_node(state: State) -> dict:
    """节点2：统计字数，写入 word_count（纯 Python，不调 LLM）"""
    count = len(state["answer"])   # 读取 llm_node 写入的 answer
    return {"word_count": count}


def print_node(state: State) -> dict:
    """节点3：打印最终结果（读取所有字段，不写入任何字段）"""
    print(f"问题：{state['question']}")
    print(f"回答：{state['answer']}")
    print(f"字数：{state['word_count']}")
    return {}


# ── 3. Graph：3 个节点串联 ─────────────────────────────────────────────────
builder = StateGraph(State)
builder.add_node("llm_node", llm_node)
builder.add_node("count_node", count_node)
builder.add_node("print_node", print_node)

builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", "count_node")
builder.add_edge("count_node", "print_node")
builder.add_edge("print_node", END)

graph = builder.compile()


# ── 4. 运行 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = graph.invoke({
        "question": "用两句话介绍一下 Python 语言",
        "role": "你是一个简洁的技术助手，回答控制在两句话以内。",
    })
    # invoke 返回最终 State，包含所有字段
    print("\n最终 State：", result)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
关键规律：后面的节点能读到前面节点写入的字段

  llm_node   写入 answer
                 ↓
  count_node 读取 answer → 写入 word_count
                 ↓
  print_node 读取 answer + word_count（都能读到）

节点不一定要调 LLM：
  count_node 只是 len(state["answer"])，纯 Python。
  节点就是普通函数，想做什么都行。
"""
