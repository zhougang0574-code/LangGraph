"""
第二课：消息格式 —— HumanMessage 与 SystemMessage
=================================================
与第一课的区别：只改了 LLM 的调用方式，Graph 结构完全不变。

第一课：llm.invoke("字符串")          ← 最简单，但不够灵活
第二课：llm.invoke([HumanMessage(...)]) ← 标准消息格式
        llm.invoke([SystemMessage(...), HumanMessage(...)]) ← 给 LLM 设定角色

新概念（只有这一个）：
  消息列表 = [SystemMessage, HumanMessage, AIMessage, ...]
  - HumanMessage：用户说的话
  - SystemMessage：给 LLM 的"角色设定"，决定它的回答风格
  - AIMessage：LLM 的回复（一般不需要手动创建）
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


# ── 1. State：比第一课多了一个 role 字段 ───────────────────────────────────
# role 字段用来控制 LLM 的回答风格
# 这样同一个 Graph，传不同的 role，得到不同风格的回答
class State(TypedDict):
    question: str
    role: str     # 新增：LLM 的角色设定（SystemMessage 的内容）
    answer: str


# ── 2. 节点：只改了 LLM 的调用方式，其余和第一课一样 ─────────────────────
def llm_node(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content=state["role"]),      # 角色设定
        HumanMessage(content=state["question"]),   # 用户问题
    ])
    return {"answer": response.content}


def answer_node(state: State) -> dict:
    print(f"回答字数：{len(state['answer'])}")
    return {}


# ── 3. Graph：和第一课完全一样的结构 ──────────────────────────────────────
builder = StateGraph(State)
builder.add_node("llm_node", llm_node)
builder.add_node("answer_node", answer_node)
builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", "answer_node")
builder.add_edge("answer_node", END)

graph = builder.compile()


# ── 4. 运行：同一个问题，不同 role，观察回答风格的变化 ────────────────────
if __name__ == "__main__":
    question = "什么是人工智能？"

    print("=" * 50)
    print("角色1：正经的学术助手")
    print("=" * 50)
    result1 = graph.invoke({
        "question": question,
        "role": "你是一个严谨的学术助手，用正式、专业的语言回答问题，回答控制在3句话以内。",
    })
    print(result1["answer"])

    print("\n" + "=" * 50)
    print("角色2：幽默的助手")
    print("=" * 50)
    result2 = graph.invoke({
        "question": question,
        "role": "你是一个幽默风趣的助手，用轻松有趣的语气回答问题，可以加一些比喻，回答控制在3句话以内。",
    })
    print(result2["answer"])


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
消息列表的格式：

  llm.invoke([消息1, 消息2, ...])

  ┌─────────────────────────────────────────────────────┐
  │  消息类型         作用                               │
  │─────────────────────────────────────────────────────│
  │  SystemMessage   给 LLM 的角色/规则设定              │
  │                  LLM 会按这个设定来回答              │
  │                  通常放在列表第一位                  │
  │─────────────────────────────────────────────────────│
  │  HumanMessage    用户说的话（这一轮的输入）           │
  │─────────────────────────────────────────────────────│
  │  AIMessage       LLM 上一轮的回复（多轮对话用）       │
  │                  本课暂时用不到                      │
  └─────────────────────────────────────────────────────┘

★ 对比第一课：

  第一课：llm.invoke("什么是人工智能？")
          → LLM 按默认风格回答

  第二课：llm.invoke([
              SystemMessage(content="你是幽默的助手"),
              HumanMessage(content="什么是人工智能？"),
          ])
          → LLM 按幽默风格回答

  字符串写法是消息列表的简写形式，内部会自动转成 HumanMessage。
  实际开发中绝大多数情况都用消息列表，因为你几乎总是需要设定角色。
"""
