"""
第六课：消息列表 —— 追加而不是覆盖
=====================================
前五课的 State 字段都是"覆盖"行为：
  节点返回 {"answer": "新值"} → answer 被替换成新值

本课引入"追加"行为：
  节点返回 {"messages": [新消息]} → 新消息追加到列表末尾，旧消息保留

新概念（只有这一个）：
  Annotated[list[BaseMessage], add_messages]
  - Annotated 的第二个参数 add_messages 是一个"reducer 函数"
  - 它告诉 LangGraph：这个字段不要覆盖，要用 add_messages 函数来合并
  - 效果：每次返回的消息追加到列表末尾

为什么需要追加？
  对话历史不能覆盖——LLM 需要看到完整的对话记录才能理解上下文。
  HumanMessage("你好") + AIMessage("你好！") + HumanMessage("我叫小明")
  这三条都要保留，不能只留最新的一条。
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ── 1. State：只有一个字段，类型是消息列表 ────────────────────────────────
class State(TypedDict):
    # Annotated 第二个参数 add_messages 是 reducer：
    #   节点返回 {"messages": [新消息]} → 追加，不覆盖
    messages: Annotated[list[BaseMessage], add_messages]


# ── 2. 节点：调用 LLM，把回复追加到 messages ──────────────────────────────
def llm_node(state: State) -> dict:
    response = llm.invoke(state["messages"])   # 把完整历史传给 LLM
    return {"messages": [response]}            # 追加 AIMessage，不覆盖


def print_node(state: State) -> dict:
    print(f"\n对话共 {len(state['messages'])} 条消息：")
    for msg in state["messages"]:
        role = msg.__class__.__name__.replace("Message", "")
        print(f"  [{role}] {msg.content[:50]}")
    return {}


# ── 3. Graph：和前几课一样的线性结构 ─────────────────────────────────────
builder = StateGraph(State)
builder.add_node("llm_node",   llm_node)
builder.add_node("print_node", print_node)
builder.add_edge(START,        "llm_node")
builder.add_edge("llm_node",   "print_node")
builder.add_edge("print_node", END)

graph = builder.compile()


# ── 4. 运行 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = graph.invoke({
        "messages": [HumanMessage(content="用一句话介绍 Python")]
    })

    # 最终 State 里 messages 有两条：HumanMessage + AIMessage
    print("\n最终 messages 列表：")
    for msg in result["messages"]:
        print(f"  {msg.__class__.__name__}: {msg.content[:60]}")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
覆盖 vs 追加：

  普通字段（覆盖）：
    初始 State:   answer = ""
    llm_node 返回 {"answer": "Python 是..."}
    结果 State:   answer = "Python 是..."   ← 旧值被替换

  消息列表（追加）：
    初始 State:   messages = [HumanMessage("用一句话介绍 Python")]
    llm_node 返回 {"messages": [AIMessage("Python 是...")]}
    结果 State:   messages = [HumanMessage(...), AIMessage(...)]  ← 追加，两条都在

★ 核心规律：
  字段类型是 Annotated[list[BaseMessage], add_messages]
  → LangGraph 调用 add_messages(旧列表, 新列表) 合并
  → 结果是两个列表拼在一起，不是替换

  字段类型是普通 str / int
  → LangGraph 直接赋值：新值替换旧值
"""
