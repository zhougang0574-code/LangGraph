"""
第九课：对话记忆 —— 两次 invoke() 之间保留历史
================================================
第八课的 ReAct Agent 每次 invoke() 都是全新的开始，
前一次的对话内容不会保留。

本课新概念（只有一个）：
  MemorySaver —— 把 State 存到内存，下次 invoke() 时自动恢复

两个关键变化：
  1. compile() 时传入 checkpointer=MemorySaver()
  2. invoke() 时传入 config={"configurable": {"thread_id": "某个ID"}}

thread_id 的作用：
  相同的 thread_id → 同一个对话，State 会叠加
  不同的 thread_id → 独立对话，互不干扰
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ── 1. State：和第六课一样 ────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── 2. 节点：和第六课一样 ─────────────────────────────────────────────────
def llm_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ── 3. Graph：结构和第六课一样，只有 compile() 多了一个参数 ──────────────
builder = StateGraph(State)
builder.add_node("llm_node", llm_node)
builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", END)

# ★ 唯一的变化：传入 checkpointer
#   MemorySaver 把每次执行后的 State 存在内存里
#   下次同一个 thread_id 进来时，自动从上次结束的地方恢复
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# ── 4. 运行：用 thread_id 区分不同的对话 ─────────────────────────────────
if __name__ == "__main__":
    # 每次 invoke() 都需要带上 config，指定是哪一个对话线程
    config_a = {"configurable": {"thread_id": "对话A"}}

    print("=" * 50)
    print("第1轮：告诉 LLM 我的名字")
    print("=" * 50)
    result = graph.invoke(
        {"messages": [HumanMessage(content="你好，我叫小明")]},
        config=config_a,
    )
    print("LLM：", result["messages"][-1].content)

    print("\n" + "=" * 50)
    print("第2轮：问 LLM 我叫什么（同一个 thread_id）")
    print("=" * 50)
    result = graph.invoke(
        {"messages": [HumanMessage(content="你还记得我叫什么名字吗？")]},
        config=config_a,
    )
    print("LLM：", result["messages"][-1].content)

    print("\n" + "=" * 50)
    print("第3轮：换一个 thread_id（新对话，不记得之前的内容）")
    print("=" * 50)
    config_b = {"configurable": {"thread_id": "对话B"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content="你还记得我叫什么名字吗？")]},
        config=config_b,
    )
    print("LLM：", result["messages"][-1].content)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
MemorySaver 做了什么：

  第1轮 invoke() 结束后，MemorySaver 把 State 存起来：
    thread "对话A" → messages: [HumanMessage("你好我叫小明"), AIMessage("你好小明！")]

  第2轮 invoke() 开始时，MemorySaver 把上次的 State 取出来：
    先恢复 messages: [HumanMessage("你好我叫小明"), AIMessage("你好小明！")]
    再追加新的 HumanMessage("你还记得我叫什么名字吗？")
    LLM 看到完整历史，自然知道你叫小明

  thread "对话B" 是全新的，没有历史记录。

没有 MemorySaver 的情况（之前的课）：
  每次 invoke() 都从空的 State 开始，不记得任何内容。

MemorySaver 是存在内存里的，程序重启后会丢失。
后续可以换成 SqliteSaver / PostgresSaver 实现持久化，但那是更高级的话题。
"""
