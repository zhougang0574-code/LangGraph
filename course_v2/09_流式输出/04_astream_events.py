"""
【09 流式输出 / 04】astream_events —— token 级别流式输出
===============================================
〔09_流式输出/01〕的 graph.stream() 是节点跑完才吐出结果（粗粒度）。
本课引入 astream_events：LLM 每生成一个 token 就立刻输出（细粒度）。

新概念（只有这一个）：
  graph.astream_events(input, config, version="v2")
    → 异步迭代器，持续产出事件
    → 每个事件有 event 字段，表示事件类型
    → 过滤 event == "on_chat_model_stream" 即可拿到每个 token

对比〔09_流式输出/01〕：
  graph.stream()          → 节点执行完才输出，粗粒度
  graph.astream_events()  → LLM 每个 token 立刻输出，细粒度（打字机效果）

注意：astream_events 是异步方法，需要 async/await + asyncio.run()
"""

import asyncio
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
    streaming=True,   # ★ 开启 LLM 流式输出
)


# ── State 和节点 ──────────────────────────────────────────
class State(TypedDict):
    messages: list


def chat_node(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content="你是一个助手，请详细回答问题。"),
        *state["messages"],
    ])
    return {"messages": state["messages"] + [response]}


builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
graph = builder.compile()


# ══════════════════════════════════════════════════════
# ★ astream_events：逐 token 流式输出
# ══════════════════════════════════════════════════════
async def stream_tokens():
    print("▶ 开始流式输出（打字机效果）：\n")

    async for event in graph.astream_events(
        {"messages": [HumanMessage(content="用三句话介绍一下 Python 语言")]},
        version="v2",   # 固定写 v2
    ):
        # 只关心 LLM 输出 token 的事件
        if event["event"] == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                print(token, end="", flush=True)  # 逐 token 打印，不换行

    print("\n\n▶ 输出完毕")


# ══════════════════════════════════════════════════════
# 扩展：查看所有事件类型（调试用）
# ══════════════════════════════════════════════════════
async def show_all_events():
    print("\n── 所有事件类型 ──────────────────────────────────")
    seen = set()
    async for event in graph.astream_events(
        {"messages": [HumanMessage(content="1+1等于几")]},
        version="v2",
    ):
        kind = event["event"]
        if kind not in seen:
            seen.add(kind)
            print(f"  {kind}")


if __name__ == "__main__":
    asyncio.run(stream_tokens())
    asyncio.run(show_all_events())


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
astream_events 产出的事件类型（常用）：

  on_chain_start        → Graph / 节点开始执行
  on_chain_end          → Graph / 节点执行完毕
  on_chat_model_start   → LLM 开始生成
  on_chat_model_stream  → LLM 输出一个 token（★ 打字机效果就靠这个）
  on_chat_model_end     → LLM 生成完毕

典型过滤写法：
  if event["event"] == "on_chat_model_stream":
      token = event["data"]["chunk"].content

event 结构：
  {
    "event":    "on_chat_model_stream",
    "name":     "ChatOpenAI",          # 哪个 LLM
    "data":     {"chunk": AIMessageChunk(content="你")},
    "metadata": {"langgraph_node": "chat", ...}  # 哪个节点产生的
  }

stream() vs astream_events()：
  graph.stream()          同步，节点粒度，适合看每步 State 变化
  graph.astream_events()  异步，token 粒度，适合做打字机 UI 效果
"""
