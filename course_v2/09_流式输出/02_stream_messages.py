"""
【09 流式输出 / 02】stream messages —— 最简单的逐 token 打字机
=========================================================
〔09_流式输出/01〕的 updates / values 是"节点级"：节点跑完才吐一次，
看不到 LLM 一个字一个字往外蹦的过程。

要"打字机效果"，最简单的办法就是 stream_mode="messages"。
（更细、能拿到所有事件类型的 astream_events 见〔09_流式输出/04〕。）

新概念（只有这一个）：
  graph.stream(input, stream_mode="messages")
    → 每次 yield 一个二元组 (message_chunk, metadata)
        message_chunk : LLM 吐出的一小片（.content 是这一小段文本）
        metadata      : 字典，告诉你这片来自哪个节点、哪次模型调用
    → 它是同步的，比 astream_events 简单：一个 for 循环 + 拼 .content 就行

对比〔09_流式输出/01〕：
  stream_mode="updates"  → 节点级，llm_node 整段回答一次性给
  stream_mode="messages" → token 级，LLM 边生成边吐，逐片到手
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

# streaming=True：让底层模型开启流式，token 才会一片片来
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    streaming=True,
)


# ── State 和节点：和〔02_状态State/02〕一样 ────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]


def chat_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
graph = builder.compile()


# ── 运行：逐 token 打印 ────────────────────────────────────────────────────
if __name__ == "__main__":
    inputs = {"messages": [HumanMessage(content="用三句话介绍一下 Python 语言")]}

    print("▶ 打字机输出：\n")
    for message_chunk, metadata in graph.stream(inputs, stream_mode="messages"):
        # message_chunk.content 是这一小片文本，逐片打印、不换行
        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)
    print("\n\n▶ 输出完毕")

    # ── 看看 metadata 里有什么（调试用）──────────────────────────────────
    print("\n── 每片的来源 metadata（取前几片看看）──────────")
    for i, (chunk, metadata) in enumerate(graph.stream(inputs, stream_mode="messages")):
        if i >= 3:
            break
        print(f"  片{i}: content={chunk.content!r}  来自节点={metadata.get('langgraph_node')}")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
为什么要 streaming=True？
  ChatOpenAI(streaming=True) 让模型按流式返回；
  否则模型一次性返回整段，stream_mode="messages" 也只能拿到一大片。

(message_chunk, metadata) 元组：
  message_chunk  是 AIMessageChunk，.content 是这一小段文本
  metadata       常用字段：
      langgraph_node  这片来自哪个节点（多节点图里用来区分）
      ls_model_name   哪个模型

只想要某个节点的 token：
  if metadata.get("langgraph_node") == "chat":
      print(message_chunk.content, end="")

★ 三种"流"的取舍：
  stream_mode="updates"   节点级，看每步状态变化（〔09_流式输出/01〕）
  stream_mode="messages"  token 级，做打字机最省事（本课）
  astream_events          token 级 + 全事件类型，最细但要 async（〔09_流式输出/04〕）
  日常做打字机 UI，优先用本课的 messages 模式。
"""
