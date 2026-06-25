"""
【09 流式输出 / 01】节点级流式 —— updates / values / 多模式
=========================================================
前面所有课都用 graph.invoke()：等整个 Graph 跑完，一次性返回最终 State。
本课引入 graph.stream()：每个节点执行完立刻返回它的输出（节点级粒度）。

新概念（只有这一个）：
  graph.stream(input, config, stream_mode=...)
    - 返回一个迭代器，每次 yield 一个 chunk
    - stream_mode="updates"：每个 chunk 是 {节点名: 该节点返回的 dict}（只看变化）
    - stream_mode="values" ：每个 chunk 是完整的 State 快照（看全量）
    - stream_mode=["values","updates"]：同时要多种，chunk 变成 (模式名, 数据) 元组
    - stream_mode="debug" ：最啰嗦，连每个任务的开始/结束都给（深度调试用）

invoke() vs stream() 对比：
  invoke()  → 等所有节点都跑完，返回最终 State
  stream()  → 每个节点跑完就 yield 一次，实时拿到中间结果

本课讲的是"节点级"流式；想要 LLM 逐 token 的"打字机效果"，
见〔09_流式输出/02〕（stream messages）和〔09_流式输出/04〕（astream_events）。
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


# ── State 和节点：和〔02_状态State/02〕完全一样 ────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def llm_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def print_node(state: State) -> dict:
    last = state["messages"][-1]
    print(f"[print_node] 字数：{len(last.content)}")
    return {}


builder = StateGraph(State)
builder.add_node("llm_node",   llm_node)
builder.add_node("print_node", print_node)
builder.add_edge(START,        "llm_node")
builder.add_edge("llm_node",   "print_node")
builder.add_edge("print_node", END)

graph = builder.compile()


# ── 运行 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    input_state = {"messages": [HumanMessage(content="用3句话介绍Python")]}

    # ── 方式一：invoke()（回顾）─────────────────────────────────────────
    print("=" * 50)
    print("invoke()：等全部完成后一次性返回")
    print("=" * 50)
    result = graph.invoke(input_state)
    print("最终回答：", result["messages"][-1].content[:60], "...")

    # ── 方式二：stream() with stream_mode="updates" ───────────────────
    print("\n" + "=" * 50)
    print('stream(stream_mode="updates")：每个节点完成就返回一次')
    print("=" * 50)
    for chunk in graph.stream(input_state, stream_mode="updates"):
        # chunk 是一个 dict：{节点名: 该节点返回的内容}
        node_name = list(chunk.keys())[0]
        node_output = chunk[node_name]
        print(f"  节点 [{node_name}] 输出：{node_output}")

    # ── 方式三：stream() with stream_mode="values" ────────────────────
    print("\n" + "=" * 50)
    print('stream(stream_mode="values")：每个节点完成后返回完整 State 快照')
    print("=" * 50)
    for i, snapshot in enumerate(graph.stream(input_state, stream_mode="values")):
        msg_count = len(snapshot["messages"])
        print(f"  快照 {i}：messages 共 {msg_count} 条")

    # ── 方式四：同时要多种模式 stream_mode=["values","updates"] ─────────
    print("\n" + "=" * 50)
    print('stream(stream_mode=["values","updates"])：chunk 变成 (模式名, 数据)')
    print("=" * 50)
    for mode, chunk in graph.stream(input_state, stream_mode=["values", "updates"]):
        # 多模式时，每个 chunk 是元组：第一个元素告诉你它属于哪种模式
        print(f"  [{mode}] keys={list(chunk.keys())}")

    # ── 方式五：debug 模式（最详细，调试用）────────────────────────────
    print("\n" + "=" * 50)
    print('stream(stream_mode="debug")：每个任务的开始/结束都给')
    print("=" * 50)
    for chunk in graph.stream(input_state, stream_mode="debug"):
        # debug chunk 形如 {"type": "task"/"task_result", "timestamp":..., "step":...}
        print(f"  type={chunk.get('type')}  step={chunk.get('step')}")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
stream_mode="updates" 的输出格式：

  第1个 chunk（llm_node 完成后）：
    {"llm_node": {"messages": [AIMessage("Python 是...")]}}

  第2个 chunk（print_node 完成后）：
    {"print_node": {}}    ← print_node 返回空 dict

stream_mode="values" 的输出格式：

  第0个 snapshot（初始 State，invoke 输入）：
    {"messages": [HumanMessage("用3句话介绍Python")]}

  第1个 snapshot（llm_node 完成后）：
    {"messages": [HumanMessage(...), AIMessage(...)]}

  第2个 snapshot（print_node 完成后）：
    {"messages": [HumanMessage(...), AIMessage(...)]}  ← print_node 没改 State，所以一样

多模式 / debug：
  stream_mode=["values","updates"] → 每个 chunk 是 (模式名, 数据)，
    靠第一个元素区分这条是哪种模式的数据；要几种就传几种。
  stream_mode="debug" → 连每个任务的开始(task)/结束(task_result)都吐，
    平时太啰嗦，排查"卡在哪一步 / 谁报错"时很有用。

★ 各模式速记：
  updates  只看每步"变了什么"（最常用）
  values   每步看"完整 State"
  [..]     多种一起要，chunk 带模式名
  debug    任务级最详细日志
  messages 见〔09_流式输出/02〕：逐 token（打字机）
  custom   见〔09_流式输出/03〕：节点里自己往流里塞进度

什么时候用 stream()：
  - 多节点 Pipeline，想知道哪个节点慢
  - 调试时看每个节点的输出
  - 前端需要"打字机效果"（逐步显示，不等全部完成）
  - ReAct Agent 多轮循环，想实时知道每一步在做什么
"""
