"""
【09 流式输出 / 03】stream custom —— 节点里自己往流里塞进度
=========================================================
前面的流式都是"框架替你吐"：updates 吐状态变化、messages 吐 token。
但有时你想在一个节点内部，主动往外推自定义信息——比如：
  "正在检索…"→"正在生成…"→"正在校对…" 这种进度条文案。

新概念（只有这一个）：
  在节点里用 get_stream_writer() 拿到一个 writer，调用它发送任意数据：
    writer = get_stream_writer()
    writer({"progress": "步骤1：检索"})

  外面用 stream_mode="custom" 接收这些数据：
    for chunk in graph.stream(inputs, stream_mode="custom"):
        ...

  可以和别的模式组合，但列表里至少要有 "custom"：
    stream_mode=["updates", "custom"]   → chunk 变成 (模式名, 数据) 元组

对比前两课：
  updates/messages → 框架自动产生的流
  custom           → 你在节点里手动 writer(...) 产生的流
"""

from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END


# ── State ─────────────────────────────────────────────────────────────────
class State(TypedDict):
    query: str
    answer: str


# ── 节点：在内部多次 writer(...) 推送进度 ──────────────────────────────────
def work_node(state: State) -> dict:
    writer = get_stream_writer()        # ★ 拿到流写入器

    writer({"progress": "步骤1：分析查询", "status": "running"})
    # ……这里本该做真正的活，演示就略过……
    writer({"progress": "步骤2：生成结果", "status": "running"})
    writer({"progress": "步骤3：完成", "status": "done"})

    return {"answer": f"处理结果：{state['query'].upper()}"}


builder = StateGraph(State)
builder.add_node("work", work_node)
builder.add_edge(START, "work")
builder.add_edge("work", END)
graph = builder.compile()


# ── 运行 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    inputs = {"query": "hello world", "answer": ""}

    # ── 只收 custom：拿到的就是节点里 writer(...) 发的那些 dict ───────────
    print("── stream_mode='custom' ──────────────────────")
    for chunk in graph.stream(inputs, stream_mode="custom"):
        print("  收到进度：", chunk)

    # ── custom + updates 组合：chunk 变成 (模式名, 数据) ──────────────────
    print("\n── stream_mode=['custom','updates'] ──────────")
    for mode, chunk in graph.stream(inputs, stream_mode=["custom", "updates"]):
        print(f"  [{mode}] {chunk}")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
get_stream_writer() 的用法：
  - 只能在"节点函数内部"调用（它靠运行时上下文拿到当前这条流）
  - writer(任意可序列化对象)，想发几次发几次
  - 发的内容完全由你定，框架原样透传给 stream 的消费方

接收端：
  stream_mode="custom"            → chunk 就是你 writer 发的那个对象
  stream_mode=["custom","updates"] → chunk 是 (模式名, 数据)；
                                     列表里至少要有一个 "custom"

★ 典型场景：
  长任务的进度条 / 阶段提示 / 中间产物预览——
  这些既不是"状态变化"（updates 管不了），也不是"LLM token"（messages 管不了），
  就用 custom 自己往流里塞。

  updates/values（〔09_流式输出/01〕）：框架产生
  messages（〔09_流式输出/02〕）：       框架产生（LLM token）
  custom（本课）：                       你在节点里手动产生
"""
