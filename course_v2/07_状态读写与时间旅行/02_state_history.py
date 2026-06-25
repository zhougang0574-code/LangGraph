"""
【07 状态读写与时间旅行 / 02】get_state_history() —— 查看所有历史快照
===================================================
〔07_状态读写与时间旅行/01〕的 get_state() 只能看"当前"State。
本课引入 get_state_history()：看这个 thread 里每一步执行后的所有 State 快照。

新概念（只有这一个）：
  graph.get_state_history(config)
    → 返回迭代器，从最新到最旧，每个元素是一个 StateSnapshot
    → 每个节点执行完都会产生一个快照（checkpointer 存的就是这些）

用途：
  1. 调试：看每一步 State 是怎么变化的
  2. 时光回溯：找到某个历史快照，从那里重新分支执行

前置条件：需要 checkpointer（和〔06_持久化与记忆/01〕一样）
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ── State 和节点：和〔01_基础/02〕类似的三节点流水线 ──────────────────────────────
class State(TypedDict):
    question: str
    answer: str
    word_count: int


def llm_node(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content="用一句话简洁回答。"),
        HumanMessage(content=state["question"]),
    ])
    return {"answer": response.content}


def count_node(state: State) -> dict:
    return {"word_count": len(state["answer"])}


def print_node(state: State) -> dict:
    print(f"问题：{state['question']}")
    print(f"回答：{state['answer']}")
    print(f"字数：{state['word_count']}")
    return {}


builder = StateGraph(State)
builder.add_node("llm_node",   llm_node)
builder.add_node("count_node", count_node)
builder.add_node("print_node", print_node)
builder.add_edge(START,        "llm_node")
builder.add_edge("llm_node",   "count_node")
builder.add_edge("count_node", "print_node")
builder.add_edge("print_node", END)

graph = builder.compile(checkpointer=MemorySaver())


# ── 运行 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "history-demo"}}

    print("▶ 执行 Graph...")
    graph.invoke({"question": "什么是机器学习？"}, config=config)

    # ── ★ get_state_history()：查看所有历史快照 ──────────────────────────
    print("\n── 所有历史快照（从最新到最旧）────────────────────────────")
    history = list(graph.get_state_history(config))

    for i, snapshot in enumerate(history):
        print(f"\n快照 {i}：")
        print(f"  下一步节点：{snapshot.next}")
        print(f"  State 值：  {snapshot.values}")

    # ── 时光回溯：找到某个历史快照，从那里重新执行 ─────────────────────
    print("\n── 时光回溯：从 count_node 之前的快照重新跑 ────────────────")

    # 找到"下一步是 count_node"的快照（即 llm_node 刚跑完的那一刻）
    target = None
    for snapshot in history:
        if snapshot.next == ("count_node",):
            target = snapshot
            break

    if target:
        print(f"找到目标快照，answer = {target.values.get('answer', '')[:40]}...")
        # 用那个快照的 config 重新 invoke，Graph 从 count_node 开始跑
        result = graph.invoke(None, config=target.config)
        print("重新执行完毕，word_count =", result["word_count"])


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
get_state_history() 返回的快照顺序（从新到旧）：

  快照 0：next=()           ← Graph 已结束（print_node 跑完后）
  快照 1：next=("print_node",) ← count_node 跑完，下一步是 print_node
  快照 2：next=("count_node",) ← llm_node 跑完，下一步是 count_node
  快照 3：next=("llm_node",)  ← 刚开始，下一步是 llm_node
  快照 4：next=("__start__",) ← 初始输入状态

时光回溯的用法：
  找到某个历史快照 snapshot
  用 snapshot.config（它有自己的 checkpoint_id）重新 invoke
  Graph 从那个节点开始执行，跳过之前已经跑过的节点

典型场景：
  LLM 生成的答案不满意 → 找到 llm_node 之前的快照 → 重新 invoke
  → LLM 重新生成（因为 LLM 有随机性，结果可能不同）

get_state() vs get_state_history()：
  get_state()         → 只看"当前"这一个快照
  get_state_history() → 看这个 thread 里全部历史快照
"""
