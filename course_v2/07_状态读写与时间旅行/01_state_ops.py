"""
【07 状态读写与时间旅行 / 01】外部访问 State —— get_state() + update_state()
=========================================================
〔08_中断与人工干预/01〕学了 interrupt()：Graph 暂停，等人类决定。
但暂停后能做的不只是"批准/拒绝"——还可以查看当前 State、直接修改 State，
然后再决定是否继续。

新概念（一套两件，配合使用）：
  graph.get_state(config)
    → 返回 StateSnapshot，可以看当前 State 的值、下一步要跑哪个节点
  graph.update_state(config, values)
    → 直接从外部修改 State 的某些字段，不需要通过节点

典型使用场景：
  Graph 在 interrupt() 暂停
  → get_state() 查看 LLM 生成了什么
  → 发现方案有问题，update_state() 直接改掉
  → Command(resume="yes") 继续（此时 execute_node 拿到的是修改后的方案）
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ── State、节点、Graph 和〔08_中断与人工干预/01〕完全一样 ─────────────────────────────────
class State(TypedDict):
    question: str
    plan: str
    approved: bool


def plan_node(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content="你是助手，用2-3句话给出解决方案，简洁清晰。"),
        HumanMessage(content=state["question"]),
    ])
    return {"plan": response.content}


def review_node(state: State) -> dict:
    decision = interrupt("等待人类审批...")
    return {"approved": decision.strip().lower() == "yes"}


def execute_node(state: State) -> dict:
    if state["approved"]:
        print("\n✓ 执行方案：")
        print(state["plan"])
    else:
        print("\n✗ 方案被拒绝。")
    return {}


builder = StateGraph(State)
builder.add_node("plan_node",    plan_node)
builder.add_node("review_node",  review_node)
builder.add_node("execute_node", execute_node)
builder.add_edge(START,          "plan_node")
builder.add_edge("plan_node",    "review_node")
builder.add_edge("review_node",  "execute_node")
builder.add_edge("execute_node", END)

graph = builder.compile(checkpointer=MemorySaver())


# ── 运行：展示 get_state() 和 update_state() ─────────────────────────────
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "ops-demo"}}

    # 第1次 invoke：跑到 interrupt() 暂停
    print("▶ 第1次 invoke，Graph 跑到 interrupt() 暂停")
    graph.invoke({"question": "如何优化一个慢查询 SQL？"}, config=config)

    # ── ★ get_state()：查看当前 State ────────────────────────────────────
    print("\n── get_state() ──────────────────────────────")
    snapshot = graph.get_state(config)

    print("当前 State 值：")
    for key, value in snapshot.values.items():
        display = str(value)[:80] + "..." if len(str(value)) > 80 else value
        print(f"  {key}: {display}")

    print("\n下一步将执行：", snapshot.next)
    # snapshot.next 是一个 tuple，暂停时显示即将执行的节点名

    # ── ★ update_state()：直接修改 State ─────────────────────────────────
    print("\n── update_state()：把 plan 替换成人工修改的版本 ─────────────────")
    graph.update_state(
        config,
        {"plan": "【人工修改】使用索引优化：1) 为 WHERE 字段加索引 2) 避免 SELECT *"},
    )

    # 再次 get_state() 验证修改生效
    snapshot_after = graph.get_state(config)
    print("修改后 plan：", snapshot_after.values["plan"])

    # ── 恢复执行 ──────────────────────────────────────────────────────────
    print("\n▶ 第2次 invoke，传入 Command(resume='yes') 继续")
    graph.invoke(Command(resume="yes"), config=config)
    # execute_node 拿到的 plan 是修改后的版本


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
StateSnapshot 的主要字段：

  snapshot.values       → 当前 State 的所有字段值（dict）
  snapshot.next         → 下一步要运行的节点名 tuple；Graph 结束后是空 tuple ()
  snapshot.tasks        → 当前挂起的任务列表（interrupt 时有值）
  snapshot.config       → 这个 snapshot 对应的 config

update_state() 的行为：
  直接写入 checkpointer，不执行任何节点
  只更新你指定的字段，其他字段保持不变
  等效于"在外部悄悄改了 State"，节点恢复执行时看到的就是改后的值

get_state() 和 update_state() 不需要 interrupt()——
  即使 Graph 正在运行中，也可以查看 State（但通常在暂停时用更有意义）
"""
