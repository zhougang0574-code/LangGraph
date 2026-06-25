"""
【08 中断与人工干预 / 02】interrupt_before / interrupt_after —— 外部断点
===========================================================
〔08_中断与人工干预/01〕在节点内部写 interrupt()，节点代码里有硬编码的暂停逻辑。

新概念（只有这一个）：
  graph.compile(interrupt_before=["节点名"])
  graph.compile(interrupt_after=["节点名"])

  在 compile 阶段声明断点，节点代码完全不需要改动。
  Graph 在执行到指定节点之前（或之后）自动暂停。
  恢复时直接 graph.invoke(None, config=config)，不需要 Command(resume=...)。

对比〔08_中断与人工干预/01〕：
  interrupt()（节点内）   → 节点代码里写，节点可以拿到人类输入的值（Command.resume）
  interrupt_before/after  → compile 时声明，节点代码不变，恢复时从断点继续跑

适合场景：
  不想改节点代码，只是想在某个节点前/后插入人工审核
  比如：plan → [审核] → execute → review
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


# ── State 和节点（节点代码完全不涉及 interrupt）────────────
class State(TypedDict):
    task: str
    plan: str
    result: str


def plan_node(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content="你是规划助手，根据任务制定一个简短的执行计划，50字以内。"),
        HumanMessage(content=state["task"]),
    ])
    print(f"[plan_node] 计划：{response.content}")
    return {"plan": response.content}


def execute_node(state: State) -> dict:
    # 这个节点代码里没有任何 interrupt，断点是外部声明的
    response = llm.invoke([
        SystemMessage(content="你是执行助手，按照计划执行任务并给出结果，100字以内。"),
        HumanMessage(content=f"任务：{state['task']}\n计划：{state['plan']}"),
    ])
    print(f"[execute_node] 结果：{response.content}")
    return {"result": response.content}


def review_node(state: State) -> dict:
    print(f"[review_node] 最终结果已就绪")
    return {}


builder = StateGraph(State)
builder.add_node("plan_node",    plan_node)
builder.add_node("execute_node", execute_node)
builder.add_node("review_node",  review_node)
builder.add_edge(START,          "plan_node")
builder.add_edge("plan_node",    "execute_node")
builder.add_edge("execute_node", "review_node")
builder.add_edge("review_node",  END)

# ★ 在 execute_node 执行前暂停，等人工审核 plan
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["execute_node"],   # ★ 新增，节点代码不变
)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "task-001"}}

    # ── 第一次 invoke：跑到 execute_node 之前自动暂停 ──────
    print("=" * 50)
    print("▶ 第一次 invoke（跑到断点暂停）")
    print("=" * 50)
    graph.invoke({"task": "写一篇关于 Python 的短文", "plan": "", "result": ""}, config=config)

    # 查看当前状态：Graph 停在 execute_node 之前
    snapshot = graph.get_state(config)
    print(f"\n当前暂停位置，下一步：{snapshot.next}")
    print(f"已生成的计划：{snapshot.values['plan'][:50]}...")

    # ── 人工审核（此处模拟审核通过）──────────────────────
    human_decision = input("\n计划是否通过审核？(yes/no): ").strip().lower()

    if human_decision == "yes":
        print("\n▶ 审核通过，继续执行")
        # 恢复：传 None 即可，不需要 Command(resume=...)
        result = graph.invoke(None, config=config)
        print(f"\n最终结果：{result['result'][:80]}...")
    else:
        print("\n✗ 审核未通过，流程终止")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
interrupt_before vs interrupt_after：

  interrupt_before=["execute_node"]
    → plan_node 跑完后，execute_node 还没开始时暂停
    → snapshot.next == ("execute_node",)
    → 恢复后从 execute_node 开始继续

  interrupt_after=["plan_node"]
    → plan_node 跑完后立刻暂停（效果和上面一样，语义不同）
    → snapshot.next == ("execute_node",)

〔08_中断与人工干预/01〕 interrupt() vs 本课 interrupt_before：

  interrupt()（节点内写）：
    节点代码里有暂停逻辑
    节点可以拿到人类输入的值（通过 Command.resume=value 传回）
    恢复：graph.invoke(Command(resume=人类输入), config=config)
    适合：需要人类提供信息，节点根据这个信息做决策

  interrupt_before（compile 时声明）：
    节点代码完全不变
    不传值给节点，只是单纯暂停等审核
    恢复：graph.invoke(None, config=config)
    适合：只需要人工"放行"，不需要传额外信息给节点
"""
