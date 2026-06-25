"""
【08 中断与人工干预 / 01】Human-in-the-loop —— 暂停等人类决定，再继续
======================================================
前几课的 Graph 都是自动跑完的，人无法干预中间过程。
本课引入 interrupt()：在节点内部暂停 Graph，把控制权交给人类，
人类做出决定后，再传入 Command(resume=...) 继续执行。

新概念（两个，但是配套使用，算一套）：
  interrupt(value)       —— 在节点内部暂停 Graph，把 value 展示给调用方
  Command(resume=value)  —— 调用方传入，让暂停的 Graph 继续，并把 value 返回给 interrupt()

前置条件：
  interrupt() 必须配合 checkpointer 使用（和〔06_持久化与记忆/01〕的 MemorySaver 一样）
  原因：暂停时需要把当前 State 存起来，恢复时才能接着跑

流程：
  第1次 invoke()  → 跑到 interrupt() 暂停，返回 Interrupt 对象
  人类查看 / 输入
  第2次 invoke(Command(resume=...))  → 从暂停处继续，interrupt() 返回人类输入的值
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


# ── 1. State ──────────────────────────────────────────────────────────────
class State(TypedDict):
    question: str   # 用户问题
    plan: str       # LLM 生成的方案
    approved: bool  # 人类是否批准


# ── 2. 节点 ───────────────────────────────────────────────────────────────
def plan_node(state: State) -> dict:
    """LLM 生成方案"""
    response = llm.invoke([
        SystemMessage(content="你是一个助手，用2-3句话给出解决方案，简洁清晰。"),
        HumanMessage(content=state["question"]),
    ])
    return {"plan": response.content}


def review_node(state: State) -> dict:
    """暂停，等人类审批"""
    print("\n── AI 生成的方案 ──────────────────────")
    print(state["plan"])
    print("────────────────────────────────────────")

    # ★ 核心：interrupt() 暂停 Graph，把提示信息传给调用方
    #   Graph 停在这里，等待 Command(resume=...) 传入
    #   人类输入的值会作为 interrupt() 的返回值
    decision = interrupt("请输入 yes 批准，其他任意内容拒绝：")

    approved = decision.strip().lower() == "yes"
    return {"approved": approved}


def execute_node(state: State) -> dict:
    """根据审批结果决定下一步"""
    if state["approved"]:
        print("\n✓ 方案已批准，开始执行...")
        print(f"执行方案：{state['plan']}")
    else:
        print("\n✗ 方案被拒绝，流程终止。")
    return {}


# ── 3. Graph（线性结构，和〔01_基础/02〕一样）──────────────────────────────────────
builder = StateGraph(State)
builder.add_node("plan_node",    plan_node)
builder.add_node("review_node",  review_node)
builder.add_node("execute_node", execute_node)
builder.add_edge(START,          "plan_node")
builder.add_edge("plan_node",    "review_node")
builder.add_edge("review_node",  "execute_node")
builder.add_edge("execute_node", END)

# interrupt() 必须配合 checkpointer
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# ── 4. 运行 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "review-1"}}

    print("第1次 invoke：Graph 跑到 interrupt() 暂停")
    result = graph.invoke(
        {"question": "如何用 Python 读取一个 CSV 文件？"},
        config=config,
    )
    # Graph 暂停后返回，result 里包含 __interrupt__ 信息
    print("\nGraph 已暂停，等待人类输入...")
    print("（暂停信息：", result, "）")

    # 模拟人类输入
    human_input = input("\n请输入你的决定（yes/no）：")

    print("\n第2次 invoke：传入 Command(resume=...) 继续")
    result = graph.invoke(
        Command(resume=human_input),
        config=config,
    )
    print("\nGraph 执行完毕。")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
interrupt() 的执行流程：

  第1次 invoke():
    plan_node 运行完，生成 plan
    进入 review_node，打印方案
    执行到 interrupt("请输入...")
      → Graph 暂停，把提示保存到 checkpoint
      → invoke() 返回（不是 END，是中途暂停）

  人类看到方案，决定输入 "yes" 或其他

  第2次 invoke(Command(resume="yes")):
    从 checkpoint 恢复，回到 review_node 的 interrupt() 调用处
    interrupt() 返回 "yes"（即 Command(resume=...) 传入的值）
    review_node 继续执行：approved = True
    execute_node 运行，打印执行信息
    到达 END，Graph 完成

interrupt() vs input()：
  普通 Python input()：阻塞当前进程，等待终端输入
  LangGraph interrupt()：暂停 Graph，可以跨进程、跨请求、跨时间恢复
                         适合 Web 应用（用户点按钮来恢复，而不是终端输入）
"""
