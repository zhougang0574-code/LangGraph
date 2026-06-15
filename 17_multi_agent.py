"""
第十七课：Multi-Agent —— Supervisor 控制多个子 Agent
====================================================
把前面学过的积木拼在一起：
  第4课  条件路由        → Supervisor 根据 LLM 决策路由
  第8课  ReAct Agent    → 每个子 Agent 内部是完整的工具调用循环
  第16课 子图当节点      → 每个子 Agent 是一个子图

新概念（只有这一个）：
  Supervisor 模式
    → Supervisor 节点：LLM 读取任务，输出下一步去哪个 Agent
    → 子 Agent 跑完后结果写回共享 State，再交还给 Supervisor
    → Supervisor 再次判断：继续分配任务 or 结束

架构：
  START → supervisor → math_agent  → supervisor → ...
                    ↘ writer_agent → supervisor → ...
                    ↘ END（Supervisor 认为任务完成）
"""

import os
from typing import TypedDict, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ══════════════════════════════════════════════════════
# 1. 共享 State
# ══════════════════════════════════════════════════════
class State(TypedDict):
    task: str        # 用户的任务
    result: str      # 子 Agent 的执行结果（逐步累积）
    next: str        # Supervisor 的路由决策


# ══════════════════════════════════════════════════════
# 2. 子 Agent A：数学计算（有工具的 ReAct Agent）
# ══════════════════════════════════════════════════════
@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积"""
    return a * b

math_tools = [add, multiply]
math_llm = llm.bind_tools(math_tools)

# 子 Agent 有自己的内部 State（同名字段和外层 State 通信）
class MathState(TypedDict):
    task: str    # 从外层传入
    result: str  # 写回外层

def math_agent_node(state: MathState) -> dict:
    response = math_llm.invoke([
        SystemMessage(content="你是数学助手，用工具计算后给出结果，简洁回答。"),
        HumanMessage(content=state["task"]),
    ])
    return {"result": response.content if response.content else str(response.tool_calls)}

def math_tool_node(state: MathState) -> dict:
    # 手动执行工具（简化版，直接用 ToolNode 逻辑）
    last = state.get("_last_response")
    return {}

# 用更简洁的方式：直接让 LLM 调用工具并返回最终结果
def math_agent_full(state: MathState) -> dict:
    from langgraph.prebuilt import ToolNode
    messages = [
        SystemMessage(content="你是数学助手，用工具计算，只输出最终数字结果和一句说明。"),
        HumanMessage(content=state["task"]),
    ]
    # 简单循环：让 LLM 调用工具直到得出答案
    while True:
        response = math_llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return {"result": f"[数学Agent] {response.content}"}
        # 执行工具
        tool_map = {t.name: t for t in math_tools}
        for tc in response.tool_calls:
            tool_result = tool_map[tc["name"]].invoke(tc["args"])
            from langchain_core.messages import ToolMessage
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tc["id"],
                name=tc["name"],
            ))

math_builder = StateGraph(MathState)
math_builder.add_node("agent", math_agent_full)
math_builder.add_edge(START, "agent")
math_builder.add_edge("agent", END)
math_subgraph = math_builder.compile()


# ══════════════════════════════════════════════════════
# 3. 子 Agent B：文字写作（纯 LLM，无工具）
# ══════════════════════════════════════════════════════
class WriterState(TypedDict):
    task: str    # 从外层传入
    result: str  # 写回外层

def writer_agent_node(state: WriterState) -> dict:
    response = llm.invoke([
        SystemMessage(content="你是写作助手，根据要求生成简洁的文字内容，100字以内。"),
        HumanMessage(content=state["task"]),
    ])
    return {"result": f"[写作Agent] {response.content}"}

writer_builder = StateGraph(WriterState)
writer_builder.add_node("agent", writer_agent_node)
writer_builder.add_edge(START, "agent")
writer_builder.add_edge("agent", END)
writer_subgraph = writer_builder.compile()


# ══════════════════════════════════════════════════════
# 4. Supervisor：LLM 决定路由给哪个 Agent
# ══════════════════════════════════════════════════════
def supervisor_node(state: State) -> dict:
    task = state["task"]
    result = state.get("result", "")

    prompt = f"""你是任务调度员，根据任务内容决定下一步。

任务：{task}
已有结果：{result if result else "（还没有）"}

可选项：
- math_agent：需要数学计算时选择
- writer_agent：需要写作、生成文字时选择
- FINISH：任务已完成，可以结束

只输出一个词：math_agent 或 writer_agent 或 FINISH"""

    response = llm.invoke([HumanMessage(content=prompt)])
    decision = response.content.strip()

    # 容错：如果输出不在预期范围内，结束
    if decision not in ("math_agent", "writer_agent"):
        decision = "FINISH"

    return {"next": decision}


def route(state: State) -> Literal["math_agent", "writer_agent", "__end__"]:
    if state["next"] == "FINISH":
        return "__end__"
    return state["next"]


# ══════════════════════════════════════════════════════
# 5. 外层 Graph：Supervisor 控制两个子 Agent
# ══════════════════════════════════════════════════════
builder = StateGraph(State)
builder.add_node("supervisor",    supervisor_node)
builder.add_node("math_agent",   math_subgraph)    # 子图当节点
builder.add_node("writer_agent", writer_subgraph)  # 子图当节点

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route)

# 子 Agent 跑完后，结果写回 State，再交还给 Supervisor
builder.add_edge("math_agent",   "supervisor")
builder.add_edge("writer_agent", "supervisor")

graph = builder.compile()


# ══════════════════════════════════════════════════════
# 6. 运行
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("任务1：数学计算")
    print("=" * 50)
    result = graph.invoke({
        "task": "计算 36 乘以 7 加上 100 等于多少",
        "result": "",
        "next": "",
    })
    print("最终结果：", result["result"])

    print("\n" + "=" * 50)
    print("任务2：文字写作")
    print("=" * 50)
    result = graph.invoke({
        "task": "写一段介绍 Python 的简短文字",
        "result": "",
        "next": "",
    })
    print("最终结果：", result["result"])


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
执行流程（以数学任务为例）：

  supervisor（第1次）：
    LLM 看到"计算 36 乘以 7 加上 100"
    输出 next = "math_agent"

  math_agent（子图）：
    内部 ReAct 循环：调用 multiply(36,7)=252，add(252,100)=352
    result = "[数学Agent] 结果是 352"
    写回外层 State

  supervisor（第2次）：
    LLM 看到 result 已有答案
    输出 next = "FINISH"

  END

Supervisor 模式的核心：
  1. Supervisor 是唯一的"决策者"，子 Agent 只负责执行
  2. 子 Agent 跑完后结果写回共享 State，Supervisor 再次判断
  3. Supervisor 可以多次调用不同子 Agent，最终决定何时结束
"""
