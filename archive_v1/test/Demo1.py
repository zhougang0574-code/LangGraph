"""
Demo1：Multi-Agent + create_agent
=================================
第17课的子 Agent 用的是手写子图（自己搭 StateGraph + ReAct 循环）。
本 Demo 改用 create_agent（第14课）替代，更简洁。

对比：
  第17课子图写法：自己定义 SubState → 写 agent 节点 → 搭 builder → compile()
  本 Demo 写法：  create_agent(llm, tools) 一行搞定，再包一层函数节点做 State 转换

关键点：
  create_agent 内部用的是 MessagesState（messages 字段）
  外层 State 用的是自定义字段（task / result）
  所以需要在函数节点里做一次转换：
    输入：把 state["task"] 包成 HumanMessage 传给 create_agent
    输出：从 create_agent 返回的 messages[-1] 取出文本写回 result
"""

import os
from typing import TypedDict, Literal

import dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

dotenv.load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ══════════════════════════════════════════════════════
# 1. 共享 State
# ══════════════════════════════════════════════════════
class State(TypedDict):
    task: str    # 用户的任务
    result: str  # 子 Agent 的执行结果
    next: str    # Supervisor 的路由决策


# ══════════════════════════════════════════════════════
# 2. 子 Agent A：数学计算
#    create_agent 一行创建，内部自动处理工具调用循环
# ══════════════════════════════════════════════════════
@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积"""
    return a * b

math_react = create_agent(llm, [add, multiply])

def math_agent_node(state: State) -> dict:
    # create_agent 用 MessagesState，把 task 转成 messages 格式传入
    response = math_react.invoke({
        "messages": [HumanMessage(content=state["task"])]
    })
    # 取最后一条消息（AI 的最终回答）写回外层 State
    return {"result": f"[数学Agent] {response['messages'][-1].content}"}


# ══════════════════════════════════════════════════════
# 3. 子 Agent B：文字写作
#    无工具，传空列表，用 system_prompt 定制角色
# ══════════════════════════════════════════════════════
writer_react = create_agent(
    llm,
    [],
    system_prompt="你是写作助手，根据要求生成简洁的文字内容，100字以内。"
)

def writer_agent_node(state: State) -> dict:
    response = writer_react.invoke({
        "messages": [HumanMessage(content=state["task"])]
    })
    return {"result": f"[写作Agent] {response['messages'][-1].content}"}


# ══════════════════════════════════════════════════════
# 4. Supervisor：LLM 决定路由给哪个子 Agent
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

    if decision not in ("math_agent", "writer_agent"):
        decision = "FINISH"

    return {"next": decision}


def route(state: State) -> Literal["math_agent", "writer_agent", "__end__"]:
    if state["next"] == "FINISH":
        return "__end__"
    return state["next"]


# ══════════════════════════════════════════════════════
# 5. 外层 Graph
# ══════════════════════════════════════════════════════
builder = StateGraph(State)
builder.add_node("supervisor",    supervisor_node)
builder.add_node("math_agent",   math_agent_node)
builder.add_node("writer_agent", writer_agent_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route)
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
    result = graph.invoke({"task": "计算 36 乘以 7 加上 100 等于多少", "result": "", "next": ""})
    print("最终结果：", result["result"])

    print("\n" + "=" * 50)
    print("任务2：文字写作")
    print("=" * 50)
    result = graph.invoke({"task": "写一段介绍 Python 的简短文字", "result": "", "next": ""})
    print("最终结果：", result["result"])
