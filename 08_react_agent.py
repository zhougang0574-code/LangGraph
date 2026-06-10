"""
第八课：ReAct Agent —— 工具真正被执行，循环直到结束
====================================================
与第七课的区别：
  第七课  LLM 声明要调用工具，但工具没有被执行
  第八课  工具被自动执行，结果反馈给 LLM，LLM 再决定下一步

新概念（这三个是一套，合为一个概念）：
  ToolNode       —— 内置节点，读取 tool_calls，自动执行工具，结果追加为 ToolMessage
  tools_condition —— 内置路由函数，有 tool_calls → 去 tools，没有 → END
  循环结构        —— agent → tools → agent → ... → END

ReAct 的含义：
  Reason（推理）：LLM 思考要做什么
  Act（行动）   ：调用工具执行
  Observe（观察）：把工具结果喂回 LLM
  循环，直到 LLM 认为不需要再调用工具
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ── 1. 工具定义（和第七课一样）────────────────────────────────────────────
@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和"""
    result = a + b
    print(f"  [工具执行] {a} + {b} = {result}")
    return result


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积"""
    result = a * b
    print(f"  [工具执行] {a} × {b} = {result}")
    return result


tools = [add, multiply]
llm_with_tools = llm.bind_tools(tools)


# ── 2. State（和第六、七课一样）──────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── 3. agent 节点（和第七课的 llm_node 一样）──────────────────────────────
def agent_node(state: State) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ── 4. 构建 Graph：核心变化在这里 ─────────────────────────────────────────
builder = StateGraph(State)

builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))  # 内置节点，传入工具列表即可

builder.add_edge(START, "agent")

# tools_condition 是内置路由函数：
#   最后一条消息有 tool_calls → 返回 "tools"（去执行工具）
#   最后一条消息没有 tool_calls → 返回 END（直接结束）
builder.add_conditional_edges("agent", tools_condition)

# 工具执行完后，把结果追加到 messages，再回到 agent
# 这就是循环：agent → tools → agent → tools → ... → END
builder.add_edge("tools", "agent")

graph = builder.compile()


# ── 5. 运行 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("问题1：一步工具调用")
    print("=" * 50)
    result = graph.invoke({
        "messages": [HumanMessage(content="25 加 37 等于多少？")]
    })
    print("最终回答：", result["messages"][-1].content)

    print("\n" + "=" * 50)
    print("问题2：两步工具调用（先乘后加）")
    print("=" * 50)
    result = graph.invoke({
        "messages": [HumanMessage(content="先算 6 乘以 7，再把结果加上 100，最终是多少？")]
    })
    print("最终回答：", result["messages"][-1].content)

    print("\n" + "=" * 50)
    print("问题3：不需要工具")
    print("=" * 50)
    result = graph.invoke({
        "messages": [HumanMessage(content="天空为什么是蓝色的？")]
    })
    print("最终回答：", result["messages"][-1].content)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
循环执行流程（以"先算6×7再加100"为例）：

  第1轮 agent：
    LLM 看到问题，决定先调用 multiply(6, 7)
    返回 AIMessage(tool_calls=[multiply(6,7)])

  tools_condition 判断：有 tool_calls → 去 tools

  tools（ToolNode）：
    执行 multiply(6, 7) = 42
    追加 ToolMessage(content="42")

  第2轮 agent：
    LLM 看到历史消息 + ToolMessage("42")
    决定再调用 add(42, 100)
    返回 AIMessage(tool_calls=[add(42,100)])

  tools_condition 判断：有 tool_calls → 去 tools

  tools（ToolNode）：
    执行 add(42, 100) = 142
    追加 ToolMessage(content="142")

  第3轮 agent：
    LLM 看到结果 142，不需要再调用工具了
    返回 AIMessage(content="最终结果是 142")

  tools_condition 判断：没有 tool_calls → END

ToolNode 做了什么：
  1. 读取最后一条 AIMessage 的 tool_calls
  2. 找到对应的工具函数执行
  3. 把结果包装成 ToolMessage 追加到 messages
"""
