"""
第三课：ReAct Agent
===================
学习要点：
  1. Reducer（add_messages） —— 消息列表追加而非覆盖
  2. @tool 装饰器           —— 把普通函数变成 LLM 可调用的工具
  3. bind_tools()           —— 让 LLM 知道有哪些工具可用
  4. ToolNode               —— LangGraph 内置节点，自动执行工具调用
  5. tools_condition        —— 内置路由函数，判断是否还有工具需要调用
  6. 循环结构               —— agent → tools → agent → ... → END

ReAct 模式解释：
  Reason（推理）：LLM 思考下一步该做什么
  Act（行动）   ：调用工具执行
  Observe（观察）：把工具结果喂回给 LLM
  循环上述过程，直到 LLM 认为不需要再调用工具为止
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ── 1. 定义 State ─────────────────────────────────────────────────────────
# 重点：Annotated[list[BaseMessage], add_messages]
#
# 第一课的 State 字段默认行为是"覆盖"：
#   节点返回 {"answer": "新值"} → 直接替换旧值
#
# 但消息历史需要"追加"：
#   节点返回 {"messages": [新消息]} → 追加到列表末尾，不覆盖
#
# Annotated 的第二个参数 add_messages 就是告诉 LangGraph：
#   "这个字段用 add_messages 函数来合并，而不是直接覆盖"
from typing import TypedDict

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── 2. 定义工具（Tools） ───────────────────────────────────────────────────
# @tool 装饰器做了两件事：
#   1. 把函数的 docstring 作为工具描述，告诉 LLM 这个工具是干什么的
#   2. 把函数包装成 LangChain Tool 对象，可以传给 bind_tools()

@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和"""
    result = a + b
    print(f"[tool:add] {a} + {b} = {result}")
    return result


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积"""
    result = a * b
    print(f"[tool:multiply] {a} × {b} = {result}")
    return result


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气情况"""
    weather_data = {
        "北京": "晴天，气温 25°C，东南风 3 级",
        "上海": "多云，气温 22°C，东风 2 级",
        "广州": "小雨，气温 28°C，南风 4 级",
        "深圳": "晴天，气温 30°C，无风",
    }
    result = weather_data.get(city, f"暂无 {city} 的天气数据")
    print(f"[tool:get_weather] {city} → {result}")
    return result


tools = [add, multiply, get_weather]

# bind_tools()：把工具列表注册到 LLM
# LLM 收到这些工具的描述后，可以在回复中"声明"要调用哪个工具
# 注意：bind_tools 后 LLM 可能返回普通文本，也可能返回 tool_calls 列表
llm_with_tools = llm.bind_tools(tools)


# ── 3. 定义节点 ────────────────────────────────────────────────────────────

def agent_node(state: State) -> dict:
    """Agent 节点：调用 LLM，LLM 决定回答还是调用工具"""
    print(f"\n[agent_node] 消息数：{len(state['messages'])}")
    response = llm_with_tools.invoke(state["messages"])

    if response.tool_calls:
        print(f"[agent_node] LLM 决定调用工具：{[t['name'] for t in response.tool_calls]}")
    else:
        print(f"[agent_node] LLM 直接回答：{response.content[:50]}...")

    # add_messages reducer 会把这条新消息追加到列表末尾
    return {"messages": [response]}


# ToolNode 是 LangGraph 内置节点，不需要自己写
# 它会自动：
#   1. 读取最后一条消息的 tool_calls 列表
#   2. 逐个调用对应的工具函数
#   3. 把工具结果包装成 ToolMessage 追加到 messages


# ── 4. 构建 Graph ──────────────────────────────────────────────────────────
graph_builder = StateGraph(State)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))  # 内置节点，直接传工具列表

graph_builder.set_entry_point("agent")

# tools_condition 是内置路由函数，逻辑如下：
#   if 最后一条消息包含 tool_calls（LLM 要调用工具）:
#       return "tools"   → 去执行工具
#   else:
#       return END       → LLM 直接给出了最终答案，结束
graph_builder.add_conditional_edges(
    "agent",
    tools_condition,
)

# 工具执行完后，把结果返回给 agent，形成循环
# 这就是 ReAct 的 Observe → Reason 部分
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()


# ── 5. 运行测试 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        "25 加 37 等于多少？",
        "北京和上海今天天气怎么样？",
        "先算 6 乘以 7，再把结果加上 100，最终是多少？",
    ]

    for question in test_cases:
        print(f"\n{'═' * 55}")
        print(f"问题：{question}")
        print('─' * 55)

        result = graph.invoke({"messages": [HumanMessage(content=question)]})

        print(f"\n最终回答：{result['messages'][-1].content}")


# ── 运行结果说明 ────────────────────────────────────────────────────────────
"""
执行流程（循环结构）：

    初始 State: {messages: [HumanMessage("先算 6×7 再加 100")]}
         │
         ▼
    [agent_node]          → LLM 决定先调用 multiply(6,7)
         │                  返回 AIMessage(tool_calls=[multiply])
         │
         │  tools_condition 判断：有 tool_calls → 去 tools
         ▼
    [tools(ToolNode)]     → 执行 multiply(6,7) = 42
         │                  追加 ToolMessage(content="42")
         │
         ▼  add_edge("tools", "agent") 循环回来
    [agent_node]          → LLM 看到 42，决定再调用 add(42, 100)
         │                  返回 AIMessage(tool_calls=[add])
         │
         ▼
    [tools(ToolNode)]     → 执行 add(42, 100) = 142
         │                  追加 ToolMessage(content="142")
         │
         ▼
    [agent_node]          → LLM 看到 142，不需要再调用工具
         │                  返回 AIMessage(content="结果是 142")
         │
         │  tools_condition 判断：没有 tool_calls → END
         ▼
        END

messages 列表最终包含全部对话历史：
  HumanMessage → AIMessage(tool_calls) → ToolMessage → AIMessage(tool_calls) → ToolMessage → AIMessage(最终答案)

★ 4 个核心知识点：

1. add_messages Reducer
   - 普通字段：新值覆盖旧值
   - add_messages：新消息追加到列表末尾
   - 这样 messages 就保存了完整的对话+工具调用历史

2. bind_tools() 告诉 LLM 有工具可用
   - LLM 的回复可能是普通文本（content 有值）
   - 也可能是工具调用声明（tool_calls 有值，content 为空）

3. ToolNode 自动执行工具
   - 读取最后一条 AIMessage 的 tool_calls
   - 找到对应函数执行，结果包装成 ToolMessage
   - 不需要手写这段逻辑

4. tools_condition 决定循环还是结束
   - 有 tool_calls → 继续循环调工具
   - 没有 tool_calls → 结束，返回最终答案
"""