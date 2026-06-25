"""
【05 工具与Agent / 01】工具定义 —— 告诉 LLM 有哪些函数可以调用
================================================
与〔02_状态State/02〕的区别：
  〔02_状态State/02〕  LLM 只能回答文字
  〔05_工具与Agent/01〕  LLM 知道有工具可用，可以"决定"调用工具

新概念（只有这一个）：
  @tool 装饰器  —— 把普通函数变成 LLM 可以调用的工具
  bind_tools()  —— 把工具注册给 LLM，让它知道有哪些工具

本课只观察 LLM 的"决策"：
  - 如果问题需要工具：response.tool_calls 有内容（LLM 要调用工具）
  - 如果问题不需要工具：response.content 有内容（LLM 直接回答）
  工具的实际执行留到下一课。
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ── 1. 定义工具：@tool 装饰器做了两件事 ───────────────────────────────────
# 1. 把函数的 docstring 作为工具描述，告诉 LLM 这个工具是干什么的
# 2. 把函数包装成 LangChain Tool 对象，可以传给 bind_tools()
@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积"""
    return a * b


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    data = {"北京": "晴天 25°C", "上海": "多云 22°C", "广州": "小雨 28°C"}
    return data.get(city, f"暂无 {city} 的天气数据")


tools = [add, multiply, get_weather]

# ── 2. bind_tools()：把工具列表注册给 LLM ────────────────────────────────
# LLM 收到这些工具的描述后，回复时可以：
#   - 直接回答（response.content 有值）
#   - 声明要调用工具（response.tool_calls 有值）
llm_with_tools = llm.bind_tools(tools)


# ── 3. State：和〔02_状态State/02〕一样 ────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── 4. 节点 ────────────────────────────────────────────────────────────────
def llm_node(state: State) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def print_node(state: State) -> dict:
    last = state["messages"][-1]
    if last.tool_calls:
        print("LLM 决定调用工具：")
        for tc in last.tool_calls:
            print(f"  工具名：{tc['name']}")
            print(f"  参数：  {tc['args']}")
    else:
        print("LLM 直接回答：")
        print(f"  {last.content}")
    return {}


# ── 5. Graph：和〔02_状态State/02〕结构完全一样 ───────────────────────────────────────
builder = StateGraph(State)
builder.add_node("llm_node",   llm_node)
builder.add_node("print_node", print_node)
builder.add_edge(START,        "llm_node")
builder.add_edge("llm_node",   "print_node")
builder.add_edge("print_node", END)

graph = builder.compile()


# ── 6. 运行：用两种问题观察 LLM 的不同反应 ───────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("问题1：需要工具（数学计算）")
    print("=" * 50)
    graph.invoke({"messages": [HumanMessage(content="25 加 37 等于多少？")]})

    print("\n" + "=" * 50)
    print("问题2：不需要工具（常识问题）")
    print("=" * 50)
    graph.invoke({"messages": [HumanMessage(content="天空为什么是蓝色的？")]})

    print("\n" + "=" * 50)
    print("问题3：需要工具（天气查询）")
    print("=" * 50)
    graph.invoke({"messages": [HumanMessage(content="北京今天天气怎么样？")]})


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
LLM 回复的两种形态：

  问题需要工具时：
    response.content   = ""（空）
    response.tool_calls = [{"name": "add", "args": {"a": 25, "b": 37}, ...}]

  问题不需要工具时：
    response.content   = "天空是蓝色的原因是..."
    response.tool_calls = []（空列表）

@tool 的作用：
  - 读取函数的 docstring → 生成工具描述（LLM 靠这个描述判断要不要用）
  - 读取函数的参数类型 → 生成参数 schema（LLM 靠这个填写参数）
  - 所以 docstring 和参数类型注解要写准确

本课只观察 tool_calls，工具还没有被实际执行。
下一课引入 ToolNode，自动执行 tool_calls 里声明的工具。
"""
