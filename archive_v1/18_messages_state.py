"""
第十三课：MessagesState —— 内置的消息状态，省掉重复代码
=========================================================
从第六课开始，每次定义 State 都要写这一段：

  from typing import Annotated
  from langchain_core.messages import BaseMessage
  from langgraph.graph.message import add_messages

  class State(TypedDict):
      messages: Annotated[list[BaseMessage], add_messages]

这已经成了固定模板，每课都重复。

新概念（只有这一个）：
  MessagesState —— LangGraph 内置的 State 类，已经帮你写好了上面那段
  from langgraph.graph import MessagesState

两种用法：
  1. 直接用 MessagesState（State 只需要 messages 字段时）
  2. 继承 MessagesState，添加自己的额外字段
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ── 对比：之前的写法 vs 现在的写法 ───────────────────────────────────────

# 之前（第六课起每次都要写）：
# from typing import Annotated
# from langchain_core.messages import BaseMessage
# from langgraph.graph.message import add_messages
# class State(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]

# 现在（直接用内置的）：
# from langgraph.graph import MessagesState
# State = MessagesState   ← 就这一行，完全等价


# ── 用法 1：直接用 MessagesState ──────────────────────────────────────────
print("=" * 50)
print("用法1：直接使用 MessagesState")
print("=" * 50)


def llm_node_simple(state: MessagesState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


builder1 = StateGraph(MessagesState)  # ← 直接传 MessagesState
builder1.add_node("llm", llm_node_simple)
builder1.add_edge(START, "llm")
builder1.add_edge("llm", END)
graph1 = builder1.compile()

result = graph1.invoke({"messages": [HumanMessage(content="用一句话解释什么是API")]})
print("回答：", result["messages"][-1].content[:80])


# ── 用法 2：继承 MessagesState，添加额外字段 ──────────────────────────────
print("\n" + "=" * 50)
print("用法2：继承 MessagesState，加入自己的字段")
print("=" * 50)


# 继承后可以添加任意额外字段
# messages 字段自动继承（已经带 add_messages reducer）
class MyState(MessagesState):
    topic: str   # 额外字段，普通覆盖语义


@tool
def get_topic_info(topic: str) -> str:
    """获取指定话题的简介"""
    data = {
        "Python": "Python 是一种简洁易读的高级编程语言，广泛用于数据科学和 Web 开发。",
        "LangGraph": "LangGraph 是构建有状态 AI 工作流的框架，支持循环、分支和持久化。",
    }
    return data.get(topic, f"暂无 {topic} 的信息")


tools = [get_topic_info]
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: MyState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def after_tools_node(state: MyState) -> dict:
    # 额外字段可以正常读写
    last = state["messages"][-1]
    return {"topic": f"已查询：{last.content[:20]}"}


builder2 = StateGraph(MyState)
builder2.add_node("agent", agent_node)
builder2.add_node("tools", ToolNode(tools))
builder2.add_node("after_tools", after_tools_node)
builder2.add_edge(START, "agent")
builder2.add_conditional_edges("agent", tools_condition)
builder2.add_edge("tools", "after_tools")
builder2.add_edge("after_tools", END)
graph2 = builder2.compile()

result2 = graph2.invoke({
    "messages": [HumanMessage(content="介绍一下 LangGraph")],
    "topic": "",
})
print("topic 字段：", result2["topic"])
print("最终回答：", result2["messages"][-1].content[:80])


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
MessagesState 的定义（LangGraph 源码里大致是这样）：

  class MessagesState(TypedDict):
      messages: Annotated[list[BaseMessage], add_messages]

继承时：
  class MyState(MessagesState):
      extra: str   ← 普通字段，覆盖语义
      count: int   ← 普通字段，覆盖语义

  messages 字段自动继承，仍然是 add_messages reducer（追加语义）
  extra、count 是普通字段（赋值覆盖）

从第六课到现在，所有用到 messages 的课都可以把 State 定义换成 MessagesState，
代码行数减少，效果完全一样。
"""
