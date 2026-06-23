"""
【04 工具与Agent / 03】create_agent —— 一行代码建好 ReAct Agent
===================================================
〔04_工具与Agent/02〕手动搭了 ReAct Agent：
  1. 定义 State（MessagesState）
  2. 写 agent_node
  3. add_node("agent", agent_node)
  4. add_node("tools", ToolNode(tools))
  5. add_conditional_edges("agent", tools_condition)
  6. add_edge("tools", "agent")
  7. compile()

这七步每次都一样。langchain 把它们打包成一个函数：

新概念（只有这一个）：
  create_agent(model, tools)
    → 返回已经 compile() 好的 Graph
    → 内部结构和〔04_工具与Agent/02〕手动搭的完全相同
    → 支持传入 checkpointer、system_prompt 等可选参数

版本说明：
  旧版（已废弃）：from langgraph.prebuilt import create_react_agent
  新版（推荐）：  from langchain.agents import create_agent
  参数变化：prompt= 改名为 system_prompt=
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ── 1. 工具定义（和第七、八课一样）───────────────────────────────────────
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


# ── 2. 一行创建 ReAct Agent ───────────────────────────────────────────────
# 对比〔04_工具与Agent/02〕需要 7 步，现在只需要这一行：
graph = create_agent(llm, tools)

# 需要记忆功能时，传入 checkpointer（和〔05_持久化与记忆/01〕一样）：
# graph = create_agent(llm, tools, checkpointer=MemorySaver())

# 需要系统提示时，传入 system_prompt（注意：不是 prompt，是 system_prompt）：
# graph = create_agent(llm, tools, system_prompt="你是一个数学助手，只回答数学问题。")


# ── 3. 运行（和〔04_工具与Agent/02〕用法完全一样）──────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("问题1：需要两步工具调用")
    print("=" * 50)
    result = graph.invoke({
        "messages": [HumanMessage(content="先算 6 乘以 7，再把结果加上 100")]
    })
    print("最终回答：", result["messages"][-1].content)

    print("\n" + "=" * 50)
    print("问题2：不需要工具")
    print("=" * 50)
    result = graph.invoke({
        "messages": [HumanMessage(content="天空为什么是蓝色的？")]
    })
    print("最终回答：", result["messages"][-1].content)

    print("\n" + "=" * 50)
    print("带记忆的 Agent（传入 checkpointer）")
    print("=" * 50)
    graph_with_memory = create_agent(llm, tools, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "math-session"}}

    graph_with_memory.invoke(
        {"messages": [HumanMessage(content="我叫小明，6 乘以 7 等于多少？")]},
        config=config,
    )
    result = graph_with_memory.invoke(
        {"messages": [HumanMessage(content="我叫什么名字？")]},
        config=config,
    )
    print("最终回答：", result["messages"][-1].content)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
create_agent(model, tools) 内部等价于〔04_工具与Agent/02〕的：

  class State(MessagesState): pass

  def agent_node(state):
      return {"messages": [llm_with_tools.invoke(state["messages"])]}

  builder = StateGraph(State)
  builder.add_node("agent", agent_node)
  builder.add_node("tools", ToolNode(tools))
  builder.add_edge(START, "agent")
  builder.add_conditional_edges("agent", tools_condition)
  builder.add_edge("tools", "agent")
  return builder.compile()

新旧 API 对照：
  旧（废弃）                               新（推荐）
  from langgraph.prebuilt import           from langchain.agents import
      create_react_agent                       create_agent
  create_react_agent(llm, tools)           create_agent(llm, tools)
  create_react_agent(..., prompt="...")    create_agent(..., system_prompt="...")
  create_react_agent(...,                  create_agent(...,
      checkpointer=MemorySaver())              checkpointer=MemorySaver())

可选参数：
  checkpointer  → 传入 MemorySaver() 等，启用多轮记忆（〔05_持久化与记忆/01〕）
  system_prompt → 系统提示，字符串或 SystemMessage（〔02_状态State/01〕）

什么时候用 create_agent，什么时候手动搭：
  需要工具调用的标准 ReAct 循环 → create_agent，省事
  需要自定义路由、额外节点、非标准逻辑 → 手动搭（〔04_工具与Agent/02〕的方式）
"""
