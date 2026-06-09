"""
第四课：Checkpointer 持久化记忆
================================
学习要点：
  1. MemorySaver         —— 内存级 Checkpointer，保存每一步的 State 快照
  2. compile(checkpointer) —— 编译时注入持久化能力
  3. thread_id           —— 会话标识符，区分不同对话
  4. config              —— invoke/stream 时传入，指定当前是哪个会话
  5. 多轮对话            —— 同一 thread_id 下，下一轮能看到上一轮的历史
  6. 多会话隔离          —— 不同 thread_id 之间互不影响

对比第三课：
  第三课的 Agent 是无记忆的——每次 invoke 都从头开始，不知道上次说了什么
  本课加入 Checkpointer 后——同一 thread_id 的多次 invoke 共享完整对话历史
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ── 1. State（与第三课完全相同）───────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── 2. 工具定义（与第三课相同）────────────────────────────────────────────
@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积"""
    return a * b


tools = [add, multiply]
llm_with_tools = llm.bind_tools(tools)


# ── 3. 节点定义（与第三课相同）────────────────────────────────────────────
def agent_node(state: State) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ── 4. 构建 Graph ──────────────────────────────────────────────────────────
graph_builder = StateGraph(State)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")

# ★ 关键区别：compile 时传入 checkpointer
# MemorySaver 把每一步执行后的 State 快照保存在内存中
# 下次同一 thread_id 调用时，从最新快照恢复，接着上次继续
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


# ── 5. 辅助函数：带 thread_id 的对话 ─────────────────────────────────────
def chat(thread_id: str, user_message: str) -> str:
    """向指定会话发送一条消息，返回 AI 回复"""
    # config 是 LangGraph 的运行时配置
    # thread_id 是会话标识符，相同 thread_id = 同一个对话
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )
    return result["messages"][-1].content


# ── 6. 运行测试 ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── 测试1：多轮对话记忆 ─────────────────────────────────────────────
    print("═" * 55)
    print("测试1：多轮对话（同一 thread_id = 同一会话）")
    print("═" * 55)

    print("\n第1轮：")
    reply = chat("session_A", "3 乘以 4 等于多少？")
    print(f"AI：{reply}")

    print("\n第2轮：（不重复问题，直接追问）")
    reply = chat("session_A", "把刚才的结果再加上 8 是多少？")
    print(f"AI：{reply}")

    print("\n第3轮：")
    reply = chat("session_A", "我们这次对话一共算了几道数学题？")
    print(f"AI：{reply}")

    # ── 测试2：多会话隔离 ─────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("测试2：多会话隔离（不同 thread_id = 不同会话）")
    print("═" * 55)

    print("\n会话A 第1轮：")
    reply = chat("user_alice", "5 加 6 等于多少？")
    print(f"AI(Alice)：{reply}")

    print("\n会话B 第1轮（全新会话，不知道 Alice 问了什么）：")
    reply = chat("user_bob", "刚才那道题的答案是多少？")
    print(f"AI(Bob)：{reply}")

    # ── 查看某个会话的完整历史 ────────────────────────────────────────
    print("\n" + "═" * 55)
    print("查看 session_A 的完整消息历史：")
    print("═" * 55)
    config = {"configurable": {"thread_id": "session_A"}}
    state_snapshot = graph.get_state(config)
    for i, msg in enumerate(state_snapshot.values["messages"]):
        role = msg.__class__.__name__.replace("Message", "")
        print(f"  [{i+1}] {role}: {msg.content[:60]}{'...' if len(msg.content) > 60 else ''}")


# ── 运行结果说明 ────────────────────────────────────────────────────────────
"""
执行机制：

  第1次 invoke(session_A, "3×4=?")
    ├── Checkpointer 查找 session_A 的历史 → 空（第一次）
    ├── 正常执行 Graph
    └── Checkpointer 保存执行后的 State 快照到 session_A

  第2次 invoke(session_A, "把刚才的结果再加8")
    ├── Checkpointer 查找 session_A 的历史 → 找到上次的 messages
    ├── 把本次新消息追加到历史 messages 后面
    ├── LLM 看到完整历史，知道"刚才的结果"是 12
    └── Checkpointer 再次保存最新 State 快照

  invoke(user_bob, "刚才那道题的答案是多少？")
    ├── Checkpointer 查找 user_bob 的历史 → 空（全新会话）
    ├── LLM 没有上下文，不知道"刚才那道题"是什么
    └── AI 回复"不知道你指的是哪道题"

★ 4 个核心知识点：

1. MemorySaver 保存的是 State 快照，不只是消息
   - 每个节点执行后都会触发一次快照保存
   - 可以通过 graph.get_state(config) 查看当前快照

2. thread_id 是会话的唯一标识
   - 相同 thread_id → 同一对话，共享历史
   - 不同 thread_id → 独立对话，互不干扰
   - 可以是任意字符串：user_id、session_uuid 等

3. config 是运行时参数，每次 invoke 都要传
   - config = {"configurable": {"thread_id": "xxx"}}
   - 不传 config 或不传 thread_id，Checkpointer 不生效

4. MemorySaver 是进程内存储，重启后丢失
   - 生产环境用 SqliteSaver 或 PostgresSaver 替代
   - 接口完全相同，只需替换 compile 时的 checkpointer 参数
"""