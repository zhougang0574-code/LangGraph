"""
【06 持久化与记忆 / 02】SqliteSaver —— 持久化存储
=====================================
〔06_持久化与记忆/01〕用 MemorySaver 实现了多轮记忆，但数据存在内存里，程序重启就丢了。

新概念（只有这一个）：
  SqliteSaver —— 把 checkpointer 数据写到 SQLite 文件
    from langgraph.checkpoint.sqlite import SqliteSaver
    with SqliteSaver.from_conn_string("checkpoints.db") as memory:
        graph = builder.compile(checkpointer=memory)

  对比 MemorySaver：
    MemorySaver()                          → 内存，重启即丢
    SqliteSaver.from_conn_string("x.db")  → 磁盘文件，重启后仍在

用法上和 MemorySaver 完全一样（drop-in 替换），只换一行 import + 创建方式。

本课示例：
  模拟两次"启动程序"，第二次用同一个 thread_id 继续对话，
  证明状态跨进程保存下来了。
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver  # ★ 新增
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)

DB_PATH = "checkpoints.db"  # 数据库文件路径，运行后会在当前目录生成


# ── State 和节点（和〔06_持久化与记忆/01〕一样）──────────────────────────
class State(TypedDict):
    messages: list


def chat_node(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content="你是一个助手，记住用户说的所有信息。"),
        *state["messages"],
    ])
    return {"messages": state["messages"] + [response]}


builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)


# ══════════════════════════════════════════════════════
# 模拟第一次启动程序
# ══════════════════════════════════════════════════════
print("=" * 50)
print("▶ 第一次启动（写入数据库）")
print("=" * 50)

with SqliteSaver.from_conn_string(DB_PATH) as memory:
    graph = builder.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "session-001"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="你好，我叫小明，我喜欢打篮球。")]},
        config=config,
    )
    print("AI：", result["messages"][-1].content)
    print(f"\n✓ 状态已写入 {DB_PATH}")

# with 块结束，数据库连接关闭，模拟"程序退出"


# ══════════════════════════════════════════════════════
# 模拟第二次启动程序（重新打开数据库）
# ══════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("▶ 第二次启动（从数据库恢复状态）")
print("=" * 50)

with SqliteSaver.from_conn_string(DB_PATH) as memory:
    graph = builder.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "session-001"}}  # 同一个 thread_id

    # 直接问，不需要再介绍自己
    result = graph.invoke(
        {"messages": [HumanMessage(content="我叫什么名字？我有什么爱好？")]},
        config=config,
    )
    print("AI：", result["messages"][-1].content)
    print("\n✓ AI 记住了上一次的对话内容，状态跨「重启」保存成功")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
MemorySaver vs SqliteSaver：

  MemorySaver()
    → 数据存在 Python 进程内存
    → with 块结束 / 程序退出 → 全部丢失
    → 适合：本地调试、不需要持久化

  SqliteSaver.from_conn_string("checkpoints.db")
    → 数据写到 SQLite 文件（.db）
    → 程序重启后用同一 thread_id 依然能读到历史状态
    → 适合：本地开发需要持久化、单机部署

  用法完全相同，只换这一行：
    - memory = MemorySaver()
    + with SqliteSaver.from_conn_string("checkpoints.db") as memory:

其他持久化方案（生产环境）：
  PostgresSaver → 连接 PostgreSQL，适合多实例部署
  用法和 SqliteSaver 几乎一样，换 connection string 即可
"""
