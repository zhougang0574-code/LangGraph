"""
【05 持久化与记忆 / 03】Store API —— 跨 thread 共享记忆
============================================
〔05_持久化与记忆/01〕的 checkpointer（MemorySaver）是每个 thread 的私有状态：
  thread-A 的历史和 thread-B 完全隔离，互看不到。

有时候我们需要跨会话共享信息，比如：
  "第一次对话记住用户叫小明，第二次对话（新 thread）还能认出他"

新概念（只有这一个）：
  Store —— 所有 thread 共享的键值存储
    from langgraph.store.memory import InMemoryStore
    store = InMemoryStore()
    graph = builder.compile(checkpointer=memory, store=store)

  节点里通过参数注入访问 store：
    def node(state, *, store: BaseStore):
        store.put(命名空间, key, value)   # 写入
        item = store.get(命名空间, key)   # 读取

对比：
  checkpointer  → 每个 thread 独有，存执行历史和 State 快照
  store         → 全局共享，跨 thread 存长期记忆（用户画像、偏好等）
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore          # 节点参数类型注解用
from langgraph.store.memory import InMemoryStore   # ★ 新增

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ── State ────────────────────────────────────────────────
class State(TypedDict):
    user_id: str    # 标识是哪个用户
    messages: list


# ══════════════════════════════════════════════════════
# 节点里通过 *, store: BaseStore 注入 store
# ══════════════════════════════════════════════════════
def chat_node(state: State, *, store: BaseStore) -> dict:
    user_id = state["user_id"]
    namespace = ("users", user_id)   # 命名空间：以元组表示层级

    # ── 读取 store 里该用户的历史记忆 ──────────────────
    memory_item = store.get(namespace, "profile")
    memory_text = ""
    if memory_item:
        profile = memory_item.value
        memory_text = f"已知用户信息：{profile}"

    # ── 调用 LLM ────────────────────────────────────────
    response = llm.invoke([
        SystemMessage(content=f"你是一个助手。{memory_text}"),
        *state["messages"],
    ])

    # ── 把对话中提到的新信息存入 store ──────────────────
    last_msg = state["messages"][-1].content
    if "我叫" in last_msg or "我喜欢" in last_msg:
        existing = memory_item.value if memory_item else {}
        if "我叫" in last_msg:
            name = last_msg.split("我叫")[1].split("，")[0].split("。")[0].strip()
            existing["name"] = name
        if "我喜欢" in last_msg:
            hobby = last_msg.split("我喜欢")[1].split("，")[0].split("。")[0].strip()
            existing["hobby"] = hobby
        store.put(namespace, "profile", existing)  # ★ 写入 store
        print(f"  [store] 已保存用户信息：{existing}")

    return {"messages": state["messages"] + [response]}


builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# ★ 同时传入 checkpointer 和 store
memory = MemorySaver()
store  = InMemoryStore()
graph  = builder.compile(checkpointer=memory, store=store)


# ══════════════════════════════════════════════════════
# 运行
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    USER_ID = "user-001"

    # ── 第一次对话（thread-A）────────────────────────────
    print("=" * 50)
    print("第一次对话（thread-A）：介绍自己")
    print("=" * 50)
    config_a = {"configurable": {"thread_id": "thread-A"}}

    result = graph.invoke(
        {"user_id": USER_ID, "messages": [HumanMessage(content="你好，我叫小明，我喜欢打篮球。")]},
        config=config_a,
    )
    print("AI：", result["messages"][-1].content)

    # ── 第二次对话（thread-B，全新 thread）──────────────
    print("\n" + "=" * 50)
    print("第二次对话（thread-B，全新 thread）：问 AI 是否记得")
    print("=" * 50)
    config_b = {"configurable": {"thread_id": "thread-B"}}  # 不同 thread_id

    result = graph.invoke(
        {"user_id": USER_ID, "messages": [HumanMessage(content="你还记得我叫什么名字吗？我有什么爱好？")]},
        config=config_b,
    )
    print("AI：", result["messages"][-1].content)
    print("\n✓ thread-B 没有 thread-A 的对话历史，但通过 store 记住了用户信息")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
checkpointer vs store 对比：

  checkpointer（MemorySaver / SqliteSaver）
    作用：保存每次执行的 State 快照（执行历史）
    隔离：每个 thread_id 独立，thread-A 和 thread-B 互相看不到
    场景：同一个会话的多轮对话

  store（InMemoryStore）
    作用：跨 thread 共享的键值存储
    共享：所有 thread 都能读写同一个 store
    场景：跨会话的长期记忆（用户画像、偏好、事实）

store 的命名空间（namespace）：
  用元组表示层级，比如 ("users", "user-001")
  类似文件夹路径，方便按用户 / 类型分类管理

store API：
  store.put(namespace, key, value)   → 写入
  store.get(namespace, key)          → 读取单条（返回 Item 对象）
  store.search(namespace)            → 搜索该命名空间下所有条目
  item.value                         → 取出存储的值
"""
