"""
第二十二课：自定义 Reducer —— 控制 State 字段如何合并
======================================================
前面见过两种内置 Reducer：
  add_messages   → 把新消息追加进列表（第13课）
  operator.add   → 把新列表拼接进旧列表（第18课）

它们的本质都是一个函数：(旧值, 新值) → 合并后的值

新概念（只有这一个）：
  自己写这个函数，用 Annotated 挂上去，就是自定义 Reducer

  def my_reducer(old, new):
      ...合并逻辑...
      return merged

  class State(TypedDict):
      field: Annotated[类型, my_reducer]

本课三个例子，展示 Reducer 的不同用法：
  1. 只保留最新 N 条日志（滑动窗口）
  2. 合并字典（新值覆盖旧值的对应 key）
  3. 去重列表（已有的不重复添加）
"""

import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ══════════════════════════════════════════════════════
# Reducer 1：滑动窗口 —— 只保留最新 3 条日志
# ══════════════════════════════════════════════════════
def keep_last_3(old: list, new: list) -> list:
    return (old + new)[-3:]   # 合并后只取最后3条


# ══════════════════════════════════════════════════════
# Reducer 2：合并字典 —— 新值覆盖旧值的对应 key
# ══════════════════════════════════════════════════════
def merge_dict(old: dict, new: dict) -> dict:
    return {**old, **new}   # 新 key 覆盖旧 key，旧 key 保留


# ══════════════════════════════════════════════════════
# Reducer 3：去重列表 —— 已有的不重复添加
# ══════════════════════════════════════════════════════
def unique_append(old: list, new: list) -> list:
    result = list(old)
    for item in new:
        if item not in result:
            result.append(item)
    return result


# ── State：三个字段各用一个不同的 Reducer ─────────────
class State(TypedDict):
    question: str
    logs:     Annotated[list[str], keep_last_3]   # 滑动窗口
    profile:  Annotated[dict,      merge_dict]     # 字典合并
    tags:     Annotated[list[str], unique_append]  # 去重


# ── 节点 ─────────────────────────────────────────────
def node_a(state: State) -> dict:
    return {
        "logs":    ["node_a 执行"],
        "profile": {"step": "a", "visited": "node_a"},
        "tags":    ["langgraph", "node_a"],
    }

def node_b(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content="用一句话回答。"),
        HumanMessage(content=state["question"]),
    ])
    return {
        "logs":    ["node_b 执行", f"LLM 回答：{response.content[:20]}..."],
        "profile": {"step": "b", "answer_len": len(response.content)},
        "tags":    ["langgraph", "llm", "node_b"],  # "langgraph" 已存在，不会重复
    }

def node_c(state: State) -> dict:
    return {
        "logs":    ["node_c 执行"],   # logs 超过3条后会滑窗
        "profile": {"step": "c"},
        "tags":    ["node_c"],
    }


builder = StateGraph(State)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_node("node_c", node_c)
builder.add_edge(START,    "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", "node_c")
builder.add_edge("node_c", END)
graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke({
        "question": "什么是人工智能？",
        "logs":     [],
        "profile":  {},
        "tags":     [],
    })

    print("logs（只保留最新3条）：", result["logs"])
    print("profile（字典合并）：  ", result["profile"])
    print("tags（去重）：         ", result["tags"])


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
Reducer 的本质：

  每次节点返回新值时，LangGraph 调用 Reducer：
    merged = reducer(state中的旧值, 节点返回的新值)
  然后把 merged 写回 State。

  没有 Reducer 时（普通字段）：直接用新值覆盖旧值。
  有 Reducer 时：由你决定怎么合并。

三种 Reducer 对比：

  keep_last_3：
    旧 ["a","b"]  + 新 ["c","d","e"]  → ["c","d","e"]（最新3条）

  merge_dict：
    旧 {"x":1, "y":2}  + 新 {"y":9, "z":3}  → {"x":1, "y":9, "z":3}
    （新 key 覆盖旧 key，旧有而新没有的 key 保留）

  unique_append：
    旧 ["a","b"]  + 新 ["b","c"]  → ["a","b","c"]
    （"b" 已存在，不重复添加）

已学过的内置 Reducer：
  add_messages（第13课）：把新消息追加，重复 id 的消息会更新
  operator.add（第18课）：列表拼接，等价于 old + new
  自定义（本课）：写任意 (old, new) → merged 函数
"""
