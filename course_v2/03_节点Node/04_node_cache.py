"""
【03 节点Node / 04（进阶）】节点缓存 —— CachePolicy + InMemoryCache
=========================================================
有些节点很贵（调大模型、跑复杂计算），而输入又经常重复。
每次都重算既慢又费钱。

新概念（只有这一个）：
  给节点配缓存：输入相同 → 直接返回上次结果，跳过节点函数体。

  两件事要一起做：
    1. compile(cache=InMemoryCache())           为整个图开一个缓存后端
    2. add_node(..., cache_policy=CachePolicy(ttl=秒))  声明这个节点要缓存、有效期多久

  缓存命中的判定：节点输入（相关 State）和上次完全一样 → 命中，直接取缓存值。

CachePolicy 常用字段：
  ttl       缓存有效期（秒），过期后重新计算
  key_func  自定义"用哪些输入算缓存键"（不传则用默认逻辑）

本课用一个 sleep(3) 的假"耗时节点"演示：第一次慢、第二次秒回、过期后又变慢。
纯 Python，不需要联网。
"""

import time
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy


# ── 1. State ──────────────────────────────────────────────────────────────
class State(TypedDict):
    x: int
    result: int


# ── 2. 一个"很贵"的节点：sleep 3 秒模拟耗时计算 ──────────────────────────
def expensive_node(state: State) -> dict:
    print("  [expensive_node] 真正在计算...（耗时 3 秒）")
    time.sleep(3)
    return {"result": state["x"] * 2}


# ── 3. 构建：节点声明 cache_policy，compile 提供 cache 后端 ───────────────
builder = StateGraph(State)
builder.add_node(
    "expensive",
    expensive_node,
    cache_policy=CachePolicy(ttl=8),   # 缓存 8 秒
)
builder.add_edge(START, "expensive")
builder.add_edge("expensive", END)

graph = builder.compile(cache=InMemoryCache())   # ★ 必须给图配缓存后端


# ── 4. 运行：同样的输入跑三次，观察"算 / 取缓存 / 过期重算"────────────────
if __name__ == "__main__":
    print("① 第一次执行（无缓存，会真算，约 3 秒）")
    t = time.time()
    print("   结果：", graph.invoke({"x": 5}), f"  耗时 {time.time()-t:.1f}s")

    print("\n② 第二次执行（命中缓存，秒回）")
    t = time.time()
    print("   结果：", graph.invoke({"x": 5}), f"  耗时 {time.time()-t:.1f}s")

    print("\n③ 等 8 秒让缓存过期...")
    time.sleep(8)
    t = time.time()
    print("   第三次执行（缓存已过期，重新算，又约 3 秒）")
    print("   结果：", graph.invoke({"x": 5}), f"  耗时 {time.time()-t:.1f}s")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
执行预期：
  ①  ~3.0s   缓存里没有 → 真算
  ②  ~0.0s   输入 {x:5} 和上次一样、未过期 → 命中缓存，跳过函数体
  ③  ~3.0s   8 秒 TTL 已过 → 缓存失效，重新算

两个必须配套的设置（少一个都不生效）：
  compile(cache=InMemoryCache())         图级：缓存存哪
  add_node(..., cache_policy=CachePolicy(ttl=8))  节点级：这个节点要缓存、存多久

换不同输入会怎样？
  invoke({"x": 6}) 与 {"x": 5} 输入不同 → 不命中，照常真算。

★ 核心规律：
  缓存命中 = 同一个节点 + 同样的输入 + 未过期。
  缓存（本课，省重复计算） 和 持久化 checkpointer（〔06_持久化与记忆/01〕，存对话历史）
  是两回事：前者是"加速重复计算"，后者是"记住多轮状态"。

  InMemoryCache 存在内存，进程退出即清空；生产可换持久化缓存后端。
"""
