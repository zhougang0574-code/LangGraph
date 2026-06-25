"""
【03 节点Node / 03（进阶）】节点重试 —— RetryPolicy
=====================================
真实节点会调外部服务（LLM、API、数据库），这些调用偶尔会抖动失败。
前面的课里，节点一旦抛异常，整个图就挂了。

新概念（只有这一个）：
  add_node(名字, 函数, retry_policy=RetryPolicy(...))
    → 节点抛异常时，按策略自动重试，不用自己写 try/except 循环

  RetryPolicy 常用字段：
    max_attempts     最大尝试次数（含第一次）
    initial_interval 首次重试前等待秒数
    backoff_factor   退避倍数（每次重试间隔 ×backoff_factor，避免重试风暴）
    jitter           是否加随机抖动（多个节点同时失败时错开重试）
    retry_on         只对哪些异常重试：异常类型 / 类型元组 / 一个 (exc)->bool 函数

默认行为（不传 retry_on）：
  对大多数异常都重试，但对 ValueError / TypeError / KeyError 等"代码 bug 类"
  异常不重试（因为重试也不会变好）。

本课用一个"前两次失败、第三次成功"的假节点来演示，纯 Python、不需要联网。
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy


# ── 1. State ──────────────────────────────────────────────────────────────
class State(TypedDict):
    result: str


# ── 2. 一个"不稳定"的节点：前两次抛异常，第三次成功 ─────────────────────
attempt = 0   # 全局计数器，仅用于演示重试次数


def flaky_node(state: State) -> dict:
    global attempt
    attempt += 1
    print(f"  第 {attempt} 次尝试调用...")
    if attempt < 3:
        raise ConnectionError(f"模拟网络抖动失败（第 {attempt} 次）")
    return {"result": f"成功！共尝试了 {attempt} 次"}


# ── 3. 构建：给节点挂上 RetryPolicy ───────────────────────────────────────
builder = StateGraph(State)
builder.add_node(
    "flaky",
    flaky_node,
    retry_policy=RetryPolicy(
        max_attempts=5,        # 最多试 5 次
        initial_interval=0.2,  # 首次重试前等 0.2 秒
        backoff_factor=2.0,    # 之后 0.4s、0.8s…逐步退避
        jitter=True,           # 加随机抖动
        retry_on=ConnectionError,  # 只对连接类异常重试
    ),
)
builder.add_edge(START, "flaky")
builder.add_edge("flaky", END)

graph = builder.compile()


# ── 4. 运行 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("▶ 开始执行（前两次会失败，自动重试）")
    result = graph.invoke({"result": ""})
    print("\n最终结果：", result["result"])


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
没有 RetryPolicy（前面所有课）：
  flaky_node 第一次抛 ConnectionError → 整个 invoke() 直接报错退出。

有 RetryPolicy：
  第1次失败 → 等 0.2s → 第2次失败 → 等 0.4s → 第3次成功 → 继续往下跑。
  只要在 max_attempts 内成功，对外就像没失败过一样。

retry_on 的三种写法：
  retry_on=ConnectionError              只对这一种异常重试
  retry_on=(ConnectionError, TimeoutError)  对元组里任一种重试
  retry_on=lambda e: "rate limit" in str(e)  自定义判断函数，返回 True 才重试

★ 核心规律：
  重试适合"瞬时故障"（网络抖动、限流、超时）；
  对 ValueError / KeyError 这类"代码写错了"的异常，重试无意义，
  默认策略本来就不重试它们——别用重试去掩盖 bug。

  退避（backoff）+ 抖动（jitter）是为了避免"一失败就疯狂重试"压垮下游服务。
"""
