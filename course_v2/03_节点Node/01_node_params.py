"""
【03 节点Node / 01】给节点绑定额外参数 —— functools.partial
================================================
前面所有课的节点都是"只接收 state 一个参数"的函数：def node(state): ...
但有时一个节点逻辑想复用，只是某些配置不同（阈值、前缀、模式…），
不想为每种配置都复制一份函数。

新概念（只有这一个）：
  用 functools.partial 把额外参数"预先绑定"到节点函数上，
  再把绑定后的函数 add_node 进图。

  def worker(state, factor, label): ...
  add_node("a", partial(worker, factor=2, label="A"))   # a 用 factor=2
  add_node("b", partial(worker, factor=3, label="B"))   # b 用 factor=3

  这样同一份 worker 逻辑，配置不同，复用成两个节点。

注意：state 永远是第一个参数，由 LangGraph 自动传入；
      partial 绑定的是 state 之外的"配置参数"。
为便于离线运行，本课节点是纯 Python，不调用 LLM。
"""

from functools import partial
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ── 1. State ──────────────────────────────────────────────────────────────
class State(TypedDict):
    value: int
    log: str


# ── 2. 一份"带配置参数"的通用节点 ─────────────────────────────────────────
# state 是第一个参数（LangGraph 自动传）；factor、label 是配置参数（partial 绑定）
def scale_node(state: State, factor: int, label: str) -> dict:
    new_value = state["value"] * factor
    return {
        "value": new_value,
        "log": state["log"] + f"[{label}: ×{factor} → {new_value}] ",
    }


# ── 3. 用 partial 把同一份逻辑绑成两个配置不同的节点 ──────────────────────
double = partial(scale_node, factor=2, label="double")   # ×2
triple = partial(scale_node, factor=3, label="triple")   # ×3


# ── 4. 构建 Graph ──────────────────────────────────────────────────────────
builder = StateGraph(State)
builder.add_node("double", double)   # 直接把 partial 结果当节点
builder.add_node("triple", triple)
builder.add_edge(START,    "double")
builder.add_edge("double", "triple")
builder.add_edge("triple", END)

graph = builder.compile()


# ── 5. 运行 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = graph.invoke({"value": 5, "log": ""})
    print("value：", result["value"])   # 5 ×2 ×3 = 30
    print("log：  ", result["log"])


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
为什么不用全局变量或多写几个函数？
  - 全局变量：节点共享状态，容易串味、难测试
  - 复制函数：double_node / triple_node 逻辑重复，改一处要改多处
  partial 让"同一份逻辑 + 不同配置"成为最干净的复用方式。

partial 的本质：
  partial(scale_node, factor=2, label="double")
  等价于一个新函数 f(state) = scale_node(state, factor=2, label="double")
  LangGraph 调用它时只传 state，factor / label 已经被预先填好。

★ 核心规律：
  节点签名里，state 永远排第一、由框架注入；
  其余参数要么用 partial 预绑定，要么用别的注入方式——
  比如运行时依赖（模型名、数据库连接）更适合用〔03_节点Node/02〕的 Runtime context。
"""
