"""
【04 控制流Edge / 02】入口与出口的几种写法 —— set_entry_point / 条件入口
=========================================================
前面的课设置入口/出口都用：
  builder.add_edge(START, "first")    # 入口
  builder.add_edge("last", END)       # 出口

本课把"入口/出口"的几种等价写法讲清楚，都是常见代码里会遇到的写法。
不引入新的图能力，只是同一件事的不同语法。

新概念（只有这一个）：
  set_entry_point / set_finish_point 是 add_edge(START/END, ...) 的语法糖

  builder.set_entry_point("first")    ≡  builder.add_edge(START, "first")
  builder.set_finish_point("last")    ≡  builder.add_edge("last", END)

  再回顾一种"动态入口"：条件入口点（〔04_控制流Edge/01〕已用过）
  builder.add_conditional_edges(START, 路由函数, 映射)
    → 从 START 出发就分叉，根据输入决定第一个执行哪个节点

为便于离线运行，本课节点是纯 Python，不调用 LLM。
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ── 1. State ──────────────────────────────────────────────────────────────
class State(TypedDict):
    text: str
    kind: str


# ── 2. 节点 ───────────────────────────────────────────────────────────────
def greet_node(state: State) -> dict:
    return {"kind": "greeting", "text": "你好！很高兴见到你"}


def bye_node(state: State) -> dict:
    return {"kind": "farewell", "text": "再见，祝你愉快"}


def other_node(state: State) -> dict:
    return {"kind": "other", "text": "我在听，请继续"}


# ── 3. 条件入口的路由函数：从 START 就决定走哪个节点 ──────────────────────
def route_from_start(state: State) -> str:
    t = state["text"].lower()
    if "hi" in t or "你好" in t:
        return "greet"
    if "bye" in t or "再见" in t:
        return "bye"
    return "other"


# ══════════════════════════════════════════════════════
# 写法 A：set_entry_point / set_finish_point（语法糖）
# ══════════════════════════════════════════════════════
def build_with_sugar():
    b = StateGraph(State)
    b.add_node("greet", greet_node)
    b.set_entry_point("greet")     # ≡ add_edge(START, "greet")
    b.set_finish_point("greet")    # ≡ add_edge("greet", END)
    return b.compile()


# ══════════════════════════════════════════════════════
# 写法 B：条件入口点（从 START 直接分叉）
# ══════════════════════════════════════════════════════
def build_with_conditional_entry():
    b = StateGraph(State)
    b.add_node("greet", greet_node)
    b.add_node("bye",   bye_node)
    b.add_node("other", other_node)
    # 入口就是个分叉：根据输入决定第一个执行谁
    b.add_conditional_edges(
        START, route_from_start,
        {"greet": "greet", "bye": "bye", "other": "other"},
    )
    b.add_edge("greet", END)
    b.add_edge("bye",   END)
    b.add_edge("other", END)
    return b.compile()


# ── 4. 运行 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── 写法 A：set_entry_point 语法糖 ──────────────")
    g = build_with_sugar()
    print(g.invoke({"text": "随便说点啥", "kind": ""}))
    try:
        g.get_graph().print_ascii()          # 需 pip install grandalf
    except ImportError:
        print("  （装了 grandalf 可在此打印 ASCII 结构图）")

    print("\n── 写法 B：条件入口点（START 直接分叉）─────────")
    g = build_with_conditional_entry()
    for t in ["Hi there", "bye now", "今天天气如何"]:
        print(f"输入 {t!r} → {g.invoke({'text': t, 'kind': ''})['kind']}")
    try:
        g.get_graph().print_ascii()          # 需 pip install grandalf
    except ImportError:
        print("  （装了 grandalf 可在此打印 ASCII 结构图）")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
三种入口写法的关系：

  add_edge(START, "x")              最显式，本课程默认用法
  set_entry_point("x")              语法糖，等价于上面这行
  add_conditional_edges(START, ...) 动态入口，从一开始就按输入分叉

出口同理：
  add_edge("x", END)  ≡  set_finish_point("x")

★ 核心规律：
  START / END 是两个特殊节点（图的"虚拟"起点和终点），不是你写的函数。
  入口 = 谁接在 START 后面；出口 = 谁连向 END。
  set_entry_point / set_finish_point 只是把这两条边写得更短，能力完全一样。

  读别人的代码时三种写法都会遇到，认识即可，自己写挑一种风格统一就行。
"""
