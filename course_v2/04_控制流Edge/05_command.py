"""
【04 控制流Edge / 05】Command —— 节点内同时"改状态 + 决定去哪"
=========================================================
〔04_控制流Edge/01〕的路由是"分工"的：
  节点只改 State；额外写一个路由函数 + add_conditional_edges 决定走向。

本课换一种风格：让节点自己在返回时同时做两件事——
  既更新 State，又指定下一个去哪个节点。这就是 Command。

新概念（只有这一个）：
  节点返回 Command(update=..., goto=...)
    update —— 要写进 State 的字段（等价于以前 return 的那个 dict）
    goto   —— 下一个要去的节点名（或 END）

  有了 Command，节点的返回类型从 dict 变成 Command；
  路由逻辑写在节点内部，不再需要单独的路由函数和条件边。

对比〔04_控制流Edge/01〕：
  条件边：节点(改状态) + 路由函数(只读、选路) + add_conditional_edges 接线
  Command：节点(改状态 + 选路 二合一)，更内聚，省掉路由函数

为便于离线运行，本课用关键词判断代替 LLM，逻辑确定、好观察。
"""

from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ── 1. State ──────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, lambda old, new: old + new]   # 简单的列表追加 reducer
    done: bool


# ── 2. 决策节点：返回 Command，自己决定 update 什么、goto 哪 ───────────────
def dispatcher(state: State) -> Command:
    # 任务已完成 → 收尾并去 END
    if state["done"]:
        return Command(update={"messages": ["[系统] 全部完成"]}, goto=END)

    last = state["messages"][-1]
    if "数学" in last:
        return Command(
            update={"messages": ["[调度] 这是数学任务，转交 math"]},
            goto="math",
        )
    if "翻译" in last:
        return Command(
            update={"messages": ["[调度] 这是翻译任务，转交 translate"]},
            goto="translate",
        )
    # 识别不了 → 标记完成并结束
    return Command(update={"messages": ["[调度] 无法识别，结束"], "done": True}, goto=END)


# ── 3. 业务节点：干完活，用 Command 把控制权交回 dispatcher ────────────────
def math(state: State) -> Command:
    return Command(
        update={"messages": ["[math] 2 + 2 = 4"], "done": True},
        goto="dispatcher",
    )


def translate(state: State) -> Command:
    return Command(
        update={"messages": ["[translate] Hello → 你好"], "done": True},
        goto="dispatcher",
    )


# ── 4. 构建：注意几乎不用写边——走向都在 Command.goto 里 ──────────────────
builder = StateGraph(State)
builder.add_node("dispatcher", dispatcher)
builder.add_node("math",       math)
builder.add_node("translate",  translate)
builder.add_edge(START, "dispatcher")   # 只需要指定入口；其余跳转靠 Command.goto
graph = builder.compile()


# ── 5. 运行 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for task in ["帮我算个数学题", "帮我翻译一句话", "随便聊聊"]:
        print(f"\n【输入】{task}")
        result = graph.invoke({"messages": [task], "done": False})
        for m in result["messages"]:
            print("  ", m)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
执行流程（以"帮我算个数学题"为例）：

  START → dispatcher
    dispatcher 返回 Command(update=[...], goto="math")  ← 改状态 + 选路
  → math
    math 返回 Command(update=[..., done=True], goto="dispatcher")
  → dispatcher
    这次 done=True → Command(update=[...], goto=END)
  → END

Command 的两个字段：
  update={...}   和以前节点 return 的 dict 一样，按 reducer 合并进 State
  goto="节点名"   下一步去哪；也可以是 END

什么时候用 Command、什么时候用条件边（〔04_控制流Edge/01〕）：
  逻辑简单、想让"图结构"一眼可见  → 条件边（路由独立、可视化清晰）
  节点本就要根据自己算出的结果选路 → Command（改状态和选路在一处，更内聚）
  多 Agent 互相转交控制权          → Command 尤其顺手（见〔10_子图与多Agent〕）

★ Command 家族（同一个类，三种典型用法）：
  Command(update=, goto=)            本课：改状态 + 普通路由
  Command(resume=...)                〔08_中断与人工干预/01〕：恢复被 interrupt 暂停的图
  Command(goto=[Send(...)], graph=Command.PARENT)
                                     〔10_子图与多Agent〕：跨图把任务移交给别的 Agent
"""
