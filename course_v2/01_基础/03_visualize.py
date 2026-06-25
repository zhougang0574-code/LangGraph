"""
【01 基础 / 03】图可视化 —— 把 Graph 画出来看结构
=====================================
与〔01_基础/02〕的区别：不引入任何新的 Graph 能力，只学会"把已经搭好的图画出来"。
这是个贯穿全程的调试技能：图一复杂，光看代码很难想象结构，画出来一目了然。

新概念（只有这一个）：
  compile() 后的 graph 有 .get_graph()，它再提供三种画法：

  graph.get_graph().print_ascii()       → 在终端打印 ASCII 结构图（需 pip install grandalf）
  graph.get_graph().draw_mermaid()      → 输出 Mermaid 文本，可贴到在线编辑器看美图
  graph.get_graph().draw_mermaid_png()  → 渲染成 PNG 字节（需要联网，时好时坏）

本课用一个"带分叉"的小图来演示，分叉结构画出来最能体现可视化的价值。
为了能离线直接跑，本课节点都是纯 Python，不调用 LLM。
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ── 1. State ──────────────────────────────────────────────────────────────
class State(TypedDict):
    x: int
    label: str


# ── 2. 节点（纯 Python，便于离线运行）─────────────────────────────────────
def check_node(state: State) -> dict:
    return {}   # 只用来当分叉点，不改 State


def even_node(state: State) -> dict:
    return {"label": f"{state['x']} 是偶数"}


def odd_node(state: State) -> dict:
    return {"label": f"{state['x']} 是奇数"}


# ── 3. 路由函数（和〔04_控制流Edge/01〕同一套机制）─────────────────────────
def route(state: State) -> str:
    return "even" if state["x"] % 2 == 0 else "odd"


# ── 4. 构建一个带分叉的 Graph ─────────────────────────────────────────────
builder = StateGraph(State)
builder.add_node("check", check_node)
builder.add_node("even",  even_node)
builder.add_node("odd",   odd_node)
builder.add_edge(START, "check")
builder.add_conditional_edges("check", route, {"even": "even", "odd": "odd"})
builder.add_edge("even", END)
builder.add_edge("odd",  END)

graph = builder.compile()


# ── 5. 运行 + 三种可视化 ───────────────────────────────────────────────────
if __name__ == "__main__":
    result = graph.invoke({"x": 4, "label": ""})
    print("运行结果：", result)

    # ── 画法 1：ASCII（最常用，终端直接看）────────────────────────────────
    # print_ascii 依赖第三方库 grandalf：pip install grandalf
    print("\n── print_ascii() ─────────────────────────────")
    try:
        graph.get_graph().print_ascii()
    except ImportError:
        print("  （需要先 pip install grandalf 才能画 ASCII 图）")

    # ── 画法 2：Mermaid 文本 ──────────────────────────────────────────────
    # 把下面这段贴到 https://mermaid.live 或 ProcessOn 的 Mermaid 编辑器，能看到美图
    print("\n── draw_mermaid() ────────────────────────────")
    print(graph.get_graph().draw_mermaid())

    # ── 画法 3：渲染 PNG（需要联网，失败属正常，已注释）──────────────────
    # png_bytes = graph.get_graph().draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(png_bytes)
    # print("已生成 graph.png")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
三种画法对比：

  print_ascii()
    优点：零依赖、终端直接看、调试最顺手
    缺点：复杂图会比较挤

  draw_mermaid()
    输出一段 Mermaid 文本（graph TD; ...），贴到在线编辑器看彩色美图
    适合：放进文档 / README / 汇报

  draw_mermaid_png()
    本地渲染成图片字节，但默认走 mermaid.ink 在线服务，常因网络失败
    报错 "Failed to reach https://mermaid.ink" 是常态，不是你的代码问题

★ 核心规律：
  可视化操作的对象是 compile() 之后的 graph，不是 builder。
  必须先 graph.get_graph()，再调 print_ascii() / draw_mermaid()。

  画出来的是"结构"（节点 + 边），与具体某次运行的数据无关——
  不 invoke 也能画，搭好图随时可以画出来检查接线对不对。
"""
