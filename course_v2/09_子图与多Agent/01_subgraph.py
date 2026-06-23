"""
【09 子图与多Agent / 01】Subgraph —— 把 Graph 当节点用
=========================================
前十五课的节点都是 Python 函数。
本课引入：编译好的 Graph 也可以作为另一个 Graph 的节点。

新概念（只有这一个）：
  subgraph = builder.compile()
  outer_builder.add_node("子图名", subgraph)   ← 直接把 Graph 当节点

用途：
  把复杂逻辑拆成独立的子图，外层 Graph 像调用函数一样调用它。
  子图内部有自己的节点、边，外层 Graph 不需要关心细节。

State 通信规则：
  外层 State 和子图 State 通过"同名字段"传递数据。
  子图执行时，外层 State 里有的字段会传进去；
  子图执行完，子图 State 里有的字段会写回外层 State。
  只传递两边 State 都有的字段（交集）。
"""

import os
from typing import TypedDict

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


# ── 子图：负责"生成回答 + 统计字数"────────────────────────────────────────
class SubState(TypedDict):
    question: str   # 从外层传入
    answer: str     # 写回外层
    word_count: int # 写回外层


def llm_node(state: SubState) -> dict:
    response = llm.invoke([
        SystemMessage(content="用一句话简洁回答。"),
        HumanMessage(content=state["question"]),
    ])
    return {"answer": response.content}


def count_node(state: SubState) -> dict:
    return {"word_count": len(state["answer"])}


sub_builder = StateGraph(SubState)
sub_builder.add_node("llm_node",   llm_node)
sub_builder.add_node("count_node", count_node)
sub_builder.add_edge(START,        "llm_node")
sub_builder.add_edge("llm_node",   "count_node")
sub_builder.add_edge("count_node", END)

subgraph = sub_builder.compile()   # ← 编译好，准备当节点用


# ── 外层 Graph：负责"打印结果"────────────────────────────────────────────
class OuterState(TypedDict):
    question: str   # 传给子图
    answer: str     # 子图写回
    word_count: int # 子图写回
    label: str      # 只有外层有，子图不知道这个字段


def label_node(state: OuterState) -> dict:
    return {"label": f"【问题】{state['question']}"}


def print_node(state: OuterState) -> dict:
    print(state["label"])
    print(f"回答：{state['answer']}")
    print(f"字数：{state['word_count']}")
    return {}


outer_builder = StateGraph(OuterState)
outer_builder.add_node("label_node", label_node)
outer_builder.add_node("qa_subgraph", subgraph)   # ★ 子图当节点
outer_builder.add_node("print_node",  print_node)

outer_builder.add_edge(START,          "label_node")
outer_builder.add_edge("label_node",   "qa_subgraph")  # 调用子图
outer_builder.add_edge("qa_subgraph",  "print_node")
outer_builder.add_edge("print_node",   END)

graph = outer_builder.compile()


# ── 运行 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = graph.invoke({
        "question": "什么是深度学习？",
        "answer": "",
        "word_count": 0,
        "label": "",
    })
    print("\n最终 State：", result)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
执行流程：

  OuterState 初始：
    question="什么是深度学习？", answer="", word_count=0, label=""

  label_node 执行：
    label = "【问题】什么是深度学习？"

  qa_subgraph 执行：
    外层传入子图的字段（交集：question、answer、word_count）：
      SubState: question="什么是深度学习？", answer="", word_count=0
    子图内部跑 llm_node → count_node
    子图结束，SubState: answer="深度学习是...", word_count=18
    子图结果写回外层（交集字段）：
      OuterState.answer    = "深度学习是..."
      OuterState.word_count = 18
      OuterState.label 不变（子图不知道这个字段）

  print_node 执行：
    读 OuterState，打印结果

State 通信规则总结：
  外层 → 子图：两边 State 都有的字段自动传入
  子图 → 外层：两边 State 都有的字段自动写回
  只有外层有的字段（label）：子图看不到，也不会被子图清除
  只有子图有的字段：不会出现在外层 State 里
"""
