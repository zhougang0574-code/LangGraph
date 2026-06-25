"""
【10 子图与多Agent / 02】子图代理模式 —— 父子 State 不同时怎么接
=========================================================
〔10_子图与多Agent/01〕里，子图能直接当节点，是因为父图和子图"有同名字段"——
靠交集字段自动传入、写回。

但很多时候父图、子图是不同团队写的，字段名根本对不上：
  父图只有 user_query / final_answer
  子图只有 analysis_input / analysis_result
没有同名字段，就不能再"直接把子图当节点"了。

新概念（只有这一个）：
  代理节点（proxy node）——在父图里写一个普通节点，由它手动：
    1. 把父图 State  →  转换成子图要的输入
    2. compiled_subgraph.invoke(子图输入)  手动调用子图
    3. 把子图输出   →  映射回父图 State

  即"不把子图当节点，而是在某个父图节点里手动调用它"。

对比〔10_子图与多Agent/01〕：
  同名字段 → 子图直接 add_node 当节点，框架自动传参（最省事）
  字段不同 → 代理节点手动转换 + 手动 invoke（最灵活，能适配任意结构）
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


# ── 子图：自己的一套字段，和父图完全不同 ──────────────────────────────────
class SubState(TypedDict):
    analysis_input: str     # 子图的输入字段
    analysis_result: str    # 子图的输出字段


def analysis_node(state: SubState) -> dict:
    response = llm.invoke([
        SystemMessage(content="你是分析助手，针对给定问题给出一句话分析。"),
        HumanMessage(content=state["analysis_input"]),
    ])
    return {"analysis_result": response.content}


sub_builder = StateGraph(SubState)
sub_builder.add_node("analysis", analysis_node)
sub_builder.add_edge(START, "analysis")
sub_builder.add_edge("analysis", END)
compiled_subgraph = sub_builder.compile()   # 提前编译好，给代理节点调用


# ── 父图：另一套字段 ───────────────────────────────────────────────────────
class ParentState(TypedDict):
    user_query: str         # 父图的输入字段
    final_answer: str       # 父图的输出字段


# ── 代理节点：手动完成 父→子→父 的三步转换 ───────────────────────────────
def call_subgraph_proxy(state: ParentState) -> dict:
    # 步骤1：父图 State → 子图输入（字段名对不上，手动搭桥）
    sub_input = {"analysis_input": state["user_query"], "analysis_result": ""}

    # 步骤2：手动调用子图（不是把子图当节点，而是在这里 invoke）
    sub_output = compiled_subgraph.invoke(sub_input)

    # 步骤3：子图输出 → 父图 State（再搭一次桥）
    return {"final_answer": sub_output["analysis_result"]}


parent_builder = StateGraph(ParentState)
parent_builder.add_node("proxy", call_subgraph_proxy)   # 加的是代理节点，不是子图本身
parent_builder.add_edge(START, "proxy")
parent_builder.add_edge("proxy", END)
graph = parent_builder.compile()


# ── 运行 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = graph.invoke({
        "user_query": "请分析 StateGraph 的典型使用场景",
        "final_answer": "",
    })
    print("父图最终答案：", result["final_answer"])


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
两种子图接法对比：

  直接当节点（〔10_子图与多Agent/01〕）
    outer_builder.add_node("sub", subgraph)
    条件：父子 State 有同名字段，框架按交集自动传入/写回
    优点：零样板；缺点：必须共享字段名

  代理节点（本课）
    在父图里写普通节点 → 内部 compiled_subgraph.invoke(手动构造的输入)
    条件：父子 State 任意结构都行
    优点：完全可控，能做字段改名、裁剪、组装
    代价：转换逻辑要自己写

★ 核心规律：
  子图编译后就是个能 .invoke() 的普通对象。
  "当节点"是让框架自动接线；"代理节点"是你自己接线。
  字段对得上用前者，对不上（或想精确控制）用后者。

  这也呼应〔02_状态State/05〕：给子图定义清晰的 input/output schema，
  能让代理节点的"搭桥"更有章可循。
"""
