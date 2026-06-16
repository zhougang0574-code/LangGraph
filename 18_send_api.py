"""
第十八课：Send API —— 并行执行（Fan-out / Map-Reduce）
=====================================================
前面所有课的路由函数都返回一个字符串（下一个节点名），
每次只走一条路，串行执行。

新概念（只有这一个）：
  Send(node_name, state)
    → 返回一个 Send 对象，代表"向某个节点发送一份任务"
    → 路由函数返回 Send 列表 → 多个节点实例同时并行跑
    → 这也是为什么 snapshot.next 是 tuple 而不是字符串：
      并行时 next 里同时有多个节点名

模式：Fan-out / Map-Reduce
  fan_out 节点：把一个列表拆成多份，每份 Send 给 worker
  worker 节点：并行处理每一份
  汇总：worker 的结果通过 Reducer 合并回主 State

本课示例：
  输入 3 个问题 → 并行调用 LLM 各自回答 → 汇总所有答案
"""

import operator
import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send  # ★ 新增

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ══════════════════════════════════════════════════════
# 1. State 定义
# ══════════════════════════════════════════════════════
class State(TypedDict):
    questions: list[str]                          # 输入：多个问题
    answers: Annotated[list[str], operator.add]   # 输出：汇总所有答案
    # ★ Annotated + operator.add 是 Reducer
    #   每个 worker 返回 ["答案X"]，自动 add 进列表，不会互相覆盖


class WorkerState(TypedDict):
    question: str                                 # 单个 worker 只处理一个问题
    answers: Annotated[list[str], operator.add]   # 写回主 State 的同名字段


# ══════════════════════════════════════════════════════
# 2. Fan-out 节点：把问题列表拆成多个 Send
# ══════════════════════════════════════════════════════
def fan_out(state: State) -> list[Send]:
    # 返回 Send 列表，每个 Send 对应一个并行 worker 实例
    # Send(节点名, 传给该节点的 State)
    return [
        Send("worker", {"question": q, "answers": []})
        for q in state["questions"]
    ]
    # 3 个问题 → 同时跑 3 个 worker，互不等待


# ══════════════════════════════════════════════════════
# 3. Worker 节点：处理单个问题
# ══════════════════════════════════════════════════════
def worker(state: WorkerState) -> dict:
    response = llm.invoke([
        SystemMessage(content="用一句话简洁回答。"),
        HumanMessage(content=state["question"]),
    ])
    print(f"  [worker] 问题：{state['question'][:20]}... → 完成")
    # 返回列表，Reducer 会把它 add 进主 State 的 answers
    return {"answers": [f"Q: {state['question']}\nA: {response.content}"]}


# ══════════════════════════════════════════════════════
# 4. 汇总节点：所有 worker 跑完后执行
# ══════════════════════════════════════════════════════
def summarize(state: State) -> dict:
    print("\n── 所有回答汇总 ────────────────────────────")
    for i, ans in enumerate(state["answers"], 1):
        print(f"\n{i}. {ans}")
    return {}


# ══════════════════════════════════════════════════════
# 5. 搭 Graph
# ══════════════════════════════════════════════════════
builder = StateGraph(State)
builder.add_node("worker",    worker)
builder.add_node("summarize", summarize)

# fan_out 不是普通节点，是路由函数，直接挂在 START 上
builder.add_conditional_edges(START, fan_out)   # ★ 路由函数返回 Send 列表

# 所有 worker 跑完后，汇入 summarize
builder.add_edge("worker", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()


# ══════════════════════════════════════════════════════
# 6. 运行
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    result = graph.invoke({
        "questions": [
            "什么是机器学习？",
            "什么是深度学习？",
            "什么是强化学习？",
        ],
        "answers": [],
    })
    print("\n最终 answers 数量：", len(result["answers"]))


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
执行流程：

  START
    ↓ fan_out() 返回 [Send("worker", q1), Send("worker", q2), Send("worker", q3)]
  worker(q1)  worker(q2)  worker(q3)   ← 3 个实例同时并行跑
    ↓              ↓            ↓
  每个返回 {"answers": ["Q:...\nA:..."]}
  Reducer(operator.add) 把 3 个列表合并成一个 answers 列表
    ↓
  summarize()  ← 等所有 worker 都完成后才执行
    ↓
  END

为什么 snapshot.next 是 tuple？
  普通路由：next = ("summarize",)          ← 只有一个节点
  并行路由：next = ("worker", "worker", "worker")  ← 同时有 3 个
  设计成 tuple 就是为了支持这种多节点并行的情况。

Reducer 的作用：
  3 个 worker 同时写 answers 字段，如果没有 Reducer 会互相覆盖。
  Annotated[list[str], operator.add] 告诉 LangGraph：
    不要覆盖，把每次写入的列表 add（拼接）进去。
  这和第13课 MessagesState 里的 add_messages 是同一个道理。
"""
