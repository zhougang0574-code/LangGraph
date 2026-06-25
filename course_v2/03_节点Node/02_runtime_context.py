"""
【03 节点Node / 02】运行时上下文 —— Runtime / context_schema 注入依赖
=========================================================
〔03_节点Node/01〕用 partial 绑定的是"建图时就固定"的配置。
但有些东西要到"每次 invoke 时"才知道，而且它们不属于业务数据（不该塞进 State）：
  模型名、数据库连接、API key、当前用户身份……

这些叫"运行时依赖"。把它们塞进 State 会污染业务数据；用 partial 又没法每次调用都换。

新概念（只有这一个）：
  context_schema —— 声明一份"运行时上下文"的结构（和 State 平行、独立）

  1. StateGraph(State, context_schema=ContextSchema)   建图时声明
  2. def node(state, runtime: Runtime[ContextSchema]):  节点多接一个 runtime 参数
         runtime.context.model_name                      从 runtime.context 取依赖
  3. graph.invoke(input, context=ContextSchema(...))     每次调用时传入

State vs Context：
  State   → 业务数据，会被节点读写、会进 checkpointer、会随图流转
  Context → 运行时依赖，只读、不进 State、不随图流转，调用时注入

为便于离线运行，本课节点不真正调用 LLM，只演示如何拿到注入的依赖。
"""

from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime


# ── 1. State：业务数据 ─────────────────────────────────────────────────────
class State(TypedDict):
    question: str
    answer: str


# ── 2. Context：运行时依赖（用 dataclass 声明）────────────────────────────
@dataclass
class ContextSchema:
    model_name: str
    db_connection: str
    api_key: str


# ── 3. 节点：签名多接一个 runtime: Runtime[ContextSchema] ─────────────────
def answer_node(state: State, runtime: Runtime[ContextSchema]) -> dict:
    # 从 runtime.context 取依赖（不是从 state 取）
    model = runtime.context.model_name
    db = runtime.context.db_connection
    key = runtime.context.api_key

    print(f"  本次使用模型：{model}")
    print(f"  数据库连接：  {db}")
    print(f"  API key 前缀：{key[:5]}***")

    # 真实场景里会用这些依赖去调模型 / 查库，这里只拼一段示意结果
    answer = f"[由 {model} 处理] 针对「{state['question']}」的回答"
    return {"answer": answer}


# ── 4. 构建：声明 context_schema ──────────────────────────────────────────
builder = StateGraph(State, context_schema=ContextSchema)
builder.add_node("answer", answer_node)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)

graph = builder.compile()


# ── 5. 运行：invoke 时通过 context= 传入依赖 ───────────────────────────────
if __name__ == "__main__":
    print("── 第1次调用：用 qwen-plus + 生产库 ──────────")
    result = graph.invoke(
        {"question": "什么是依赖注入？"},
        context=ContextSchema(
            model_name="qwen-plus",
            db_connection="postgresql://prod-db:5432/app",
            api_key="sk-prod-abcdefg",
        ),
    )
    print("answer：", result["answer"])

    print("\n── 第2次调用：同一个图，换一套依赖 ───────────")
    result = graph.invoke(
        {"question": "什么是依赖注入？"},
        context=ContextSchema(
            model_name="qwen-max",
            db_connection="sqlite:///dev.db",
            api_key="sk-dev-1234567",
        ),
    )
    print("answer：", result["answer"])


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
为什么不直接把 model_name 塞进 State？
  - State 是业务数据，会进 checkpointer 持久化、会随子图流转——
    把基础设施配置塞进去，会污染数据、增加耦合。
  - Context 是"这次运行的环境"，只读、不持久化，职责清晰。

三步速记：
  建图：StateGraph(State, context_schema=ContextSchema)
  节点：def node(state, runtime: Runtime[ContextSchema]) → runtime.context.xxx
  调用：graph.invoke(input, context=ContextSchema(...))

★ 核心规律：
  节点签名里在 state 之后再写一个 runtime: Runtime[T]，
  LangGraph 会自动把 invoke 时传入的 context 注入进来。

  partial（〔03_节点Node/01〕）→ 建图时固定的配置
  Runtime context（本课）        → 每次调用注入、不进 State 的运行时依赖
"""
