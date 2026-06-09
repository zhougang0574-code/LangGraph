"""
第二课：条件路由（Conditional Routing）
=======================================
学习要点：
  1. add_conditional_edges  —— 根据状态动态决定走哪条路
  2. 路由函数（Router）      —— 接收 State，返回下一个节点的名称
  3. 多出口设计             —— 一个节点可以根据条件跳转到不同节点
  4. 与第一课对比           —— add_edge（固定）vs add_conditional_edges（动态）

场景说明：
  用户输入一个问题 → LLM 先判断问题类型（意图分类）→ 根据类型路由到不同的处理节点
    - 问候类  → greeting_handler（轻松友好的回复）
    - 技术类  → technical_handler（专业详细的回复）
    - 其他类  → general_handler（通用回复）
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ── 1. 定义 State ─────────────────────────────────────────────────────────
class State(TypedDict):
    question: str    # 用户问题
    intent: str      # 意图分类结果：greeting / technical / general
    answer: str      # 最终回答


# ── 2. 定义节点 ────────────────────────────────────────────────────────────

def preprocess_node(state: State) -> dict:
    """节点0：预处理——把问题转成小写、去掉首尾空格
    这个节点存在的唯一目的：让 set_entry_point 和 add_conditional_edges
    指向不同节点，帮助理解两者的区别。
    """
    cleaned = state["question"].strip()
    print(f"[preprocess_node] 预处理完成：'{cleaned}'")
    return {"question": cleaned}


def classify_node(state: State) -> dict:
    """节点1：意图分类，决定后续走哪条路"""
    print(f"[classify_node] 问题：{state['question']}")
    response = llm.invoke([
        SystemMessage(content=(
            "你是一个意图分类器。根据用户的问题，只返回以下三个词之一，不要有任何其他内容：\n"
            "- greeting   （问候、打招呼、闲聊）\n"
            "- technical  （技术、编程、科学、专业知识）\n"
            "- general    （其他所有问题）"
        )),
        HumanMessage(content=state["question"]),
    ])
    intent = response.content.strip().lower()
    # 防御性处理：如果 LLM 没有按格式返回，默认 general
    if intent not in ("greeting", "technical", "general"):
        intent = "general"
    print(f"[classify_node] 识别意图：{intent}")
    return {"intent": intent}


def greeting_handler(state: State) -> dict:
    """节点2a：处理问候类问题"""
    print("[greeting_handler] 处理问候...")
    response = llm.invoke([
        SystemMessage(content="你是一个友好的助手，用轻松愉快的语气回复，回复控制在2句话以内。"),
        HumanMessage(content=state["question"]),
    ])
    return {"answer": response.content}


def technical_handler(state: State) -> dict:
    """节点2b：处理技术类问题"""
    print("[technical_handler] 处理技术问题...")
    response = llm.invoke([
        SystemMessage(content="你是一个技术专家，用专业、准确、结构化的方式回答技术问题。"),
        HumanMessage(content=state["question"]),
    ])
    return {"answer": response.content}


def general_handler(state: State) -> dict:
    """节点2c：处理通用问题"""
    print("[general_handler] 处理通用问题...")
    response = llm.invoke([
        SystemMessage(content="你是一个全能助手，简洁清晰地回答用户问题。"),
        HumanMessage(content=state["question"]),
    ])
    return {"answer": response.content}


def print_result(state: State) -> dict:
    """节点3：打印最终结果"""
    print("\n" + "=" * 50)
    print(f"问题：{state['question']}")
    print(f"意图：{state['intent']}")
    print(f"回答：{state['answer']}")
    print("=" * 50)
    return {}


# ── 3. 路由函数（Router） ──────────────────────────────────────────────────
# 路由函数的规则：
#   - 接收 State 作为参数
#   - 返回值是下一个节点的"名称字符串"
#   - 返回 END 代表直接结束
def route_by_intent(state: State) -> str:
    intent = state["intent"]
    if intent == "greeting":
        return "greeting_handler"
    elif intent == "technical":
        return "technical_handler"
    else:
        return "general_handler"


# ── 4. 构建 Graph ──────────────────────────────────────────────────────────
graph_builder = StateGraph(State)

graph_builder.add_node("preprocess", preprocess_node)   # 新增预处理节点
graph_builder.add_node("classify", classify_node)
graph_builder.add_node("greeting_handler", greeting_handler)
graph_builder.add_node("technical_handler", technical_handler)
graph_builder.add_node("general_handler", general_handler)
graph_builder.add_node("print_result", print_result)

# set_entry_point：整个 Graph 的入口，invoke() 时第一个执行的节点
# → 现在入口是 preprocess，不是 classify
graph_builder.set_entry_point("preprocess")

# preprocess 执行完后，固定跳到 classify
graph_builder.add_edge("preprocess", "classify")

# add_conditional_edges 第1个参数：classify 执行完后从这里出发做条件判断
# → 这里的 "classify" 与 set_entry_point 毫无关系，只是说明"出边从哪个节点发出"
graph_builder.add_conditional_edges(
    "classify",          # 从 classify 节点出发（与入口无关）
    route_by_intent,     # 路由函数：读取 intent，返回下一节点名
    {
        "greeting_handler":  "greeting_handler",
        "technical_handler": "technical_handler",
        "general_handler":   "general_handler",
    }
)

# 三条处理路径汇聚到同一个 print_result 节点
graph_builder.add_edge("greeting_handler",  "print_result")
graph_builder.add_edge("technical_handler", "print_result")
graph_builder.add_edge("general_handler",   "print_result")
graph_builder.add_edge("print_result", END)

graph = graph_builder.compile()


# ── 5. 运行测试 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_questions = [
        "你好，今天天气怎么样？",
        "什么是递归函数？请举例说明",
        "世界上最高的山是哪座？",
    ]

    for q in test_questions:
        print(f"\n{'─' * 50}")
        graph.invoke({"question": q, "intent": "", "answer": ""})


# ── 运行结果说明 ────────────────────────────────────────────────────────────
"""
执行流程（带条件分支的数据流向图）：

    初始 State: {question: "...", intent: "", answer: ""}
         │
         ▼
    [preprocess]        → 清理问题文本（set_entry_point 指向这里）
         │  ← add_edge 固定边
         ▼
    [classify]          → LLM 判断意图，写入 intent 字段
                          （add_conditional_edges 从这里出发）
         │
         ├─ intent=="greeting"  ──▶  [greeting_handler]  ──┐
         │                                                   │
         ├─ intent=="technical" ──▶  [technical_handler] ──┤
         │                                                   │
         └─ intent=="general"   ──▶  [general_handler]   ──┘
                                                            │
                                                     [print_result]
                                                            │
                                                           END

★ 3 个核心知识点：

1. add_conditional_edges vs add_edge
   - add_edge：固定路径，永远从 A 到 B
   - add_conditional_edges：动态路径，根据路由函数的返回值决定去哪

2. 路由函数只做"决策"，不做"处理"
   - 路由函数只读取 State，返回节点名称字符串
   - 真正的业务逻辑放在各个 handler 节点里
   - 关注点分离：分类 vs 处理

3. 多路汇聚（Fan-in）
   - 三个不同的 handler 最终都 add_edge 到同一个 print_result
   - 这是 Graph 中常见的"扇出-扇入"模式
   - 不同路径处理完后，统一汇聚到后续节点继续执行
"""