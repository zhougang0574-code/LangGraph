"""
第一课：LangGraph 核心概念
============================
学习要点：
  1. State  —— 整个 Graph 的"共享内存"，用 TypedDict 定义
  2. Node   —— 普通 Python 函数，接收 State、返回更新字典
  3. Edge   —— 节点之间的有向连接，决定执行顺序
  4. Graph  —— 把 State/Node/Edge 组装起来，compile 后运行

对比 LangChain：
  LangChain Chain  = 固定的线性管道（A→B→C）
  LangGraph Graph  = 灵活的有向图，支持分支/循环/并行
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

# ── LLM 初始化（百炼 qwen-plus）────────────────────────────────────────────
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ── 1. 定义 State ─────────────────────────────────────────────────────────
# State 是贯穿整个 Graph 的"共享变量包"
# 每个节点都能读取它、并返回需要更新的字段
class State(TypedDict):
    question: str       # 用户输入的问题
    answer: str         # LLM 的回答
    word_count: int     # 回答的字数统计


# ── 2. 定义 Node（节点） ────────────────────────────────────────────────────
# 节点 = 普通函数，签名固定：接收 State，返回 dict（只需包含要更新的字段）

def llm_node(state: State) -> dict:
    """节点1：调用 LLM 回答问题"""
    print(f"[llm_node] 收到问题：{state['question']}")
    response = llm.invoke([HumanMessage(content=state["question"])])
    return {"answer": response.content}


def count_node(state: State) -> dict:
    """节点2：统计回答字数"""
    count = len(state["answer"])
    print(f"[count_node] 回答字数：{count}")
    return {"word_count": count}


def print_node(state: State) -> dict:
    """节点3：打印最终结果"""
    print("\n" + "=" * 50)
    print(f"问题：{state['question']}")
    print(f"回答：{state['answer']}")
    print(f"字数：{state['word_count']}")
    print("=" * 50)
    return {}  # 不更新任何字段，返回空字典即可


# ── 3. 构建 Graph ──────────────────────────────────────────────────────────
# StateGraph(State) 告诉 LangGraph 用哪个 TypedDict 作为状态类型
graph_builder = StateGraph(State)

# add_node("节点名称", 节点函数)
graph_builder.add_node("llm", llm_node)
graph_builder.add_node("count", count_node)
graph_builder.add_node("print", print_node)

# add_edge("从", "到") —— 固定连线，无条件跳转
graph_builder.add_edge("llm", "count")
graph_builder.add_edge("count", "print")
graph_builder.add_edge("print", END)   # END 是内置终止符

# 设置入口节点
graph_builder.set_entry_point("llm")

# compile() 把所有配置锁定，生成可执行的 graph 对象
graph = graph_builder.compile()


# ── 4. 运行 Graph ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # invoke 的参数就是初始 State（可以只填部分字段）
    result = graph.invoke({
        "question": "用一句话解释什么是人工智能",
        "answer": "",
        "word_count": 0,
    })

    print("\n[最终 State]", result)


# ── 运行结果说明 ────────────────────────────────────────────────────────────
"""
执行流程（数据流向图）：

    初始 State: {question: "...", answer: "", word_count: 0}
         │
         ▼
    [llm_node]    → 调用 qwen-plus，把回答写入 answer 字段
         │
         ▼
    [count_node]  → 读取 answer，统计字数写入 word_count 字段
         │
         ▼
    [print_node]  → 读取所有字段打印结果，返回 {} 不修改任何字段
         │
         ▼
        END

最终 State 包含所有字段的最新值，invoke() 会把它作为返回值。

★ 3 个核心知识点：

1. State 是唯一的数据载体
   - 节点之间不直接传参，全靠读写同一个 State 字典
   - 类比：State 就像 LangChain 里的 memory，但更显式、更可控

2. 节点只返回要更新的字段
   - 返回 {"answer": "xxx"} 只更新 answer，其他字段保持不变
   - 不需要把整个 State 复制一遍再返回
   - 返回 {} 代表"我不修改任何东西"

3. END 是内置终止符
   - from langgraph.graph import END
   - 任何节点的边指向 END，Graph 就在那里停止
   - 一个 Graph 可以有多个节点都指向 END（多出口）
"""
