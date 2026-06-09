"""
第六课：多 Agent 协作（Supervisor 模式）
=========================================
学习要点：
  1. Supervisor 模式      —— 一个"调度 LLM"负责把任务分配给合适的专家 Agent
  2. 多节点循环           —— supervisor → agent → supervisor → agent → ... → END
  3. next 字段路由        —— State 增加 next 字段，记录下一步该交给谁
  4. 专家 Agent 设计      —— 每个 Agent 有自己专属的 LLM + 工具组合
  5. 对比前面的课程       —— 第三课是单 Agent 自己循环调工具，本课是多个 Agent 互相协作

场景说明：
  用户提一个复合任务 → Supervisor 分析后决定交给哪个专家 → 专家完成子任务后汇报
  → Supervisor 再决定下一步 → 直到所有子任务完成 → Supervisor 输出 FINISH

本课两个专家 Agent：
  - researcher：负责查询知识、事实核查（使用搜索类工具）
  - analyst   ：负责数学计算、数据分析（使用计算类工具）
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ── 1. State ──────────────────────────────────────────────────────────────
# 新增 next 字段：记录 Supervisor 决定的下一步执行者
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str   # "researcher" | "analyst" | "FINISH"


# ── 2. 专家工具定义 ────────────────────────────────────────────────────────

# Researcher 的工具（查询类）
@tool
def search_knowledge(topic: str) -> str:
    """搜索指定主题的背景知识和相关信息"""
    knowledge = {
        "Python":      "Python 是一种高级编程语言，以简洁易读著称，广泛用于 AI、数据科学、Web 开发。",
        "LangGraph":   "LangGraph 是 LangChain 团队开发的框架，用于构建基于图结构的 LLM 应用，支持循环、条件分支和多 Agent 协作。",
        "人工智能":     "人工智能（AI）是计算机科学的一个分支，致力于创建能模拟人类智能的系统，包括机器学习、深度学习、自然语言处理等方向。",
        "大语言模型":   "大语言模型（LLM）是基于 Transformer 架构的神经网络，通过海量文本预训练，能够理解和生成自然语言，代表有 GPT、Claude、Qwen 等。",
    }
    for key, value in knowledge.items():
        if key in topic:
            return value
    return f"未找到关于 '{topic}' 的具体资料，请尝试更精确的关键词。"


@tool
def get_latest_info(category: str) -> str:
    """获取指定类别的最新动态信息"""
    info = {
        "AI":    "2024年 AI 领域：多模态模型快速发展，Agent 框架成为热点，推理能力显著提升。",
        "编程":  "2024年编程趋势：AI 辅助编程工具普及，Rust 持续上升，Python 在 AI 领域地位稳固。",
    }
    for key, value in info.items():
        if key in category:
            return value
    return f"{category} 类别暂无最新数据。"


# Analyst 的工具（计算分析类）
@tool
def calculate(expression: str) -> str:
    """计算数学表达式，支持加减乘除和括号"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        print(f"    [tool:calculate] {expression} = {result}")
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算出错：{e}"


@tool
def get_statistics(numbers: str) -> str:
    """对一组数字（逗号分隔）计算统计信息：总和、平均值、最大值、最小值"""
    try:
        nums = [float(x.strip()) for x in numbers.split(",")]
        result = (
            f"数量: {len(nums)}, "
            f"总和: {sum(nums)}, "
            f"平均值: {sum(nums)/len(nums):.2f}, "
            f"最大值: {max(nums)}, "
            f"最小值: {min(nums)}"
        )
        print(f"    [tool:get_statistics] {result}")
        return result
    except Exception as e:
        return f"统计出错：{e}"


# ── 3. 专家 Agent 节点 ──────────────────────────────────────────────────────

researcher_tools = [search_knowledge, get_latest_info]
analyst_tools    = [calculate, get_statistics]

researcher_llm = llm.bind_tools(researcher_tools)
analyst_llm    = llm.bind_tools(analyst_tools)


def researcher_node(state: State) -> dict:
    """Researcher Agent：查询知识和信息"""
    print(f"\n  [researcher] 开始处理...")
    # 给 Researcher 专属的系统提示
    messages = [
        SystemMessage(content="你是一个知识研究专家，专门负责查询和整理信息。请使用工具获取准确信息后，给出清晰的研究结论。"),
        *state["messages"],
    ]
    response = researcher_llm.invoke(messages)
    # 如果 LLM 要调用工具，先执行工具再汇报结果
    if response.tool_calls:
        # 手动执行工具调用（简化版，不走完整的 ToolNode 流程）
        tool_map = {t.name: t for t in researcher_tools}
        tool_results = []
        for tc in response.tool_calls:
            result = tool_map[tc["name"]].invoke(tc["args"])
            tool_results.append(f"[{tc['name']}] {result}")
            print(f"    [tool:{tc['name']}] 完成")
        # 把工具结果整合后再让 LLM 给出最终研究结论
        summary_prompt = f"工具查询结果：\n" + "\n".join(tool_results) + "\n\n请基于以上信息，给出简洁的研究结论。"
        final = llm.invoke([
            SystemMessage(content="你是研究专家，请基于工具查询结果给出结论。"),
            *state["messages"],
            SystemMessage(content=summary_prompt),
        ])
        return {"messages": [final]}
    return {"messages": [response]}


def analyst_node(state: State) -> dict:
    """Analyst Agent：数学计算和数据分析"""
    print(f"\n  [analyst] 开始处理...")
    messages = [
        SystemMessage(content="你是一个数据分析专家，专门负责数学计算和数据统计。请使用工具精确计算，给出清晰的分析结果。"),
        *state["messages"],
    ]
    response = analyst_llm.invoke(messages)
    if response.tool_calls:
        tool_map = {t.name: t for t in analyst_tools}
        tool_results = []
        for tc in response.tool_calls:
            result = tool_map[tc["name"]].invoke(tc["args"])
            tool_results.append(f"[{tc['name']}] {result}")
            print(f"    [tool:{tc['name']}] 完成")
        summary_prompt = f"计算结果：\n" + "\n".join(tool_results) + "\n\n请基于以上数据，给出简洁的分析结论。"
        final = llm.invoke([
            SystemMessage(content="你是数据分析专家，请基于计算结果给出结论。"),
            *state["messages"],
            SystemMessage(content=summary_prompt),
        ])
        return {"messages": [final]}
    return {"messages": [response]}


# ── 4. Supervisor 节点 ─────────────────────────────────────────────────────
MEMBERS  = ["researcher", "analyst"]
OPTIONS  = MEMBERS + ["FINISH"]

SUPERVISOR_PROMPT = f"""你是一个任务协调者，负责把用户的复合任务分配给合适的专家完成。

可用专家：
- researcher：负责查询知识、搜索信息、事实核查
- analyst   ：负责数学计算、数据统计、数量分析

工作流程：
1. 分析当前对话历史，判断任务是否还有未完成的部分
2. 如果有未完成的任务，返回应该处理的专家名称（researcher 或 analyst）
3. 如果所有任务已完成，返回 FINISH

⚠️ 只返回以下选项之一，不要有任何其他内容：{", ".join(OPTIONS)}"""


def supervisor_node(state: State) -> dict:
    """Supervisor：分析进度，决定下一步交给谁"""
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    next_agent = response.content.strip()

    # 防御性处理：LLM 没按格式返回时，默认结束
    if next_agent not in OPTIONS:
        next_agent = "FINISH"

    print(f"\n[supervisor] → {next_agent}")
    # Supervisor 不向 messages 追加内容，只更新路由字段
    return {"next": next_agent}


# ── 5. 路由函数 ────────────────────────────────────────────────────────────
def supervisor_router(state: State) -> str:
    """读取 next 字段，返回下一个节点名称"""
    return state["next"]


# ── 6. 构建 Graph ──────────────────────────────────────────────────────────
graph_builder = StateGraph(State)

graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("researcher", researcher_node)
graph_builder.add_node("analyst",    analyst_node)

graph_builder.set_entry_point("supervisor")

# Supervisor 根据 next 字段决定去哪
graph_builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "researcher": "researcher",
        "analyst":    "analyst",
        "FINISH":     END,
    }
)

# 每个专家完成后，汇报给 Supervisor，由它决定下一步
graph_builder.add_edge("researcher", "supervisor")
graph_builder.add_edge("analyst",    "supervisor")

graph = graph_builder.compile()


# ── 7. 运行测试 ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    test_cases = [
        # 只需要 researcher
        "介绍一下 LangGraph 是什么",
        # 只需要 analyst
        "计算 (123 + 456) * 2 的结果，并对 10, 20, 30, 40, 50 做统计分析",
        # 需要两个专家协作
        "先查一下大语言模型的背景知识，然后计算 2 的 10 次方是多少",
    ]

    for question in test_cases:
        print(f"\n{'═' * 60}")
        print(f"问题：{question}")
        print('─' * 60)

        result = graph.invoke({
            "messages": [HumanMessage(content=question)],
            "next": "",
        })

        print(f"\n最终回答：{result['messages'][-1].content[:200]}")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
Supervisor 模式的 Graph 结构：

         ┌─────────────────────────────────────────┐
         │                                         │
    START ──▶  [supervisor]  ──▶  [researcher]  ──┤
                   │                               │
                   ├──▶  [analyst]    ─────────────┘
                   │
                   └──▶  END（当 next=="FINISH"）

执行流程示例（"查大语言模型 + 计算 2^10"）：

  [supervisor] 分析：有查询任务 → next="researcher"
  [researcher] 调用 search_knowledge("大语言模型") → 输出研究结论
  [supervisor] 分析：还有计算任务 → next="analyst"
  [analyst]    调用 calculate("2**10") → 输出分析结论
  [supervisor] 分析：所有任务完成 → next="FINISH"
  END

★ 4 个核心知识点：

1. Supervisor 是"决策节点"，不是"执行节点"
   - Supervisor 只做路由决策，更新 next 字段
   - 不向 messages 追加内容，避免污染对话历史
   - 每个专家执行完后都回到 supervisor，由它决定下一步

2. next 字段是 Supervisor 模式的核心
   - State 增加 next 字段专门用于路由
   - Supervisor 写入，router 函数读取
   - 与 messages 字段职责分离：messages 存内容，next 存控制流

3. 专家 Agent 各自有独立的 LLM + 工具 + 系统提示
   - researcher_llm = llm.bind_tools(researcher_tools)
   - analyst_llm    = llm.bind_tools(analyst_tools)
   - 不同专家用不同系统提示，让 LLM "扮演"不同角色

4. 循环结构：supervisor → agent → supervisor → ... → END
   - 与第三课 ReAct 的 agent → tools → agent 类似，都是循环
   - 区别：ReAct 是单 Agent 循环调工具，本课是多 Agent 轮流被调度
   - 终止条件：Supervisor 判断所有任务完成，返回 FINISH
"""