"""
第五课：Human-in-the-Loop（人工介入）
======================================
学习要点：
  1. interrupt_before    —— compile 时指定在哪个节点前暂停
  2. 暂停后查看状态       —— graph.get_state() 查看 Agent 准备做什么
  3. 批准继续            —— graph.invoke(None, config) 恢复执行
  4. 拒绝/修改           —— graph.update_state() 修改状态后再恢复
  5. 应用场景            —— 危险操作（删除、发送、支付）执行前人工审核

核心思想：
  Agent 自主规划要调用哪个工具 → 暂停等待人工审核 → 人批准/修改/拒绝 → 继续或取消
  Checkpointer 是 Human-in-the-Loop 的前提，没有 Checkpointer 无法暂停后恢复
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
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
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── 2. 模拟"危险操作"工具 ──────────────────────────────────────────────────
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送电子邮件（不可撤回，需要人工审核）"""
    print(f"    [tool:send_email] 邮件已发送 → {to}，主题：{subject}")
    return f"邮件已成功发送给 {to}"


@tool
def delete_files(pattern: str) -> str:
    """删除匹配指定模式的文件（危险操作，需要人工审核）"""
    print(f"    [tool:delete_files] 已删除匹配 '{pattern}' 的文件")
    return f"已删除匹配 '{pattern}' 的所有文件"


@tool
def transfer_money(account: str, amount: float) -> str:
    """向指定账户转账（金融操作，需要人工审核）"""
    print(f"    [tool:transfer_money] 已向 {account} 转账 {amount} 元")
    return f"成功向账户 {account} 转账 {amount} 元"


tools = [send_email, delete_files, transfer_money]
llm_with_tools = llm.bind_tools(tools)


# ── 3. 节点 ────────────────────────────────────────────────────────────────
def agent_node(state: State) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ── 4. 构建 Graph ──────────────────────────────────────────────────────────
graph_builder = StateGraph(State)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")

memory = MemorySaver()

# ★ 关键：interrupt_before=["tools"]
# 含义：每次要执行 tools 节点之前，先暂停，等待外部指令
# 没有这行，工具会被立即执行，人类没有审核机会
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"],
)


# ── 5. 辅助函数 ────────────────────────────────────────────────────────────
def show_pending_tools(config: dict) -> list:
    """查看 Agent 准备调用哪些工具"""
    state = graph.get_state(config)
    last_msg = state.values["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return last_msg.tool_calls
    return []


# ── 6. 运行测试 ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── 场景1：人工批准 ─────────────────────────────────────────────────
    print("═" * 60)
    print("场景1：Agent 准备发邮件 → 人工审核 → 批准执行")
    print("═" * 60)

    config1 = {"configurable": {"thread_id": "scene_approve"}}

    # 第一次 invoke：Agent 规划工具调用后暂停
    graph.invoke(
        {"messages": [HumanMessage("给 boss@company.com 发一封邮件，主题是'周报'，内容是'本周工作已完成'")]},
        config=config1,
    )

    # 查看被暂停的工具调用
    pending = show_pending_tools(config1)
    print("\nAgent 准备执行以下操作，等待审核：")
    for tc in pending:
        print(f"  工具：{tc['name']}")
        print(f"  参数：{tc['args']}")

    print("\n[人工审核] ✅ 批准执行")
    # invoke(None, config) = 从暂停处继续，不传新消息
    result = graph.invoke(None, config=config1)
    print(f"最终回复：{result['messages'][-1].content}")

    # ── 场景2：人工拒绝 ─────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("场景2：Agent 准备删除文件 → 人工审核 → 拒绝执行")
    print("═" * 60)

    config2 = {"configurable": {"thread_id": "scene_reject"}}

    graph.invoke(
        {"messages": [HumanMessage("删除所有扩展名为 .log 的文件")]},
        config=config2,
    )

    pending = show_pending_tools(config2)
    print("\nAgent 准备执行以下操作，等待审核：")
    for tc in pending:
        print(f"  工具：{tc['name']}")
        print(f"  参数：{tc['args']}")

    print("\n[人工审核] ❌ 拒绝执行，注入取消消息")

    # update_state：向 messages 注入一条"取消"消息，模拟 agent 节点的输出
    # as_node="agent" 告诉 LangGraph：把这次更新视为 agent 节点的执行结果
    # 这样 tools_condition 检查新消息时发现没有 tool_calls，会路由到 END
    graph.update_state(
        config2,
        {"messages": [AIMessage(content="用户拒绝了该操作，文件删除已取消，未执行任何修改。")]},
        as_node="agent",
    )

    result = graph.invoke(None, config=config2)
    print(f"最终回复：{result['messages'][-1].content}")

    # ── 场景3：人工修改参数后批准 ───────────────────────────────────────
    print("\n" + "═" * 60)
    print("场景3：Agent 准备转账 → 人工发现金额有误 → 修改后批准")
    print("═" * 60)

    config3 = {"configurable": {"thread_id": "scene_modify"}}

    graph.invoke(
        {"messages": [HumanMessage("向账户 ACC-001 转账 999999 元")]},
        config=config3,
    )

    pending = show_pending_tools(config3)
    print("\nAgent 准备执行以下操作，等待审核：")
    for tc in pending:
        print(f"  工具：{tc['name']}")
        print(f"  参数：{tc['args']}")

    print("\n[人工审核] ✏️  发现金额有误，修改为 100 元后批准")

    # 获取当前最后一条消息（AIMessage with tool_calls）
    state = graph.get_state(config3)
    last_ai_msg = state.values["messages"][-1]

    # 构造修改后的 AIMessage（保留 tool_calls 结构，只改参数）
    modified_tool_calls = []
    for tc in last_ai_msg.tool_calls:
        modified_tc = dict(tc)
        modified_tc["args"] = {**tc["args"], "amount": 100.0}  # 修改金额
        modified_tool_calls.append(modified_tc)

    modified_msg = AIMessage(
        content=last_ai_msg.content,
        tool_calls=modified_tool_calls,
    )

    # 用修改后的消息替换原消息
    # as_node="agent" + 替换最后一条 = 覆盖 agent 的原始输出
    graph.update_state(config3, {"messages": [modified_msg]}, as_node="agent")

    result = graph.invoke(None, config=config3)
    print(f"最终回复：{result['messages'][-1].content}")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
三种 Human-in-the-Loop 处理方式对比：

┌──────────┬─────────────────────────────────────┬──────────────────┐
│ 处理方式  │ 操作                                 │ 效果             │
├──────────┼─────────────────────────────────────┼──────────────────┤
│ 批准     │ graph.invoke(None, config)           │ 工具正常执行      │
│ 拒绝     │ update_state(取消消息, as_node=agent) │ 工具跳过，直接结束│
│          │ + graph.invoke(None, config)         │                  │
│ 修改参数  │ update_state(修改后消息, as_node=agent)│ 用新参数执行工具 │
│          │ + graph.invoke(None, config)         │                  │
└──────────┴─────────────────────────────────────┴──────────────────┘

★ 5 个核心知识点：

0. hasattr(msg, "tool_calls") and msg.tool_calls 的含义
   - hasattr：先检查属性是否存在（HumanMessage 没有 tool_calls，直接访问会报错）
   - msg.tool_calls：再检查值是否非空（存在但为 [] 时也不该处理）
   - 等价更清晰的写法：isinstance(msg, AIMessage) and msg.tool_calls

1. interrupt_before 是 compile 的参数，不是节点的属性
   - interrupt_before=["tools"] → 每次即将执行 tools 节点前暂停
   - interrupt_before=["node_a", "node_b"] → 可以同时指定多个节点
   - 必须配合 Checkpointer 使用，否则无法保存暂停状态

2. invoke(None, config) = 从暂停处继续
   - 传 None 表示不新增消息，直接从暂停点继续执行
   - 传新消息则追加后继续执行

3. update_state(values, as_node="xxx") 的含义
   - 直接修改 Checkpointer 里保存的 State 快照
   - as_node="agent" 表示这次修改视为 agent 节点的输出
   - LangGraph 据此决定下一步从哪里恢复（继续走 tools_condition 的路）

4. 暂停点的 State 完整保留
   - 暂停期间可以用 get_state() 查看 Agent 的完整计划
   - 可以修改任何 State 字段，不只是 messages
"""


# ════════════════════════════════════════════════════════════════════════════
# 进阶：部分工具需要确认，部分工具直接执行
# ════════════════════════════════════════════════════════════════════════════
"""
思路：把工具拆成两组，分别放进两个 ToolNode：
  - safe_tool_node     ：安全工具，无 interrupt，直接执行
  - dangerous_tool_node：危险工具，interrupt_before，需要人工审核

自定义路由函数代替内置 tools_condition，根据 tool_calls 的工具名判断去哪个节点。

Graph 结构：
  agent
    ├─ 调用的全是安全工具  ──▶  [safe_tool_node]      ──▶  agent（循环）
    ├─ 调用的含有危险工具  ──▶  [dangerous_tool_node] ──▶  agent（循环，但先暂停）
    └─ 无 tool_calls      ──▶  END
"""

from langchain_core.messages import AIMessage as _AIMessage

# 安全工具（无需确认）
@tool
def get_weather_safe(city: str) -> str:
    """查询城市天气（安全，无需审核）"""
    data = {"北京": "晴天 25°C", "上海": "多云 22°C"}
    result = data.get(city, "暂无数据")
    print(f"    [tool:get_weather] {city} → {result}")
    return result

@tool
def calculate_safe(expression: str) -> str:
    """计算数学表达式（安全，无需审核）"""
    try:
        result = str(eval(expression, {"__builtins__": {}}, {}))
    except Exception:
        result = "计算出错"
    print(f"    [tool:calculate] {expression} = {result}")
    return result

# 危险工具（需要确认）—— 复用上面已定义的 send_email / delete_files / transfer_money

SAFE_TOOLS      = [get_weather_safe, calculate_safe]
DANGEROUS_TOOLS = [send_email, delete_files, transfer_money]
ALL_TOOLS       = SAFE_TOOLS + DANGEROUS_TOOLS
DANGEROUS_NAMES = {t.name for t in DANGEROUS_TOOLS}   # {'send_email', 'delete_files', 'transfer_money'}


def selective_router(state):
    """路由函数：根据 tool_calls 的工具名决定去哪个节点"""
    from langchain_core.messages import AIMessage
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return END
    # 只要有一个工具属于危险列表，就走 dangerous_tool_node（需要审核）
    names = {tc["name"] for tc in last.tool_calls}
    if names & DANGEROUS_NAMES:
        return "dangerous_tool_node"
    return "safe_tool_node"


# 构建选择性审核 Graph
llm_all_tools = llm.bind_tools(ALL_TOOLS)

def agent_node_v2(state):
    response = llm_all_tools.invoke(state["messages"])
    return {"messages": [response]}

graph2_builder = StateGraph(State)
graph2_builder.add_node("agent",              agent_node_v2)
graph2_builder.add_node("safe_tool_node",     ToolNode(SAFE_TOOLS))
graph2_builder.add_node("dangerous_tool_node", ToolNode(DANGEROUS_TOOLS))
graph2_builder.set_entry_point("agent")
graph2_builder.add_conditional_edges("agent", selective_router)
graph2_builder.add_edge("safe_tool_node",      "agent")
graph2_builder.add_edge("dangerous_tool_node", "agent")

memory2 = MemorySaver()
graph2 = graph2_builder.compile(
    checkpointer=memory2,
    interrupt_before=["dangerous_tool_node"],   # 只有危险工具节点前暂停
)

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("进阶：选择性审核 —— 安全工具直接执行，危险工具暂停审核")
    print("═" * 60)

    cfg = {"configurable": {"thread_id": "selective_test"}}

    # 先调用安全工具（天气查询），无需审核，直接跑完
    print("\n--- 请求1：只用安全工具 ---")
    result = graph2.invoke(
        {"messages": [HumanMessage("北京今天天气怎么样？另外 123 * 456 等于多少？")]},
        config=cfg,
    )
    print(f"回答：{result['messages'][-1].content}")

    # 再调用危险工具（发邮件），会在 dangerous_tool_node 前暂停
    cfg2 = {"configurable": {"thread_id": "selective_test2"}}
    print("\n--- 请求2：含危险工具，触发暂停 ---")
    graph2.invoke(
        {"messages": [HumanMessage("查一下北京天气，然后把结果发邮件给 test@test.com")]},
        config=cfg2,
    )
    state2 = graph2.get_state(cfg2)
    last = state2.values["messages"][-1]
    if isinstance(last, _AIMessage) and last.tool_calls:
        print("暂停！Agent 准备执行：")
        for tc in last.tool_calls:
            print(f"  {tc['name']}({tc['args']})")
        print("[人工审核] ✅ 批准")
        result2 = graph2.invoke(None, config=cfg2)
        print(f"回答：{result2['messages'][-1].content}")