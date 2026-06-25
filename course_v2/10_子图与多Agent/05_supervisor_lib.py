"""
【10 子图与多Agent / 05】create_supervisor —— 一行建好 Supervisor 多 Agent
=========================================================
〔10_子图与多Agent/03〕手写了 Supervisor：自己定义共享 State、写 supervisor 节点、
接路由、把每个子 Agent 当子图……能学透原理，但样板不少。

就像〔05_工具与Agent/03〕用 create_agent 一行建 ReAct Agent 一样，
langchain 生态也把"Supervisor + 多个子 Agent"打包成了一个函数。

新概念（只有这一个）：
  from langgraph_supervisor import create_supervisor

  supervisor = create_supervisor(
      agents=[agent_a, agent_b],   # 每个子 Agent 由 create_agent 建，必须带 name
      model=llm,                   # 主管自己用的模型
      prompt="你是调度主管……",     # 主管的职责说明
  ).compile()

  内部帮你做好：主管节点、把子 Agent 接成可被调用的节点、
  子 Agent 干完自动交还主管、主管判断是否结束。

安装：pip install langgraph-supervisor
  （和 SqliteSaver 一样，当前 .venv 可能尚未安装，跑本课前先装上）

对比〔10_子图与多Agent/03〕（手写）：
  手写    → 看得见每根线，适合理解原理 / 做非标准编排
  本课库  → 标准 Supervisor 模式，省样板、上手快
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ── 1. 工具（必须有 docstring，create_agent 靠它告诉 LLM 工具用途）────────
def book_flight(from_city: str, to_city: str) -> str:
    """预订航班：根据出发城市和到达城市预订机票。"""
    return f"✅ 已预订 {from_city} → {to_city} 的航班"


def book_hotel(hotel_name: str) -> str:
    """预订酒店：根据酒店名称完成预订。"""
    return f"✅ 已预订 {hotel_name}"


# ── 2. 两个子 Agent：用 create_agent 建，必须带 name ──────────────────────
flight_agent = create_agent(
    model=llm,
    tools=[book_flight],
    name="flight_agent",            # ★ name 必填，主管靠名字调度
)

hotel_agent = create_agent(
    model=llm,
    tools=[book_hotel],
    name="hotel_agent",
)


# ── 3. 一行建 Supervisor ──────────────────────────────────────────────────
supervisor = create_supervisor(
    agents=[flight_agent, hotel_agent],
    model=llm,
    prompt=(
        "你是旅行预订调度主管，协调航班助手(flight_agent)和酒店助手(hotel_agent)。\n"
        "流程：需要订机票就调 flight_agent，需要订酒店就调 hotel_agent，\n"
        "每个助手只调用一次；两件都办完后，汇总结果并结束。全程用中文。"
    ),
).compile()


# ── 4. 运行 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = supervisor.invoke({
        "messages": [HumanMessage(content="帮我订北京到上海的航班，再订如家酒店")]
    })

    print("\n── 最终对话 ──────────────────────────────────")
    for msg in result["messages"]:
        if msg.type in ("human", "ai") and msg.content.strip():
            print(f"[{msg.type}] {msg.content}")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
create_supervisor 内部大致等价于〔10_子图与多Agent/03〕手写的那套：
  - 一个 supervisor 节点：LLM 决定把任务转交给哪个子 Agent
  - 每个子 Agent 作为节点；干完后控制权交还 supervisor
  - supervisor 再判断：继续派活 or 结束

子 Agent 为什么要 name？
  主管是靠"名字"来点名调度的；没 name 没法引用，会报错。

什么时候用库、什么时候手写：
  标准的"主管 + 一组专才"结构 → create_supervisor，省事
  需要非标准路由 / 额外节点 / 自定义共享 State → 手写（〔10_子图与多Agent/03〕）

★ 三层"建 Agent"工具，由简到繁：
  create_agent（〔05_工具与Agent/03〕）        单个 ReAct Agent
  create_supervisor（本课）                    主管 + 多个子 Agent（中心调度）
  Command + Send 手写 handoff                  Agent 之间直接互相移交（去中心化）

  另一种多 Agent 协作是"去中心化 handoff"：子 Agent 用
  Command(goto=[Send(...)], graph=Command.PARENT) 直接把任务甩给同级 Agent，
  不经过主管——属于进阶玩法，理解了本课与〔04_控制流Edge/05〕的 Command 后可自行探索。
"""
