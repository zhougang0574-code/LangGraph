from typing import TypedDict

import dotenv
from langgraph.graph import StateGraph, START, END

dotenv.load_dotenv()

# llm = ChatOpenAI(
#     api_key=os.getenv("API_KEY"),
#     base_url=os.getenv("BASE_URL"),
#     model=os.getenv("MODEL"),
# )

# 定义状态
class MyState(TypedDict):
    question: str
    answer: str
    word_count:int
    select_node:str

# 定义节点：调用 LLM 并把结果写入状态
# def llm_node(state: MyState):
#     response = llm.invoke(state["question"])
#     return {"answer": response.content}

def check_node(state:MyState):
    select_node = state["select_node"]
    if select_node == "node_a":
        return "a"
    else:
        return "b"

def node_a(state:MyState):
    print("这是节点AAAA")

def node_b(state:MyState):
    print("BBBBB")


# 构建 Graph
builder = StateGraph(MyState)
builder.add_node("node_a",node_a)
builder.add_node("node_b",node_b)

builder.add_conditional_edges(START,check_node,{"a":"node_a","b":"node_b"})
builder.add_edge("node_a", END)
builder.add_edge("node_b", END)

graph = builder.compile()

# 运行
result = graph.invoke({"select_node":"node_b"})