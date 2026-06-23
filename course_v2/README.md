# LangGraph 课程 v2 —— 分类轴 + 域内细梯度

本版把课程从「纯时间轴线性 23 课」重排为「**按概念域分组**（分类轴），每个域内部保留**一文件一个新概念、小步递进**（域内梯度）」。

- 旧版（原始线性课程）完整留底在 `../archive_v1/`，未做任何改动。
- 本目录是新的学习版本，目录结构即学习路径。

---

## 一、目录结构（即学习顺序）

域与域之间已按前置依赖排序，从上往下读即可；每个文件夹内部从 `01` 到末尾是小步递进。

```
01_基础/
  01_hello_graph.py        State/Node/Edge/compile/invoke
  02_pipeline.py           多节点、纯 Python 节点、数据流动

02_状态State/
  01_messages.py           HumanMessage / SystemMessage
  02_messages_list.py      add_messages：追加 vs 覆盖
  03_messages_state.py     内置 MessagesState，省样板
  04_custom_reducer.py     自定义 Reducer 控制合并        〔进阶·可后置〕

03_控制流Edge/
  01_conditional.py        add_conditional_edges 条件路由
  02_fanin.py              多路汇聚 Fan-in
  03_send_api.py           Send 并行 Fan-out / Map-Reduce  〔进阶〕

04_工具与Agent/
  01_tools.py              @tool / bind_tools / tool_calls
  02_react_agent.py        ToolNode / tools_condition / 循环
  03_prebuilt_agent.py     create_agent 一行建 Agent

05_持久化与记忆/
  01_memory.py             MemorySaver / checkpointer / thread_id
  02_sqlite_saver.py       SqliteSaver 落盘持久化
  03_store.py              Store 跨 thread 共享记忆

06_状态读写与时间旅行/
  01_state_ops.py          get_state() / update_state()
  02_state_history.py      get_state_history() 历史快照·回放

07_中断与人工干预/
  01_interrupt.py          interrupt：暂停等人类
  02_interrupt_before.py   interrupt_before / interrupt_after 外部断点

08_流式输出/
  01_streaming.py          stream 模式（节点级）
  02_astream_events.py     astream_events（token 级）

09_子图与多Agent/
  01_subgraph.py                    子图当节点
  02_multi_agent.py                 Supervisor 多 Agent（手写子图版）
  03_multi_agent_create_agent.py    Supervisor 多 Agent（create_agent 版）〔对照〕
```

### 域间排序的依据（前置依赖）
- 状态(02) 在 Agent(04) 之前 —— Agent 依赖消息列表；
- 控制流(03) 在 Agent(04) 之前 —— Agent 用到条件路由（tools_condition）；
- 持久化(05) 在 状态读写(06) / 中断(07) 之前 —— `get_state`、`interrupt` 都需要先有 checkpointer；
- 子图/多 Agent(09) 放最后 —— 它复用前面几乎所有积木。

标〔进阶〕的两项排在各自域的末尾，第一遍可跳过，不影响主线。

---

## 二、新旧编号对照表

| 学习顺序 | 新位置 | 旧文件名（在 archive_v1/） |
|----|--------|------------|
| 1 | 01_基础/01_hello_graph | 01_hello_graph.py |
| 2 | 01_基础/02_pipeline | 03_pipeline.py |
| 3 | 02_状态State/01_messages | 02_messages.py |
| 4 | 02_状态State/02_messages_list | 06_messages_list.py |
| 5 | 02_状态State/03_messages_state | 18_messages_state.py |
| 6 | 02_状态State/04_custom_reducer | 23_custom_reducer.py |
| 7 | 03_控制流Edge/01_conditional | 04_conditional.py |
| 8 | 03_控制流Edge/02_fanin | 05_fanin.py |
| 9 | 03_控制流Edge/03_send_api | 22_send_api.py |
| 10 | 04_工具与Agent/01_tools | 07_tools.py |
| 11 | 04_工具与Agent/02_react_agent | 08_react_agent.py |
| 12 | 04_工具与Agent/03_prebuilt_agent | 19_prebuilt_agent.py |
| 13 | 05_持久化与记忆/01_memory | 09_memory.py |
| 14 | 05_持久化与记忆/02_sqlite_saver | 10_sqlite_saver.py |
| 15 | 05_持久化与记忆/03_store | 11_store.py |
| 16 | 06_状态读写与时间旅行/01_state_ops | 16_state_ops.py |
| 17 | 06_状态读写与时间旅行/02_state_history | 17_state_history.py |
| 18 | 07_中断与人工干预/01_interrupt | 14_interrupt.py |
| 19 | 07_中断与人工干预/02_interrupt_before | 15_interrupt_before.py |
| 20 | 08_流式输出/01_streaming | 12_streaming.py |
| 21 | 08_流式输出/02_astream_events | 13_astream_events.py |
| 22 | 09_子图与多Agent/01_subgraph | 20_subgraph.py |
| 23 | 09_子图与多Agent/02_multi_agent | 21_multi_agent.py |

> 各文件 docstring 的**标题行**与**正文交叉引用**均已改为新方案：标题用 `【域 / 序号】`，引用用 `〔域/序号〕`（如「与〔04_工具与Agent/01〕的区别」）。

---

## 三、运行方式

`.env` 仍放在仓库根目录（`../.env`），`load_dotenv()` 会自动向上查找，无需在每个子文件夹复制。

```
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
API_KEY=sk-xxxx
MODEL=qwen-plus
```

在仓库根目录用项目 `.venv` 直接运行任意一课，例如：

```bash
python "course_v2/04_工具与Agent/02_react_agent.py"
```

---

## 四、已完成的打磨

1. ✅ **文件内交叉引用**：23 个 docstring 的标题行 + 正文 68 处「第N课」引用已全部改为新位置标签。
2. ✅ **HTML 笔记重排**：`langgraph_notes.html` 的 23 个 `<section>` 与侧边栏 nav 已按 9 个概念域重排，加了域分组标题；每节 banner 标签、导航编号、正文 79 处「第N课」引用均已对齐新方案。
3. ✅ **代码质量**：`09_子图与多Agent/02_multi_agent.py` 删除了死代码（`math_agent_node` / `math_tool_node`）与手写 ReAct 循环，数学子 Agent 改用内置 `ToolNode` + `tools_condition`（与 〔04_工具与Agent/02〕 一致）；并新增 `09_子图与多Agent/03_multi_agent_create_agent.py`（基于 `Demo1.py` 的更简洁 `create_agent` 版本）作为对照。

## 五、仍可选的后续

- `09_子图与多Agent` 现有 3 个文件（02 手写子图、03 create_agent 版），README 第一节目录树只列了 2 个，可按需补充说明。
- HTML 中 `create_react_agent` 等个别旧 API 名称与 `.py`（`create_agent`）略有出入，属内容层面，未改动。
