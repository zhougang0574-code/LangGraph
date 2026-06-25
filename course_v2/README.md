# LangGraph 课程 v2 —— 分类轴 + 域内细梯度

本版把课程从「纯时间轴线性课程」重排为「**按概念域分组**（分类轴），每个域内部保留**一文件一个新概念、小步递进**（域内梯度）」，并对照视频速通资料补全了一批此前缺失的知识点。

- 旧版（原始线性 23 课）完整留底在 `../archive_v1/`，未做任何改动。
- 本目录是当前学习版本，**目录结构即学习路径**。

---

## 一、目录结构（即学习顺序）

域与域之间已按前置依赖排序，从上往下读即可；每个文件夹内部从 `01` 到末尾是小步递进。
标〔进阶〕的可第一遍跳过，不影响主线。

```
01_基础/
  01_hello_graph.py        State/Node/Edge/compile/invoke
  02_pipeline.py           多节点、纯 Python 节点、数据流动
  03_visualize.py          print_ascii / draw_mermaid 把图画出来          ★新增

02_状态State/
  01_messages.py           HumanMessage / SystemMessage
  02_messages_list.py      add_messages：追加 vs 覆盖
  03_messages_state.py     内置 MessagesState，省样板
  04_custom_reducer.py     自定义 Reducer 控制合并             〔进阶〕
  05_io_schema.py          input_schema / output_schema 对外接口  〔进阶〕★新增

03_节点Node/                                                          ★新增域
  01_node_params.py        functools.partial 给节点绑定额外参数
  02_runtime_context.py    context_schema / Runtime 运行时依赖注入
  03_retry_policy.py       RetryPolicy 节点重试                〔进阶〕
  04_node_cache.py         CachePolicy + InMemoryCache 节点缓存  〔进阶〕

04_控制流Edge/
  01_conditional.py        add_conditional_edges 条件路由
  02_entry_point.py        set_entry_point / 条件入口点           ★新增
  03_fanin.py              多路汇聚 Fan-in
  04_send_api.py           Send 并行 Fan-out / Map-Reduce      〔进阶〕
  05_command.py            Command(update+goto) 改状态+路由二合一   ★新增

05_工具与Agent/
  01_tools.py              @tool / bind_tools / tool_calls
  02_react_agent.py        ToolNode / tools_condition / 循环
  03_prebuilt_agent.py     create_agent 一行建 Agent

06_持久化与记忆/
  01_memory.py             MemorySaver / checkpointer / thread_id
  02_sqlite_saver.py       SqliteSaver 落盘持久化
  03_store.py              Store 跨 thread 共享记忆

07_状态读写与时间旅行/
  01_state_ops.py          get_state() / update_state()
  02_state_history.py      get_state_history() 历史快照·回放

08_中断与人工干预/
  01_interrupt.py          interrupt：暂停等人类
  02_interrupt_before.py   interrupt_before / interrupt_after 外部断点

09_流式输出/
  01_stream_modes.py       stream：updates / values / 多模式 / debug
  02_stream_messages.py    stream_mode="messages" 逐 token 打字机     ★新增
  03_stream_custom.py      get_stream_writer + custom 模式自定义流    ★新增
  04_astream_events.py     astream_events（token 级 + 全事件类型）

10_子图与多Agent/
  01_subgraph.py                    子图当节点（共享字段）
  02_subgraph_proxy.py             代理节点手动调用子图（父子State不同） ★新增
  03_multi_agent.py                Supervisor 多 Agent（手写子图版）
  04_multi_agent_create_agent.py   Supervisor 多 Agent（create_agent 版）
  05_supervisor_lib.py             create_supervisor 一行建主管多 Agent  ★新增
```

### 域间排序的依据（前置依赖）
- 状态(02) → 节点(03) → 控制流(04) 三者对应 LangGraph 的三大支柱（State / Node / Edge），由内到外；
- 工具与Agent(05) 依赖消息列表(02)与条件路由(04)；
- 持久化(06) 在 状态读写(07)、中断(08) 之前 —— `get_state`、`interrupt` 都需要先有 checkpointer；
- 流式(09) 与 子图/多Agent(10) 复用前面几乎所有积木，放最后。

---

## 二、本轮相对旧 v2 的优化（对照视频速通资料补全）

| 新增 / 加强 | 位置 | 说明 |
|---|---|---|
| 图可视化 | 01_基础/03 | `print_ascii` / `draw_mermaid`，调试图结构的必备技能 |
| 新增「节点Node」整域 | 03_节点Node | 补齐 State/Node/Edge 三支柱里缺的 Node：参数绑定、Runtime、重试、缓存 |
| input/output schema | 02_状态State/05 | 对外只暴露该暴露的字段 |
| 入口点写法 | 04_控制流Edge/02 | `set_entry_point` 语法糖 + 条件入口点 |
| Command(update+goto) | 04_控制流Edge/05 | 节点内「改状态 + 选路」二合一 |
| 流式补全 | 09_流式输出/01~03 | 多模式 / debug、`messages` 逐 token、`custom` 自定义流 |
| 子图代理模式 | 10/02 | 父子 State 不同名时的标准接法 |
| create_supervisor | 10/05 | 一行建 Supervisor 多 Agent |

> 旧 v2 已有的 23 课全部保留，仅因新增「节点Node」域和域内插入新文件而**整体重新编号**；
> 各 `.py` docstring 的标题（`【域 / 序号】`）和正文交叉引用（`〔域/序号〕`）已全部对齐新位置。

---

## 三、运行方式

`.env` 放在仓库根目录（`../.env`），`load_dotenv()` 会自动向上查找，无需在每个子文件夹复制。

```
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
API_KEY=sk-xxxx
MODEL=qwen-plus
```

在仓库根目录用项目 `.venv` 直接运行任意一课，例如：

```bash
python "course_v2/05_工具与Agent/02_react_agent.py"
```

### 依赖提示
部分课用到的库当前 `.venv` 可能尚未安装，已写进根目录 `requirements.txt`：
- `06_持久化与记忆/02_sqlite_saver.py` → `langgraph-checkpoint-sqlite`
- `10_子图与多Agent/05_supervisor_lib.py` → `langgraph-supervisor`
- `01_基础/03_visualize.py` 的 `print_ascii()` → `grandalf`（`draw_mermaid()` 不需要）

> 这些课不装对应库会报 ImportError，属正常；`pip install -r requirements.txt` 一并装上即可。

---

## 四、与 archive_v1 的对应

`archive_v1/` 是最早的线性 23 课留底，仅供回溯参考。本版在它基础上：**重排为概念域分组**、
**补全视频速通资料里的缺失知识点**、并把交叉引用从「第 N 课」改为「〔域/序号〕」。
一般情况下不需要再看 v1，照本目录顺序学即可。
