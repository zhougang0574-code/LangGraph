# 学习项目交付规范

适用于：LangGraph、LangChain 等结构化学习项目

---

## 一、每课交付顺序（必须按序执行）

### 第一步：写代码文件

文件命名：`01_xxx.py`、`02_xxx.py` 依次递增

代码文件结构要求：
1. 文件顶部用 `"""docstring"""` 写清楚本课学习要点（中文）
2. 代码关键行写注释，说明 **为什么** 这样写，不只说做了什么
3. 文件末尾用多行字符串写：
   - 执行流程图（ASCII 或文字版）
   - 本课核心知识点总结（★ 标注）

### 第二步：更新 HTML 笔记

- **单一文件**：所有课程汇总在一个 HTML 文件里（如 `langgraph_notes.html`）
- **追加模式**：每讲完一课，append 该课内容，不重新生成整个文件
- 每课内容包含：
  - 概念说明
  - 代码块（语法高亮）
  - 流程图
  - 知识点格子
  - API 速查表

---

## 二、HTML 排版硬性要求

| 项目 | 要求 |
|------|------|
| 字体大小 | `font-size: 17px` |
| 侧边栏高度 | `height: 100vh`（不能用 `min-height`） |
| 侧边栏滚动 | `overflow-y: auto`，必须可上下滑动 |
| 导航结构 | 手风琴折叠，每课一个父条目 + 子标题列表 |
| 默认状态 | 子标题默认收起（`max-height: 0`） |
| 展开状态 | 点击后添加 `.open` class → `max-height: 400px` |
| 箭头图标 | `▶` 展开时旋转 90° |
| 自动展开 | `IntersectionObserver` 监听正文滚动，自动展开对应课程、收起其他 |
| 课程顺序 | 侧边栏 nav 顺序必须与正文顺序严格一致（第1课在最上方） |
| 课程颜色 | 每课正文横幅（banner）用不同颜色区分 |

---

## 三、已踩过的坑（避免重复）

| 问题 | 原因 | 正确做法 |
|------|------|----------|
| 侧边栏无法滚动，第4课后看不到 | 用了 `min-height:100vh` | 改为 `height:100vh; overflow-y:auto` |
| 课程顺序乱（第3课跑到第2课前面） | nav div 顺序与正文不一致 | 每次 append 后检查 nav 与正文的课程顺序 |
| 新课 nav 里残留上一课的孤立块 | append 时误留了多余的 `nav-sub` | 检查 nav 结构，确保每课只有一个 nav-lesson + 一个 nav-sub |
| Python 文件 SyntaxError | f-string 里用了中文花括号引号 `"` `"` | 统一用 ASCII 单引号 `'` |

---

## 四、环境配置（阿里百炼）

```python
# .env 文件（禁止提交 git）
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
MODEL=qwen-plus
```

```python
# 代码中读取
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)
```

---

## 五、.gitignore 必须包含

```
.env
.idea/
.venv/
venv/
__pycache__/
*.pyc
```

---

## 六、新项目使用方式

1. 复制本文件到新项目根目录
2. 创建 `.env` 填入 API Key
3. 创建 `.gitignore`（参考第五节）
4. 告知 Claude：**"按照 LEARNING_SPEC.md 的规范，帮我学习 xxx"**
