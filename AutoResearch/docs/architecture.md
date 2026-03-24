# AutoResearch 架构简介

## 1. 系统目标

AutoResearch 的目标不是为某一个固定科研任务写死一套流程，而是提供一套通用的研究自动化框架：

- 用户先用自然语言描述任务
- 系统把自然语言编译成一个可执行、可阅读、可修改的中间表示
- Runtime 再严格按照这个中间表示运行

这个中间表示就是 **Task Pack**。

## 2. 两层结构

整个系统分为两层。

### Layer 1: Task Pack Compiler

输入：

- `task.md`

输出：

- `taskpack.json`
- `docs/*.md`
- `prompts/*.md`
- `skills/*.md`
- `schemas/*.json`
- `compiler_report.json`
- `README.md`

这一层的职责是把自然语言任务说明变成一个**人类可编辑的任务包**。

它不是直接执行研究任务，而是完成下面几件事：

- 识别任务类型
- 提取目标、成功标准、约束、非目标、预算
- 生成角色、prompts、skills、schemas、workflow
- 插入 review / MCP review / repair 流程
- 输出一个可直接交给 Runtime 的任务包

### Layer 2: Task Pack Runtime

输入：

- `taskpack.json`

这一层不再理解自然语言，只负责：

- 加载 Task Pack
- 校验结构和引用关系
- 初始化 runtime store
- 解析 workflow graph
- 执行 dry-run 或 stub-run
- 为后续真实 agent/tool/MCP 执行提供统一入口

也就是说，Runtime 只认 IR，不认自然语言。

## 3. 为什么要分两层

这样设计有几个直接好处：

- 用户入口自然，不需要一开始就理解 JSON 结构
- 后端执行统一，不同任务共享一套 Runtime
- 编译结果可读、可审阅、可手动微调
- prompts / skills / workflow / schema 不再硬编码在框架里
- 同一个任务既可以“自然语言驱动”，也可以“直接修改 IR 驱动”

## 4. Task Pack 是什么

Task Pack 是 AutoResearch 的中间表示，也是 Runtime 的唯一入口。

注意这里的“唯一入口”不是说所有内容都塞进一个 JSON，而是：

- `taskpack.json` 是入口
- 其他资产都由它引用

推荐的任务包目录形态如下：

```text
taskpacks/<task_id>/
  task.md
  taskpack.json
  README.md
  compiler_report.json
  docs/
  prompts/
  skills/
  schemas/
```

其中：

- `task.md` 是原始自然语言描述
- `taskpack.json` 是运行入口
- `docs/` 放任务说明、参考资料、checklist
- `prompts/` 放角色 prompt
- `skills/` 放当前任务专用 skills
- `schemas/` 放结构化输出 schema
- `compiler_report.json` 解释编译器做了哪些推断

## 5. Task Pack 的核心组成

`taskpack.json` 主要描述“结构”和“关系”，不承载大段长文本。

核心 section 包括：

- `meta`
- `mission`
- `knowledge`
- `field_library`
- `schemas`
- `role_library`
- `state_model`
- `objective_model`
- `adapter_registry`
- `workflow`
- `reporting`

它们分别回答下面这些问题：

- 这是一个什么任务
- 成功标准是什么
- 有哪些知识资产
- 有哪些结构化字段
- 有哪些角色
- 状态如何存储
- 目标如何评价
- 有哪些 agent / tool / evaluator / reviewer
- 流程图怎么走

## 6. Workflow 是什么

AutoResearch 不把流程写死成线性阶段，而是用一个 workflow graph 来表达。

典型流程如下：

```text
draft_proposal
  -> mcp_review
  -> internal_review
  -> review_gate
  -> execute_round
  -> evaluate_result
  -> record_round
```

如果 review 要求修改，还会进入：

```text
review_gate
  -> repair_plan
  -> mcp_review
```

这样做的原因是不同任务的结构完全不同：

- 论文 idea 研究更偏 proposal / critique / synthesis
- 论文写作更偏 draft / review / revise
- 性能优化更偏 benchmark / patch / benchmark / compare

所以真正稳定的不是“某个固定流程”，而是“可配置的 workflow graph”。

## 7. Skills 在系统里的位置

Skills 分两类。

### Compile-time skills

用于第一层编译过程，例如：

- 任务分类
- workflow 组装
- schema 生成
- prompt 生成
- task-local skill 生成

### Runtime skills

用于第二层执行过程，例如：

- 实验执行规范
- benchmark 使用说明
- 论文写作约束
- review checklist
- profiling 方法

当前设计里，编译器默认会优先生成**任务私有的 runtime skills**，先放在当前 task pack 里，而不是立刻沉淀成共享技能。

## 8. MCP Review 的位置

外部 MCP review 是 Runtime 中的一等能力。

在 Task Pack 里，它不是隐式发生，而是通过显式 workflow node 表达：

```json
{
  "kind": "mcp_review",
  "uses": "external_mcp_review"
}
```

这样做的好处是：

- review 流程对人可见
- review 点可以调整
- review 输入输出可以建模
- repair 分支可以显式连接

## 9. 当前代码边界

目前代码已经具备：

- Markdown -> Task Pack bundle 的编译能力
- Task Pack 的结构校验
- `mcp_review` 节点支持
- Runtime dry-run
- Runtime stub-run

目前还没有完成的部分是：

- 真正调用外部 agent
- 真正调用 MCP
- 真正执行 tool / evaluator / gate 的业务逻辑

也就是说，框架层和 IR 层已经落下来了，真实执行层后续可以继续补。

## 10. 一句话总结

AutoResearch 的核心思想可以概括为：

**自然语言定义任务，Task Pack 定义执行，Runtime 只执行 Task Pack。**
