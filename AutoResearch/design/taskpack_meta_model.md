# AutoResearch Task Pack 元模型

## 1. 目标

如果要做到“换一个任务，只配一个 JSON 入口文件”，那抽象层不能再停留在：

- 训练脚本
- evaluator
- frontier

这一层。

更深一层的抽象应该是：

**任何 AutoResearch 任务，本质上都是一个“带状态的、多角色、多工具、可审查、可修复的研究工作流图”。**

所以任务配置不应该只描述：

- 跑什么脚本
- 解析什么指标

而应该完整描述：

1. 这个任务的研究对象是什么
2. 这个任务涉及哪些角色
3. 每个角色用什么 prompt、什么 skill、什么工具
4. 角色之间通过什么结构化对象交互
5. 这些对象的字段定义是什么
6. 状态怎么持久化
7. 评价目标是什么
8. workflow 图怎么走
9. 哪些地方必须 review / gate
10. 出错后怎么 repair / revise / rerun

我把这套抽象叫做：

## 2. Task Pack

每个任务由一个 **Task Pack** 描述。

Task Pack 的入口只有一个：

- `taskpack.json`

但它可以引用额外资源：

- prompt 文件
- skill 文件
- task description
- 参考文档 / paper
- JSON schema
- checklist

所以“一个任务只需要配置一个 JSON 文件”应理解为：

- **一个 JSON 文件是唯一入口**
- 所有任务定义都从它出发
- 其他文件只是它引用的资产，而不是额外的运行配置入口

## 3. 五层抽象

一个真正通用的 Task Pack，至少要覆盖五层。

### 3.1 Mission Layer

定义这个任务“研究什么”。

包括：

- 任务名称
- 任务类型
- 目标
- 成功标准
- 约束
- 允许动作边界

示例任务类型：

- `idea_research`
- `writing_research`
- `experiment_research`
- `performance_optimization`
- `bug_investigation`

### 3.2 Knowledge Layer

定义 agent 可以依赖的知识资产。

包括：

- task descriptions
- prompt library
- skill library
- references
- checklists
- examples

关键点：

- prompt 不是直接散落在代码里，而是知识资产
- skill 不是框架逻辑，而是可复用的知识/流程模块
- description / docs / paper 也是一等公民

### 3.3 Contract Layer

定义结构化对象，也就是“字段定义”。

这是你提到的重点。

我建议把“字段定义”分成两层：

1. `field_library`
   给 prompt、人类、reviewer 用的语义说明。
2. `schemas`
   给 runtime 做机器校验。

这两层不能混为一谈。

比如：

- `proposal.target.entity_name`
  是什么语义
- `evaluation.metrics.latency_ms`
  单位是什么
- `review.findings[].severity`
  合法枚举是什么

这些是 `field_library`。

而：

- `proposal` 是否缺字段
- 字段类型是否正确
- 枚举是否合法

这些是 `schema`。

### 3.4 Capability Layer

定义系统有哪些“能力”可被 workflow 调用。

包括：

- roles
- agents
- tools
- evaluators
- summarizers
- gates
- recorders

其中最重要的是 `role_library`。

因为一个任务真正的差异往往不只是工具不同，而是角色不同。

例如：

- planner
- implementer
- reviewer
- writer
- critic
- profiler
- citation_auditor

每个 role 都应该绑定：

- prompt chain
- skills
- tool access
- output schema
- context view

### 3.5 Control Layer

定义 workflow。

这里不能只用线性 stage list。

更深层的抽象应该是：

- 一个有向图 `workflow graph`

因为不同任务的流程形态不一样：

- 有的要 plan -> review -> revise -> execute
- 有的要 literature -> synthesize -> design -> write
- 有的要 benchmark -> patch -> benchmark -> compare -> review

所以 workflow 应该由：

- nodes
- edges
- conditions
- loop policies
- failure policies

共同定义。

## 4. Task Pack 应该包含什么

一个完整的 `taskpack.json` 我建议至少包含下面十个 section。

### 4.1 `meta`

元信息：

- id
- name
- version
- owner
- tags
- extends

`extends` 很重要。

因为很多任务不是从零定义，而是继承模板：

- `base_experiment_research`
- `base_writing_research`
- `base_perf_optimization`

### 4.2 `mission`

任务目标：

- `task_type`
- `objective`
- `success_definition`
- `non_goals`
- `hard_constraints`
- `allowed_edit_paths`
- `budget`

### 4.3 `knowledge`

知识资产：

- `descriptions`
- `references`
- `prompt_library`
- `skill_library`
- `checklists`

这里要特别强调：

#### Prompt Library

不是只有一个 system prompt。

应至少支持：

- global system prompt
- planner prompt
- implementer prompt
- reviewer prompt
- writer prompt
- repair prompt
- summarizer prompt

#### Skill Library

skill 也不能只理解成“某个 repo 内的 skill 文件”。

更合理的分类是四种：

1. `domain_skill`
   领域知识，如 CPU 算子优化、论文结构、实验设计。
2. `process_skill`
   流程知识，如如何做 ablation、如何做 repair、如何做 paper review。
3. `tool_skill`
   工具知识，如如何使用 profiler、如何调用 benchmark harness。
4. `safety_skill`
   约束/检查表，如 allowed paths、citation rules、review checklist。

### 4.4 `field_library`

字段语义定义。

建议至少包含：

- `enums`
- `records`
- `units`
- `aliases`

例如：

- `proposal.change_mode`
- `review.verdict`
- `evaluation.metrics.latency_ms`
- `decision.label`
- `report.sections`

为什么这一层重要：

- prompt 生成时可以引用字段说明
- reviewer 可以对字段做语义判断
- 多任务间可以复用 record definition

### 4.5 `schemas`

机器可校验 schema。

至少建议：

- `proposal`
- `review`
- `execution`
- `evaluation`
- `decision`
- `report`

### 4.6 `role_library`

这是整个抽象里最关键的 section 之一。

每个 role 应包含：

- `purpose`
- `prompt_chain`
- `skills`
- `allowed_tools`
- `input_contract`
- `output_schema`
- `context_view`
- `behavioral_rules`

例如：

#### planner

- 负责提出下一步方案
- 输出 `proposal`

#### reviewer

- 负责审 proposal / patch / result
- 输出 `review`

#### implementer

- 负责把 proposal 变成 patch 或执行动作
- 输出 `patch_result` 或 `execution_request`

#### writer

- 负责写 paper section / outline / claim revision
- 输出 `draft`

### 4.7 `state_model`

框架不能再写死只有 `memory.json` 和 `frontier.json`。

应该允许每个任务定义自己的 state sections：

- `objective_state`
- `memory`
- `artifacts`
- `timeline`
- `reports`
- `entity_registry`

同时还需要：

- `views`
  不同 role 看到的上下文摘要
- `retention`
  哪些 state 长存，哪些裁剪
- `compression`
  如何压缩历史

例如：

- planner 看 `objective_digest + recent_rounds + open_hypotheses`
- writer 看 `accepted_results + outline + citation_bank`
- reviewer 看 `artifact_diff + checklist + previous_reviews`

### 4.8 `objective_model`

这层定义“什么叫研究上更好”。

我建议拆成四块：

- `optimization_mode`
  `pareto | scalar | threshold_gate | rubric`
- `metrics`
- `constraints`
- `decision_policy`

这里的 `rubric` 很重要。

因为不是所有任务都有数值指标。

例如论文写作任务，可能靠 rubric：

- clarity
- novelty articulation
- evidence grounding
- citation correctness

### 4.9 `adapter_registry`

任何任务最终都需要调用能力。

建议统一注册：

- `agents`
- `tools`
- `evaluators`
- `summarizers`
- `recorders`
- `gates`

其中：

- `agent` 是一个能基于 role 工作的智能执行单元
- `tool` 是 shell / python / MCP / search / benchmark
- `evaluator` 是把 artifact 变成 structured evaluation
- `gate` 是 approve/reject/revise 逻辑

### 4.10 `workflow`

workflow 应定义为图，不是线性数组。

建议结构：

- `entry_node`
- `nodes`
- `edges`
- `loop_policies`
- `failure_policies`
- `stop_conditions`

其中 node 可以统一抽象成少数几类：

- `agent`
- `tool`
- `evaluator`
- `gate`
- `router`
- `reducer`
- `record`

这样很多原来看起来很不一样的流程都能统一表示。

## 5. 为什么“role”是比“stage”更深的抽象

这是和我上一版设计相比更进一步的地方。

上一版里我还是偏向写：

- `llm_plan`
- `llm_apply`
- `mcp_review`

这还不够深。

更深的抽象应该是：

- 一个 node 由某个 `role` 执行
- role 决定 prompt、skills、tools、output contract

也就是说，不是先规定“这里是 llm_plan”，而是先规定“这里由 planner role 工作”。

这样：

- 论文 idea 研究
  planner 产生 proposal
- 论文写作
  writer 产生 draft
- CPU 优化
  profiler reviewer 产生 review

统一性更强。

## 6. 为什么“skill”不是主抽象，但必须是一等公民

你问“每个任务是不是都需要一系列 skills 辅助”，答案是：

- 不是所有任务都需要很多 skill
- 但 `skill_library` 必须是一等公民

原因：

1. skill 是可重用知识模块
2. skill 可以按 role、按 node 激活
3. skill 可以独立演化，不污染框架逻辑

我建议 skill 的激活方式应该可配置：

- 全局启用
- 某个 role 启用
- 某个 node 启用
- 满足条件时启用

例如：

- `citation_audit` 只给 writer/reviewer
- `profile_analysis` 只给 profiler/reviewer
- `repair_checklist` 只在 repair loop 打开

## 7. 一份 Task Pack 如何适配三类完全不同的任务

### 7.1 论文 idea auto research

Mission：

- 评估一个论文 idea 是否值得推进

核心 role：

- literature_reviewer
- planner
- experiment_designer
- critic

主要 outputs：

- novelty review
- risk list
- experiment plan
- next-step recommendation

objective：

- 不一定是 numeric
- 更像 rubric + review gate

### 7.2 论文写作 auto research

Mission：

- 生成更高质量的论文结构和文本

核心 role：

- writer
- claim_reviewer
- citation_auditor
- editor

主要 outputs：

- outline
- section draft
- citation review
- revision report

objective：

- rubric
- gate
- maybe no frontier

### 7.3 CPU 算子性能优化

Mission：

- 提升 latency / throughput / cache efficiency

核心 role：

- planner
- implementer
- profiler
- perf_reviewer

主要 outputs：

- proposal
- patch
- benchmark result
- regression review

objective：

- scalar or pareto
- numeric evaluator
- patch review + result review 很重要

## 8. 最终应该怎样组织目录

如果按这个元模型来，我建议最终目录长这样：

```text
AutoResearch/
  engine/
  registries/
  taskpacks/
    sae/
      taskpack.json
      prompts/
      skills/
      docs/
      schemas/
    paper_idea/
      taskpack.json
      prompts/
      skills/
      docs/
      schemas/
    paper_writing/
      taskpack.json
      prompts/
      skills/
      docs/
      schemas/
    cpu_opt/
      taskpack.json
      prompts/
      skills/
      docs/
      schemas/
```

这里每个任务真正的“配置入口”都还是只有一个：

- `taskpack.json`

## 9. 一个真正通用的最小运行时，只需要理解什么

做到这一步之后，runtime 本身只需要理解下面这些通用概念：

- taskpack
- role
- schema
- state store
- adapter
- node
- edge
- gate
- objective
- artifact

它不需要理解：

- SAE
- 论文
- CPU kernel

这些都应该属于 taskpack 层。

## 10. 最关键的抽象结论

如果只保留一句话，那就是：

**AutoResearch 不应该被抽象成“可配置训练循环”，而应该被抽象成“可配置研究工作流图执行器”。**

在这个抽象下：

- prompt 是知识资产
- skill 是能力插件
- 字段定义是 contract
- review 是 gate
- evaluator 是 observation->judgement 转换器
- task JSON 是工作流图入口

这才足够高层，也才有可能真的支撑你说的三类完全不同任务。
