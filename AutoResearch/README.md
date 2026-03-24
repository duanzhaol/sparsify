# AutoResearch

现在这份代码实现的是一套两层结构的通用 AutoResearch 内核：

1. `task.md -> Task Pack bundle`
2. `taskpack.json -> Runtime`

核心原则：

- 用户入口是自然语言 Markdown
- 编译结果是**人类可读、可手改**的任务包目录
- Runtime 只消费 `taskpack.json`
- prompts / skills / docs / schemas 都是任务包资产，不硬编码在框架里

## CLI

```bash
python -m AutoResearch request-template
python -m AutoResearch compile --task-md path/to/task.md --output-root taskpacks
python -m AutoResearch validate --taskpack taskpacks/my-task/taskpack.json
python -m AutoResearch describe --taskpack taskpacks/my-task/taskpack.json
python -m AutoResearch init-runtime --taskpack taskpacks/my-task/taskpack.json --runtime-root .
python -m AutoResearch run --taskpack taskpacks/my-task/taskpack.json --dry-run
```

## Layer 1: Compiler

输入：

- `task.md`

输出：

- `taskpacks/<task_id>/task.md`
- `taskpacks/<task_id>/taskpack.json`
- `taskpacks/<task_id>/docs/*.md`
- `taskpacks/<task_id>/prompts/*.md`
- `taskpacks/<task_id>/skills/*.md`
- `taskpacks/<task_id>/schemas/*.json`
- `taskpacks/<task_id>/compiler_report.json`
- `taskpacks/<task_id>/README.md`

编译器默认是“脚手架优先”：

- 先产出一份可审阅、可微调的任务包
- 默认生成任务私有 skills
- 默认在 workflow 里插入显式 `mcp_review` 节点

## Layer 2: Runtime

Runtime 目前提供：

- Task Pack 加载与校验
- workflow 图一致性校验
- runtime store 初始化
- `run --dry-run` 工作流预览
- `run` stub-run：初始化 store，并把计划节点写入 timeline/report

Runtime 现在不会真正调用外部 agent/tool/MCP；它负责统一消费 IR、验证 IR，并把执行骨架跑通。

## 保留内容

- `AutoResearch/compiler.py`
- `AutoResearch/execution.py`
- `AutoResearch/taskpack.py`
- `AutoResearch/runtime.py`
- `AutoResearch/examples/taskpack.schema.json`
- `AutoResearch/examples/taskpack.template.json`
- `AutoResearch/examples/task.request.template.md`
- `AutoResearch/design/taskpack_meta_model.md`

## 移除内容

- SAE 专用 loop / runner / policy / prompt / controller
- 历史兼容层和旧入口
- 与抽象框架无关的任务特定材料
