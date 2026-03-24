# AutoResearch 入门：自然语言用户

这份文档面向这样一类用户：

- 你想做一个 AutoResearch 任务
- 你不想一开始就手写 `taskpack.json`
- 你更希望先用自然语言描述任务

对你来说，最重要的入口是：

- `task.md`
- `python -m AutoResearch compile`

## 1. 你需要准备什么

准备一份 Markdown 文档，例如：

```md
# CPU 算子性能优化

## Objective

优化某个 CPU 算子的性能，同时保证结果正确。

## Success Criteria

- 有明确 benchmark
- 延迟下降或吞吐提升
- 正确性不回退

## Constraints

- 不允许破坏正确性
- 改动范围要可控

## Skills

- benchmark harness
- review checklist

## Budget

max_rounds: 4
budget_hours: 2.5
```

你不需要把文档写得很“格式化”，但建议尽量写清楚下面这些信息：

- 目标是什么
- 成功标准是什么
- 有哪些硬约束
- 哪些内容不重要
- 需要什么 skills
- 大概预算是多少

如果你不写全，编译器会自动补一些默认值，但你后面最好检查一下它补得是否合理。

## 2. 第一步：获取模板

如果你不想从空白开始，可以先查看模板路径：

```bash
python -m AutoResearch request-template
```

然后参考模板写一份自己的 `task.md`。

## 3. 第二步：把自然语言编译成任务包

执行：

```bash
python -m AutoResearch compile --task-md path/to/task.md --output-root taskpacks
```

如果你想指定任务 ID：

```bash
python -m AutoResearch compile \
  --task-md path/to/task.md \
  --output-root taskpacks \
  --task-id my-task
```

编译完成后，你会得到一个目录：

```text
taskpacks/my-task/
```

里面通常包含：

- `task.md`
- `taskpack.json`
- `README.md`
- `compiler_report.json`
- `docs/`
- `prompts/`
- `skills/`
- `schemas/`

## 4. 第三步：先读这三个文件

编译完成后，建议先看：

1. `compiler_report.json`
2. `README.md`
3. `taskpack.json`

你主要要确认三件事：

- 编译器识别的任务类型是否正确
- 生成的 workflow 是否符合你的预期
- 生成的 prompts / skills 是否方向对了

## 5. 第四步：决定是否手动微调

虽然你是“自然语言入口用户”，但依然建议做一次轻量检查。

最常见的微调位置有：

- `docs/task_brief.md`
- `docs/reference_map.md`
- `docs/review_checklist.md`
- `prompts/*.md`
- `skills/*.md`
- `taskpack.json`

通常不需要一开始就改 schema。

## 6. 第五步：做结构校验

在真正运行前，先校验：

```bash
python -m AutoResearch validate --taskpack taskpacks/my-task/taskpack.json
```

如果环境里没装 `jsonschema`，结构 schema 校验会跳过，但框架内的引用关系和 workflow 校验仍然有效。

## 7. 第六步：先做 dry-run

先不要直接跑真实执行，先看运行图是否合理：

```bash
python -m AutoResearch run \
  --taskpack taskpacks/my-task/taskpack.json \
  --runtime-root . \
  --dry-run
```

这一步会告诉你：

- entry node 是什么
- workflow 节点顺序是什么
- 是否包含 `mcp_review`
- review / repair / execute 分支怎么走

如果 dry-run 看起来不对，就回去改任务包，而不是硬跑。

## 8. 第七步：什么时候应该重新 compile

下面这些情况通常适合回到 `task.md` 重编译：

- 任务目标变了
- 成功标准变了
- 任务类型判断错了
- workflow 需要大改
- 需要一批新的 prompts / skills

下面这些情况通常更适合直接手改任务包：

- 某个 prompt 语气不对
- 某个 skill 内容不够具体
- 某个 reviewer checklist 不够严格
- 某个 workflow 节点名字或引用需要小修

## 9. 编写自然语言任务时的建议

建议你写得更像“任务说明书”，不要只写一句模糊目标。

推荐包含：

- Objective
- Success Criteria
- Constraints
- Non-Goals
- Deliverables
- References
- Skills
- Prompts
- Budget

写得越清楚，编译器生成的 Task Pack 就越靠谱。

## 10. 一条推荐工作流

最推荐的使用方式是：

1. 写 `task.md`
2. `compile`
3. 看 `compiler_report.json`
4. 轻量修改 `docs/`、`prompts/`、`skills/`
5. `validate`
6. `run --dry-run`
7. 再进入真实执行阶段

## 11. 你最需要记住的一件事

对自然语言用户来说，`task.md` 不是最终配置，而是**生成任务包的源文件**。

真正运行的是 `taskpack.json`，不是 `task.md`。
