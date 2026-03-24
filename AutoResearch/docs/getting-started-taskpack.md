# AutoResearch 入门：Task Pack / 中间表示用户

这份文档面向这样一类用户：

- 你已经理解 AutoResearch 的两层结构
- 你愿意直接修改中间表示
- 你希望精细控制 workflow、prompts、skills、schemas、review 点

对你来说，真正的工作对象不是自然语言，而是 **Task Pack bundle**。

## 1. 你在修改什么

一个任务包通常长这样：

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

其中最重要的是：

- `taskpack.json`
- `prompts/*.md`
- `skills/*.md`
- `docs/*.md`

## 2. 先建立一个正确心智模型

不要把 `taskpack.json` 理解成“所有内容的容器”。

更准确的理解是：

- `taskpack.json` 描述结构、引用关系、workflow 和运行接口
- 长文本资产放在外部文件中
- Runtime 从 `taskpack.json` 出发，去找到其他资产

也就是说：

- JSON 管结构
- Markdown 管内容

## 3. 哪些地方最常改

最常见的修改区域如下。

### `mission`

当你需要改变任务定义时改这里：

- `task_type`
- `objective`
- `success_definition`
- `hard_constraints`
- `allowed_edit_paths`
- `budget`

### `knowledge`

当你需要替换知识资产时改这里：

- `descriptions`
- `references`
- `checklists`
- `prompt_library`
- `skill_library`

这里通常是“改引用关系”，而不是塞长文本。

### `role_library`

当你需要改角色职责时改这里：

- 哪些角色存在
- 每个角色的 prompt chain
- 每个角色使用哪些 skills
- 每个角色能用哪些 tools
- 每个角色输出什么 schema

### `adapter_registry`

当你需要改执行能力时改这里：

- agent adapter
- tool adapter
- evaluator adapter
- gate adapter
- reviewer adapter

如果要接入外部 MCP review，主要看：

- `adapter_registry.reviewers`

### `workflow`

当你需要改流程结构时改这里：

- `entry_node`
- `nodes`
- `edges`
- `loop_policies`
- `failure_policies`

这是最敏感的一层，也是最强大的一层。

## 4. 什么不建议直接内联

下面这些内容不建议直接塞进 `taskpack.json`：

- 很长的 system prompt
- reviewer checklist 正文
- 复杂的 task-specific skill
- 大段任务说明

这些都应该放在：

- `prompts/`
- `skills/`
- `docs/`

然后由 `taskpack.json` 引用。

这样做有两个好处：

- 人类更容易读和改
- diff 更清楚

## 5. 怎样修改 workflow

一个典型 workflow 节点可能长这样：

```json
{
  "mcp_review": {
    "kind": "mcp_review",
    "uses": "external_mcp_review",
    "input": {
      "proposal": "proposal"
    },
    "output": "external_review"
  }
}
```

你在改 workflow 时，重点检查四件事：

1. `kind` 是否被 Runtime 支持
2. `uses` 是否能在对应 registry 中找到
3. `input` / `output` 是否和上下游对得上
4. `edges` 是否把节点正确连起来

目前支持检查的主要节点类型包括：

- `agent`
- `tool`
- `evaluator`
- `gate`
- `record`
- `mcp_review`

## 6. 怎样修改 MCP Review

如果你要调整外部 review，一般改两处：

### 在 `adapter_registry.reviewers` 里定义 reviewer

例如：

```json
{
  "reviewers": {
    "external_mcp_review": {
      "type": "mcp_reviewer",
      "config": {
        "server": "replace-me",
        "tool": "replace-me"
      }
    }
  }
}
```

### 在 `workflow.nodes` 里插入 `mcp_review`

这样 review 点是可见、可调、可复用的。

如果你要增加多个 review 点，也建议每个 review 点都显式建 node，而不是把逻辑藏到某个 tool 里。

## 7. 哪些改动适合直接手改

适合直接手改的情况包括：

- 改 prompt 内容
- 增加或删减一个 skill
- 调整 reviewer checklist
- 增加一个 workflow node
- 调整 edge 条件
- 替换某个 adapter
- 修改成功标准或预算

不太适合直接手改、而更适合重新 compile 的情况包括：

- 任务本身已经换了
- 任务类型判断完全错了
- 你要从写作任务切到性能优化任务
- 角色体系需要整体重做

## 8. 修改后的标准流程

每次修改中间表示后，建议都走下面这套流程：

### 先校验

```bash
python -m AutoResearch validate --taskpack taskpacks/my-task/taskpack.json
```

### 再看摘要

```bash
python -m AutoResearch describe --taskpack taskpacks/my-task/taskpack.json --json
```

这里主要确认：

- roles 是否正确
- prompts / skills 是否都被识别
- reviewers 是否正确
- workflow node 数量是否符合预期

### 再做 dry-run

```bash
python -m AutoResearch run \
  --taskpack taskpacks/my-task/taskpack.json \
  --runtime-root . \
  --dry-run \
  --json
```

你应该重点看：

- 节点顺序
- 分支条件
- repair 环是否正确
- `mcp_review` 是否出现在你想要的位置

## 9. 常见错误

### 错误 1：改了 prompt 文件，但没更新引用

如果你重命名了 `prompts/*.md`，记得同步改：

- `knowledge.prompt_library`
- `role_library.<role>.prompt_chain`

### 错误 2：加了 skill 文件，但角色没引用

新增 skill 文件还不够，你还需要：

- 把它加入 `knowledge.skill_library`
- 把它挂到某个 role 的 `skills`

### 错误 3：workflow 节点引用了不存在的 adapter

例如：

- `kind: "mcp_review"`
- 但 `uses` 不在 `adapter_registry.reviewers` 里

这种情况 `validate` 会报错。

### 错误 4：把太多语义塞回 JSON

如果 `taskpack.json` 越改越长、越改越难读，通常说明你把本该放在 Markdown 资产里的内容塞回 JSON 里了。

## 10. 推荐编辑策略

比较稳的策略是：

1. 只把 `taskpack.json` 当“控制平面”
2. 把 prompts / skills / docs 当“内容平面”
3. 每次改完先 `validate`
4. 每次大改 workflow 前先做一份 dry-run

这样任务包会长期保持可维护。

## 11. 一句话总结

如果你直接修改中间表示，那么你其实是在做两件事：

- 定义这个任务应该怎么被执行
- 定义这个任务应该如何被 review、repair 和记录

所以修改 Task Pack 不是在“调配置参数”，而是在**编排一个研究工作流**。
