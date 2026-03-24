# AutoResearch

现在这份代码只保留一个最小的、高度抽象的内核：

- `taskpack.json` 作为任务唯一入口
- Task Pack 加载与校验
- workflow 图一致性校验
- runtime store 初始化

它不再内置 SAE、论文写作、CPU 优化等任何具体任务逻辑。
这些都应该通过 Task Pack 描述。

运行入口：

```bash
python -m AutoResearch validate --taskpack path/to/taskpack.json
python -m AutoResearch describe --taskpack path/to/taskpack.json
python -m AutoResearch init-runtime --taskpack path/to/taskpack.json --runtime-root .
```

保留内容：

- `AutoResearch/design/taskpack_meta_model.md`
- `AutoResearch/examples/taskpack.schema.json`
- `AutoResearch/examples/taskpack.template.json`
- 通用 Task Pack runtime 代码

移除内容：

- SAE 专用 loop / runner / policy / prompt / controller
- 历史兼容层和旧入口
- 与抽象框架无关的任务特定材料
