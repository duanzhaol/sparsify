# Qwen3-4B Deploy Objective AutoResearch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一套独立于现有 `research/AutoResearch/` 的轻量专用 autoresearch 框架，用于在 `Qwen3-4B` / `layers.[17].self_attn.q_proj` / `product_key_expert_jumprelu` 设定下最小化 `total_cost_ratio + best_exceed_alpha_0.50`。

**Architecture:** 新框架位于 `research/deploy_objective_autoresearch/`，只负责固定目标下的状态管理、trial 启停/续跑、checkpoint 解析、agent 决策接线与深度优先调度。训练本身仍通过现有 `scripts/autoresearch_test.sh` 执行，但新框架不复用旧 autoresearch 的状态目录、控制器、prompt、policy 或历史文件。

**Tech Stack:** Python 3.12、标准库 `dataclasses/json/pathlib/subprocess`、仓库现有 `pytest`、现有训练脚本 `scripts/autoresearch_test.sh`、Codex background/autoresearch 外层编排。

---

## File Structure

- Create: `research/deploy_objective_autoresearch/__init__.py`
- Create: `research/deploy_objective_autoresearch/config.py`
- Create: `research/deploy_objective_autoresearch/state_store.py`
- Create: `research/deploy_objective_autoresearch/metrics_extractor.py`
- Create: `research/deploy_objective_autoresearch/trial_runner.py`
- Create: `research/deploy_objective_autoresearch/proposal_policy.py`
- Create: `research/deploy_objective_autoresearch/agent_interface.py`
- Create: `research/deploy_objective_autoresearch/scheduler.py`
- Create: `research/deploy_objective_autoresearch/main.py`
- Create: `tests/research/test_deploy_objective_config_state.py`
- Create: `tests/research/test_deploy_objective_metrics.py`
- Create: `tests/research/test_deploy_objective_trial_runner.py`
- Create: `tests/research/test_deploy_objective_scheduler.py`
- Modify: `research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective/README.md`

### Task 1: 配置模型与独立状态目录

**Files:**
- Create: `research/deploy_objective_autoresearch/__init__.py`
- Create: `research/deploy_objective_autoresearch/config.py`
- Create: `research/deploy_objective_autoresearch/state_store.py`
- Test: `tests/research/test_deploy_objective_config_state.py`

- [ ] **Step 1: 写失败测试，固定运行配置与 run bootstrap 语义**

```python
from pathlib import Path

from research.deploy_objective_autoresearch.config import SearchConfig
from research.deploy_objective_autoresearch.state_store import StateStore


def test_default_search_config_uses_fixed_qwen3_4b_context(tmp_path: Path):
    cfg = SearchConfig.default(run_root=tmp_path)
    assert cfg.model_path.endswith("models/Qwen3-4B")
    assert cfg.hookpoints == "layers.[17].self_attn.q_proj"
    assert cfg.architecture == "product_key_expert_jumprelu"
    assert cfg.checkpoint_interval_tokens == 15_000_000
    assert cfg.max_new_trials == 200


def test_state_store_bootstrap_creates_independent_files(tmp_path: Path):
    cfg = SearchConfig.default(run_root=tmp_path)
    store = StateStore.bootstrap(cfg)
    assert (tmp_path / "run_config.json").exists()
    assert (tmp_path / "state.json").exists()
    assert (tmp_path / "leaderboard.json").exists()
    assert (tmp_path / "trials.jsonl").exists()
    assert store.load_state()["active_trial_id"] is None
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run: `pytest tests/research/test_deploy_objective_config_state.py -v`
Expected: FAIL，提示 `ModuleNotFoundError: No module named 'research.deploy_objective_autoresearch'`

- [ ] **Step 3: 实现最小配置对象与状态初始化**

```python
# research/deploy_objective_autoresearch/config.py
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SearchConfig:
    run_root: Path
    model_path: str
    hookpoints: str
    elbow_threshold_path: str
    architecture: str
    checkpoint_interval_tokens: int = 15_000_000
    max_new_trials: int = 200
    nproc_per_node: int = 2
    max_active_trials: int = 1

    @classmethod
    def default(cls, run_root: Path) -> "SearchConfig":
        home = Path.home()
        return cls(
            run_root=run_root,
            model_path=str(home / "models" / "Qwen3-4B"),
            hookpoints="layers.[17].self_attn.q_proj",
            elbow_threshold_path=str(
                Path.cwd() / "thresholds" / "Qwen3-4B" / "thresholds_q.json"
            ),
            architecture="product_key_expert_jumprelu",
        )

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["run_root"] = str(self.run_root)
        return payload
```

```python
# research/deploy_objective_autoresearch/state_store.py
import json
from dataclasses import dataclass
from pathlib import Path

from .config import SearchConfig


@dataclass
class StateStore:
    root: Path

    @classmethod
    def bootstrap(cls, cfg: SearchConfig) -> "StateStore":
        root = cfg.run_root
        (root / "trials").mkdir(parents=True, exist_ok=True)
        (root / "run_config.json").write_text(json.dumps(cfg.to_json(), indent=2))
        (root / "state.json").write_text(json.dumps({
            "status": "idle",
            "active_trial_id": None,
            "new_trial_count": 0,
            "checkpoint_interval_tokens": cfg.checkpoint_interval_tokens,
        }, indent=2))
        (root / "leaderboard.json").write_text(json.dumps({"entries": []}, indent=2))
        (root / "trials.jsonl").touch()
        (root / "agent_decisions.jsonl").touch()
        return cls(root)

    def load_state(self) -> dict:
        return json.loads((self.root / "state.json").read_text())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/research/test_deploy_objective_config_state.py -v`
Expected: PASS

- [ ] **Step 5: 提交该任务**

```bash
git add research/deploy_objective_autoresearch/__init__.py \
        research/deploy_objective_autoresearch/config.py \
        research/deploy_objective_autoresearch/state_store.py \
        tests/research/test_deploy_objective_config_state.py
git commit -m "feat: bootstrap deploy-objective autoresearch state"
```

### Task 2: 指标抽取与 objective 计算

**Files:**
- Create: `research/deploy_objective_autoresearch/metrics_extractor.py`
- Modify: `research/deploy_objective_autoresearch/state_store.py`
- Test: `tests/research/test_deploy_objective_metrics.py`

- [ ] **Step 1: 写失败测试，覆盖 cost 日志、step metrics、best-so-far objective**

```python
import json
from pathlib import Path

from research.deploy_objective_autoresearch.metrics_extractor import (
    extract_trial_snapshot,
)


def test_extract_trial_snapshot_uses_best_exceed_not_latest(tmp_path: Path):
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "[cost][k=80] total_ratio=0.081543x\n"
    )
    metrics_path = tmp_path / "metrics.jsonl"
    rows = [
        {"type": "step", "total_tokens": 15_000_000, "layers.17.self_attn.q_proj/exceed_alpha_0.50": 0.210},
        {"type": "step", "total_tokens": 30_000_000, "layers.17.self_attn.q_proj/exceed_alpha_0.50": 0.198},
        {"type": "step", "total_tokens": 45_000_000, "layers.17.self_attn.q_proj/exceed_alpha_0.50": 0.205},
    ]
    metrics_path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    snap = extract_trial_snapshot(
        log_path=log_path,
        metrics_path=metrics_path,
        hook_metric_prefix="layers.17.self_attn.q_proj",
    )

    assert snap.total_cost_ratio == 0.081543
    assert snap.latest_exceed_alpha_0_50 == 0.205
    assert snap.best_exceed_alpha_0_50 == 0.198
    assert snap.best_objective == 0.279543
    assert snap.tokens_seen == 45_000_000
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `pytest tests/research/test_deploy_objective_metrics.py -v`
Expected: FAIL，提示 `extract_trial_snapshot` 未定义

- [ ] **Step 3: 实现最小 metrics 抽取器**

```python
# research/deploy_objective_autoresearch/metrics_extractor.py
from dataclasses import dataclass
from pathlib import Path
import json
import re

_COST_RE = re.compile(r"total_ratio=([0-9.]+)x")


@dataclass(slots=True)
class TrialSnapshot:
    total_cost_ratio: float
    latest_exceed_alpha_0_50: float | None
    best_exceed_alpha_0_50: float | None
    latest_fvu: float | None
    best_fvu: float | None
    best_objective: float | None
    tokens_seen: int


def extract_trial_snapshot(log_path: Path, metrics_path: Path, hook_metric_prefix: str) -> TrialSnapshot:
    total_cost_ratio = _extract_total_cost_ratio(log_path)
    exceed_key = f"{hook_metric_prefix}/exceed_alpha_0.50"
    fvu_key = f"{hook_metric_prefix}/fvu"
    latest_exceed = None
    best_exceed = None
    latest_fvu = None
    best_fvu = None
    tokens_seen = 0

    with metrics_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("type") != "step":
                continue
            tokens_seen = max(tokens_seen, int(row.get("total_tokens") or 0))
            if exceed_key in row:
                latest_exceed = float(row[exceed_key])
                best_exceed = latest_exceed if best_exceed is None else min(best_exceed, latest_exceed)
            if fvu_key in row:
                latest_fvu = float(row[fvu_key])
                best_fvu = latest_fvu if best_fvu is None else min(best_fvu, latest_fvu)

    best_objective = None
    if total_cost_ratio is not None and best_exceed is not None:
        best_objective = total_cost_ratio + best_exceed

    return TrialSnapshot(
        total_cost_ratio=total_cost_ratio or 0.0,
        latest_exceed_alpha_0_50=latest_exceed,
        best_exceed_alpha_0_50=best_exceed,
        latest_fvu=latest_fvu,
        best_fvu=best_fvu,
        best_objective=best_objective,
        tokens_seen=tokens_seen,
    )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/research/test_deploy_objective_metrics.py -v`
Expected: PASS

- [ ] **Step 5: 提交该任务**

```bash
git add research/deploy_objective_autoresearch/metrics_extractor.py \
        tests/research/test_deploy_objective_metrics.py
git commit -m "feat: add deploy-objective metric extraction"
```

### Task 3: trial 启动、续跑与 checkpoint 阶梯

**Files:**
- Create: `research/deploy_objective_autoresearch/trial_runner.py`
- Modify: `research/deploy_objective_autoresearch/config.py`
- Modify: `research/deploy_objective_autoresearch/state_store.py`
- Test: `tests/research/test_deploy_objective_trial_runner.py`

- [ ] **Step 1: 写失败测试，覆盖新 trial 与继续 trial 的命令拼装**

```python
from pathlib import Path

from research.deploy_objective_autoresearch.config import SearchConfig
from research.deploy_objective_autoresearch.trial_runner import build_trial_env


def test_build_trial_env_for_new_trial_sets_first_15m_checkpoint(tmp_path: Path):
    cfg = SearchConfig.default(run_root=tmp_path)
    env = build_trial_env(
        cfg=cfg,
        params={"K": 80, "NUM_EXPERTS": 326, "LATENTS_PER_EXPERT": 96, "ACTIVE_EXPERTS": 2, "LR": "8e-4", "AUXK_ALPHA": "0.03125"},
        run_name="trial_0001",
        save_dir=tmp_path / "ckpts",
        target_tokens=15_000_000,
        resume=False,
    )
    assert env["MAX_TOKENS"] == "15000000"
    assert env["RESUME"] == "0"
    assert env["HOOKPOINTS"] == "layers.[17].self_attn.q_proj"


def test_build_trial_env_for_resume_sets_resume_flag(tmp_path: Path):
    cfg = SearchConfig.default(run_root=tmp_path)
    env = build_trial_env(
        cfg=cfg,
        params={"K": 80, "NUM_EXPERTS": 326, "LATENTS_PER_EXPERT": 96, "ACTIVE_EXPERTS": 2, "LR": "8e-4", "AUXK_ALPHA": "0.03125"},
        run_name="trial_0001",
        save_dir=tmp_path / "ckpts",
        target_tokens=30_000_000,
        resume=True,
    )
    assert env["MAX_TOKENS"] == "30000000"
    assert env["RESUME"] == "1"
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `pytest tests/research/test_deploy_objective_trial_runner.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 trial env 组装与最小 runner**

```python
# research/deploy_objective_autoresearch/trial_runner.py
from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import time

from .config import SearchConfig


@dataclass(slots=True)
class TrialProcess:
    popen_pid: int
    log_path: Path
    target_tokens: int


def build_trial_env(cfg: SearchConfig, params: dict[str, str | int | float], run_name: str, save_dir: Path, target_tokens: int, resume: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "NPROC_PER_NODE": str(cfg.nproc_per_node),
        "MODEL_PATH": cfg.model_path,
        "HOOKPOINTS": cfg.hookpoints,
        "ELBOW_THRESHOLD_PATH": cfg.elbow_threshold_path,
        "ARCHITECTURE": cfg.architecture,
        "WANDB_PROJECT": cfg.wandb_project,
        "SAVE_DIR": str(save_dir),
        "RUN_NAME": run_name,
        "MAX_TOKENS": str(target_tokens),
        "RESUME": "1" if resume else "0",
        "PRINT_COST_BREAKDOWN": "1",
    })
    for key, value in params.items():
        env[key] = str(value)
    return env


def launch_trial(cfg: SearchConfig, env: dict[str, str], log_path: Path) -> TrialProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_fh:
        proc = subprocess.Popen(
            ["bash", "scripts/autoresearch_test.sh"],
            cwd=Path.cwd(),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
    return TrialProcess(popen_pid=proc.pid, log_path=log_path, target_tokens=int(env["MAX_TOKENS"]))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/research/test_deploy_objective_trial_runner.py -v`
Expected: PASS

- [ ] **Step 5: 提交该任务**

```bash
git add research/deploy_objective_autoresearch/trial_runner.py \
        tests/research/test_deploy_objective_trial_runner.py
git commit -m "feat: add deploy-objective trial runner"
```

### Task 4: 硬规则调度、动态参数合法化与 agent 决策接口

**Files:**
- Create: `research/deploy_objective_autoresearch/proposal_policy.py`
- Create: `research/deploy_objective_autoresearch/agent_interface.py`
- Create: `research/deploy_objective_autoresearch/scheduler.py`
- Test: `tests/research/test_deploy_objective_scheduler.py`

- [ ] **Step 1: 写失败测试，覆盖 gray-zone 继续判定与参数边界修正**

```python
from research.deploy_objective_autoresearch.proposal_policy import normalize_candidate_params
from research.deploy_objective_autoresearch.scheduler import should_force_stop


def test_normalize_candidate_params_clamps_to_dynamic_bounds():
    params = normalize_candidate_params({
        "K": 81,
        "NUM_EXPERTS": 10000,
        "LATENTS_PER_EXPERT": 13,
        "ACTIVE_EXPERTS": 9,
        "LR": 0.1,
        "AUXK_ALPHA": -1,
    })
    assert params["K"] % 8 == 0
    assert 1 <= params["ACTIVE_EXPERTS"] <= 4
    assert 64 <= params["NUM_EXPERTS"] <= 1024
    assert params["LATENTS_PER_EXPERT"] % 8 == 0


def test_should_force_stop_when_objective_far_worse_than_incumbent():
    decision = should_force_stop(
        incumbent_objective=0.24,
        current_best_objective=0.31,
        last_window_delta=0.0001,
        checkpoints_seen=2,
    )
    assert decision is True
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `pytest tests/research/test_deploy_objective_scheduler.py -v`
Expected: FAIL

- [ ] **Step 3: 实现动态边界、硬 gating 与 agent prompt/response schema**

```python
# research/deploy_objective_autoresearch/proposal_policy.py

def _round_to_multiple(value: int, base: int) -> int:
    return max(base, int(round(value / base)) * base)


def normalize_candidate_params(raw: dict[str, int | float]) -> dict[str, int | float]:
    k = min(128, max(16, _round_to_multiple(int(raw.get("K", 64)), 8)))
    num_experts = min(1024, max(64, _round_to_multiple(int(raw.get("NUM_EXPERTS", 256)), 2)))
    latents = min(256, max(16, _round_to_multiple(int(raw.get("LATENTS_PER_EXPERT", 64)), 8)))
    active = min(4, max(1, int(raw.get("ACTIVE_EXPERTS", 2))))
    lr = min(2e-3, max(1e-4, float(raw.get("LR", 8e-4))))
    auxk = min(0.125, max(0.0, float(raw.get("AUXK_ALPHA", 0.03125))))
    return {
        "K": k,
        "NUM_EXPERTS": num_experts,
        "LATENTS_PER_EXPERT": latents,
        "ACTIVE_EXPERTS": active,
        "LR": lr,
        "AUXK_ALPHA": auxk,
    }
```

```python
# research/deploy_objective_autoresearch/scheduler.py

def should_force_stop(incumbent_objective: float | None, current_best_objective: float | None, last_window_delta: float | None, checkpoints_seen: int) -> bool:
    if current_best_objective is None:
        return checkpoints_seen >= 2
    if incumbent_objective is None:
        return False
    if checkpoints_seen < 2:
        return False
    if current_best_objective - incumbent_objective >= 0.05 and (last_window_delta is None or last_window_delta <= 0.001):
        return True
    return False
```

```python
# research/deploy_objective_autoresearch/agent_interface.py
import json
from dataclasses import dataclass


@dataclass(slots=True)
class AgentDecision:
    action: str
    rationale: str
    next_params: dict | None = None


def parse_agent_decision(payload: str) -> AgentDecision:
    raw = json.loads(payload)
    return AgentDecision(
        action=str(raw["action"]),
        rationale=str(raw.get("rationale") or ""),
        next_params=raw.get("next_params"),
    )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/research/test_deploy_objective_scheduler.py -v`
Expected: PASS

- [ ] **Step 5: 提交该任务**

```bash
git add research/deploy_objective_autoresearch/proposal_policy.py \
        research/deploy_objective_autoresearch/agent_interface.py \
        research/deploy_objective_autoresearch/scheduler.py \
        tests/research/test_deploy_objective_scheduler.py
git commit -m "feat: add deploy-objective scheduler and proposal policy"
```

### Task 5: CLI 入口、auto-resume 与 background 友好 step-loop

**Files:**
- Create: `research/deploy_objective_autoresearch/main.py`
- Modify: `research/deploy_objective_autoresearch/state_store.py`
- Modify: `research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective/README.md`
- Test: `tests/research/test_deploy_objective_config_state.py`
- Test: `tests/research/test_deploy_objective_trial_runner.py`

- [ ] **Step 1: 写失败测试，覆盖 `bootstrap` 与 `step --resume-latest`**

```python
from pathlib import Path

from research.deploy_objective_autoresearch.main import build_parser


def test_parser_supports_bootstrap_and_step_commands():
    parser = build_parser()
    args = parser.parse_args(["bootstrap", "--run-root", "tmp/run"])
    assert args.command == "bootstrap"
    args = parser.parse_args(["step", "--run-root", "tmp/run", "--resume-latest"])
    assert args.command == "step"
    assert args.resume_latest is True
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `pytest tests/research/test_deploy_objective_config_state.py tests/research/test_deploy_objective_trial_runner.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 CLI、一次一步的后台友好控制流与 README 更新**

```python
# research/deploy_objective_autoresearch/main.py
import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("deploy-objective-autoresearch")
    sub = parser.add_subparsers(dest="command", required=True)

    bootstrap = sub.add_parser("bootstrap")
    bootstrap.add_argument("--run-root", required=True)

    step = sub.add_parser("step")
    step.add_argument("--run-root", required=True)
    step.add_argument("--resume-latest", action="store_true")
    step.add_argument("--dry-run-agent", action="store_true")

    status = sub.add_parser("status")
    status.add_argument("--run-root", required=True)
    return parser
```

```markdown
# 在 `research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective/README.md` 追加

新增执行入口：
- `python -m research.deploy_objective_autoresearch.main bootstrap --run-root ...`
- `python -m research.deploy_objective_autoresearch.main step --run-root ... --resume-latest`
- `python -m research.deploy_objective_autoresearch.main status --run-root ...`

默认策略：
- `step` 会自动恢复最近一个未结束 trial
- 若当前 trial 已被 stop/finish，则创建下一条 trial
- 每次只推进一个 checkpoint 决策周期，便于外层 background/autoresearch 接管
```

- [ ] **Step 4: 跑目标测试确认通过**

Run: `pytest tests/research/test_deploy_objective_config_state.py tests/research/test_deploy_objective_trial_runner.py -v`
Expected: PASS

- [ ] **Step 5: 提交该任务**

```bash
git add research/deploy_objective_autoresearch/main.py \
        research/deploy_objective_autoresearch/state_store.py \
        research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective/README.md \
        tests/research/test_deploy_objective_config_state.py \
        tests/research/test_deploy_objective_trial_runner.py
git commit -m "feat: add deploy-objective autoresearch cli"
```

### Task 6: 集成验证

**Files:**
- Modify: `research/deploy_objective_autoresearch/*.py`
- Test: `tests/research/test_deploy_objective_*.py`

- [ ] **Step 1: 跑全部单测**

Run: `pytest tests/research/test_deploy_objective_*.py -v`
Expected: PASS

- [ ] **Step 2: 做一次 dry-run CLI 验证**

Run: `python -m research.deploy_objective_autoresearch.main bootstrap --run-root /tmp/deploy_obj_run`
Expected: 输出或落盘 `run_config.json`、`state.json`、`leaderboard.json`

Run: `python -m research.deploy_objective_autoresearch.main status --run-root /tmp/deploy_obj_run`
Expected: 返回 `idle` 状态且 `new_trial_count=0`

- [ ] **Step 3: 做一次 dry-run step 验证（不真正起训练）**

Run: `python -m research.deploy_objective_autoresearch.main step --run-root /tmp/deploy_obj_run --dry-run-agent`
Expected: 输出“将创建或继续哪个 trial”的结构化摘要；不启动真实训练进程

- [ ] **Step 4: 提交集成验证收尾**

```bash
git add research/deploy_objective_autoresearch tests/research research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective/README.md
git commit -m "test: validate deploy-objective autoresearch flow"
```

## Self-Review

- Spec coverage:
  - 独立目录/状态：Task 1
  - best-exceed objective：Task 2
  - 15M checkpoint + resume：Task 3
  - 深度优先/保守继续/动态边界：Task 4
  - background 友好 CLI + auto-resume：Task 5
  - 验证闭环：Task 6
- Placeholder scan: 已检查，未保留 TBD/TODO/“类似 Task N” 之类占位语句。
- Type consistency: `SearchConfig`、`StateStore`、`TrialSnapshot`、`AgentDecision`、`normalize_candidate_params`、`should_force_stop` 在各任务中命名保持一致。
