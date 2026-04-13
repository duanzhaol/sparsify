from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_RUN_ROOT = Path(
    "research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective"
)
DEFAULT_CHECKPOINT_ROOT = Path(
    "checkpoints/qwen3-4B/deploy_objective_autoresearch"
)


@dataclass(slots=True)
class DynamicBounds:
    k_min: int = 16
    k_max: int = 128
    k_step: int = 8
    num_experts_min: int = 64
    num_experts_max: int = 1024
    num_experts_step: int = 2
    latents_per_expert_min: int = 16
    latents_per_expert_max: int = 256
    latents_per_expert_step: int = 8
    active_experts_min: int = 1
    active_experts_max: int = 4
    lr_min: float = 1e-4
    lr_max: float = 2e-3
    auxk_alpha_min: float = 0.0
    auxk_alpha_max: float = 0.125


@dataclass(slots=True)
class SearchConfig:
    run_root: Path
    checkpoint_root: Path
    model_path: str
    dataset_path: str
    elbow_threshold_path: str
    architecture: str
    hookpoints: str
    wandb_project: str
    checkpoint_interval_tokens: int = 15_000_000
    max_new_trials: int = 200
    nproc_per_node: int = 2
    max_active_trials: int = 1
    expansion_factor: int = 1
    optimizer: str = "adam"
    batch_size: int = 1
    grad_acc_steps: int = 8
    micro_acc_steps: int = 1
    dead_feature_threshold: int = 10_000_000
    use_hadamard: int = 0
    compile_model: int = 1
    print_cost_breakdown: int = 1
    poll_interval_sec: int = 30
    stall_timeout_sec: int = 15 * 60
    finish_grace_sec: int = 5 * 60
    bounds: DynamicBounds = field(default_factory=DynamicBounds)

    @classmethod
    def default(cls, run_root: Path | None = None) -> "SearchConfig":
        home = Path.home()
        root = Path(run_root) if run_root is not None else DEFAULT_RUN_ROOT
        return cls(
            run_root=root,
            checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
            model_path=str(home / "models" / "Qwen3-4B"),
            dataset_path=str(
                home / "fineweb-edu" / "sample" / "10BT-tokenized-qwen3-2048"
            ),
            elbow_threshold_path=str(
                Path.cwd() / "thresholds" / "Qwen3-4B" / "thresholds_q.json"
            ),
            architecture="product_key_expert_jumprelu",
            hookpoints="layers.[17].self_attn.q_proj",
            wandb_project="qwen3-4B-product_key_expert_jumprelu-qproj",
        )

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "SearchConfig":
        data = dict(payload)
        bounds_payload = data.pop("bounds", None) or {}
        data["run_root"] = Path(data["run_root"])
        data["checkpoint_root"] = Path(data["checkpoint_root"])
        data["bounds"] = DynamicBounds(**bounds_payload)
        return cls(**data)

    def baseline_params(self) -> dict[str, int | float]:
        return {
            "K": 80,
            "NUM_EXPERTS": 326,
            "LATENTS_PER_EXPERT": 96,
            "ACTIVE_EXPERTS": 2,
            "LR": 8e-4,
            "AUXK_ALPHA": 0.03125,
        }

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_root"] = str(self.run_root)
        payload["checkpoint_root"] = str(self.checkpoint_root)
        return payload

    @property
    def run_config_path(self) -> Path:
        return self.run_root / "run_config.json"

    @property
    def state_path(self) -> Path:
        return self.run_root / "state.json"

    @property
    def leaderboard_path(self) -> Path:
        return self.run_root / "leaderboard.json"

    @property
    def trials_jsonl_path(self) -> Path:
        return self.run_root / "trials.jsonl"

    @property
    def agent_decisions_path(self) -> Path:
        return self.run_root / "agent_decisions.jsonl"

    @property
    def pending_agent_decision_path(self) -> Path:
        return self.run_root / "pending_agent_decision.json"

    @property
    def current_agent_context_path(self) -> Path:
        return self.run_root / "current_agent_context.json"

    @property
    def trials_dir(self) -> Path:
        return self.run_root / "trials"
