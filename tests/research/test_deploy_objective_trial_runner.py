from pathlib import Path

import json

from research.deploy_objective_autoresearch.config import SearchConfig
from research.deploy_objective_autoresearch.state_store import TrialRecord
from research.deploy_objective_autoresearch.trial_runner import (
    TrialRunResult,
    build_trial_env,
    next_checkpoint_target,
    update_trial_from_result,
)


PARAMS = {
    "K": 80,
    "NUM_EXPERTS": 326,
    "LATENTS_PER_EXPERT": 96,
    "ACTIVE_EXPERTS": 2,
    "LR": "8e-4",
    "AUXK_ALPHA": "0.03125",
}


def test_build_trial_env_for_new_trial_sets_first_15m_checkpoint(tmp_path: Path):
    cfg = SearchConfig.default(run_root=tmp_path)
    env = build_trial_env(
        cfg=cfg,
        params=PARAMS,
        run_name="trial_0001",
        save_dir=tmp_path / "ckpts",
        target_tokens=15_000_000,
        resume=False,
    )

    assert env["MAX_TOKENS"] == "15000000"
    assert env["RESUME"] == "0"
    assert env["HOOKPOINTS"] == "layers.[17].self_attn.q_proj"
    assert env["SAVE_DIR"] == str(tmp_path / "ckpts")


def test_build_trial_env_for_resume_sets_resume_flag(tmp_path: Path):
    cfg = SearchConfig.default(run_root=tmp_path)
    env = build_trial_env(
        cfg=cfg,
        params=PARAMS,
        run_name="trial_0001",
        save_dir=tmp_path / "ckpts",
        target_tokens=30_000_000,
        resume=True,
    )

    assert env["MAX_TOKENS"] == "30000000"
    assert env["RESUME"] == "1"


def test_next_checkpoint_target_advances_in_15m_windows():
    assert next_checkpoint_target(0, 15_000_000) == 15_000_000
    assert next_checkpoint_target(15_000_000, 15_000_000) == 30_000_000
    assert next_checkpoint_target(29_999_999, 15_000_000) == 30_000_000


def test_update_trial_from_result_advances_tokens_from_summary_when_metrics_lag(
    tmp_path: Path,
):
    cfg = SearchConfig.default(run_root=tmp_path)
    ckpt_dir = tmp_path / "trial_ckpt"
    ckpt_dir.mkdir()
    log_path = tmp_path / "train.log"
    log_path.write_text("[cost][k=80] total_ratio=0.081543x\n")
    metrics_path = ckpt_dir / "metrics.jsonl"
    metrics_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {
                    "type": "step",
                    "step": 439,
                    "total_tokens": 14_385_152,
                    "layers.17.self_attn.q_proj/exceed_alpha_0.50": 0.49907049536705017,
                    "layers.17.self_attn.q_proj/fvu": 0.44677263498306274,
                },
                {
                    "type": "step",
                    "step": 449,
                    "total_tokens": 14_712_832,
                    "layers.17.self_attn.q_proj/exceed_alpha_0.50": 0.5008922815322876,
                    "layers.17.self_attn.q_proj/fvu": 0.4535769820213318,
                },
            ]
        )
    )
    (ckpt_dir / "summary.json").write_text(
        json.dumps(
            {
                "total_steps": 458,
                "total_tokens": 15_011_840,
                "final_fvu": {"layers.17.self_attn.q_proj": 0.0056},
                "best_fvu": {"layers.17.self_attn.q_proj": 0.44677263498306274},
            }
        )
    )

    trial = TrialRecord(
        trial_id="trial_0001",
        params={
            "K": 80,
            "NUM_EXPERTS": 326,
            "LATENTS_PER_EXPERT": 96,
            "ACTIVE_EXPERTS": 2,
            "LR": 0.0008,
            "AUXK_ALPHA": 0.03125,
        },
        status="running",
        created_at="2026-04-14T00:00:00+00:00",
        updated_at="2026-04-14T00:00:00+00:00",
        save_dir=str(tmp_path / "save_dir"),
        log_path=str(log_path),
        tokens_seen=14_712_832,
    )
    result = TrialRunResult(
        returncode=0,
        target_tokens=15_000_000,
        checkpoint_root=str(ckpt_dir),
        metrics_path=str(metrics_path),
        log_path=str(log_path),
        tokens_seen=14_712_832,
    )

    updated = update_trial_from_result(cfg, trial, result)

    assert updated.tokens_seen == 15_011_840
    assert updated.checkpoint_decisions == 1
    assert updated.status == "checkpoint_ready"
