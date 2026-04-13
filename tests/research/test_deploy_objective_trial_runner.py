from pathlib import Path

from research.deploy_objective_autoresearch.config import SearchConfig
from research.deploy_objective_autoresearch.trial_runner import (
    build_trial_env,
    next_checkpoint_target,
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
