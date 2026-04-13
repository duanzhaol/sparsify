import json
from pathlib import Path

import research.deploy_objective_autoresearch.main as deploy_main
from research.deploy_objective_autoresearch.config import SearchConfig
from research.deploy_objective_autoresearch.main import build_parser, main
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


def test_parser_supports_bootstrap_and_step_commands():
    parser = build_parser()

    args = parser.parse_args(["bootstrap", "--run-root", "tmp/run"])
    assert args.command == "bootstrap"

    args = parser.parse_args(["step", "--run-root", "tmp/run", "--resume-latest"])
    assert args.command == "step"
    assert args.resume_latest is True

    args = parser.parse_args(["metric", "--run-root", "tmp/run"])
    assert args.command == "metric"

    args = parser.parse_args(
        [
            "write-decision",
            "--run-root",
            "tmp/run",
            "--action",
            "continue_current",
            "--rationale",
            "keep going",
        ]
    )
    assert args.command == "write-decision"


def test_metric_command_prints_incumbent_best_objective(tmp_path: Path, capsys):
    cfg = SearchConfig.default(run_root=tmp_path)
    store = StateStore.bootstrap(cfg)
    trial = store.create_trial(cfg, cfg.baseline_params())
    trial.best_objective = 0.2789
    trial.best_exceed_alpha_0_50 = 0.198
    trial.total_cost_ratio = 0.0809
    trial.status = "checkpoint_ready"
    store.save_trial(trial)

    exit_code = main(["metric", "--run-root", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == "0.278900000000"


def test_write_decision_normalizes_and_persists_pending_decision(
    tmp_path: Path,
    capsys,
):
    cfg = SearchConfig.default(run_root=tmp_path)
    store = StateStore.bootstrap(cfg)
    trial = store.create_trial(cfg, cfg.baseline_params())
    trial.params["LATENTS_PER_EXPERT"] = 96
    store.save_trial(trial)

    exit_code = main(
        [
            "write-decision",
            "--run-root",
            str(tmp_path),
            "--action",
            "stop_current_and_spawn_next",
            "--rationale",
            "objective plateaued",
            "--next-params-json",
            json.dumps({"K": 81, "NUM_EXPERTS": 9999, "LATENTS_PER_EXPERT": 15}),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    pending = json.loads(cfg.pending_agent_decision_path.read_text())

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert pending["action"] == "stop_current_and_spawn_next"
    assert pending["next_params"]["K"] % 8 == 0
    assert pending["next_params"]["NUM_EXPERTS"] <= 1024
    assert pending["next_params"]["LATENTS_PER_EXPERT"] == 16
    assert pending["next_params"]["ACTIVE_EXPERTS"] == 2


def test_metric_command_can_bootstrap_if_missing(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    cfg = SearchConfig.default(run_root=tmp_path)
    store = StateStore.bootstrap(cfg)

    def fake_execute_step(args, *, emit_payload):
        trial = store.create_trial(cfg, cfg.baseline_params())
        trial.best_objective = 0.271
        trial.best_exceed_alpha_0_50 = 0.191
        trial.total_cost_ratio = 0.08
        trial.status = "checkpoint_ready"
        store.save_trial(trial)
        return {"command": "step"}

    monkeypatch.setattr(deploy_main, "_execute_step", fake_execute_step)

    exit_code = main(
        [
            "metric",
            "--run-root",
            str(tmp_path),
            "--bootstrap-if-missing",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == "0.271000000000"


def test_pending_spawn_params_are_consumed_when_creating_next_trial(tmp_path: Path):
    cfg = SearchConfig.default(run_root=tmp_path)
    store = StateStore.bootstrap(cfg)
    state = store.load_state()
    state["pending_spawn_params"] = {
        "K": 96,
        "NUM_EXPERTS": 384,
        "LATENTS_PER_EXPERT": 88,
        "ACTIVE_EXPERTS": 2,
        "LR": 8e-4,
        "AUXK_ALPHA": 0.03125,
    }
    store.save_state(state)

    trial = deploy_main._maybe_create_next_trial(store, cfg)
    state = store.load_state()

    assert trial is not None
    assert trial.params["K"] == 96
    assert trial.params["NUM_EXPERTS"] == 384
    assert "pending_spawn_params" not in state
