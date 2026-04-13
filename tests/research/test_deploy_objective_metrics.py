import json
from pathlib import Path

from research.deploy_objective_autoresearch.metrics_extractor import extract_trial_snapshot


def test_extract_trial_snapshot_uses_best_exceed_not_latest(tmp_path: Path):
    log_path = tmp_path / "train.log"
    log_path.write_text("[cost][k=80] total_ratio=0.081543x\n")
    metrics_path = tmp_path / "metrics.jsonl"
    rows = [
        {
            "type": "step",
            "step": 1,
            "total_tokens": 15_000_000,
            "layers.17.self_attn.q_proj/exceed_alpha_0.50": 0.210,
            "layers.17.self_attn.q_proj/fvu": 0.320,
        },
        {
            "type": "step",
            "step": 2,
            "total_tokens": 30_000_000,
            "layers.17.self_attn.q_proj/exceed_alpha_0.50": 0.198,
            "layers.17.self_attn.q_proj/fvu": 0.305,
        },
        {
            "type": "step",
            "step": 3,
            "total_tokens": 45_000_000,
            "layers.17.self_attn.q_proj/exceed_alpha_0.50": 0.205,
            "layers.17.self_attn.q_proj/fvu": 0.307,
        },
    ]
    metrics_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    snapshot = extract_trial_snapshot(
        log_path=log_path,
        metrics_path=metrics_path,
        hook_metric_prefix="layers.17.self_attn.q_proj",
        window_start_tokens=30_000_000,
    )

    assert snapshot.total_cost_ratio == 0.081543
    assert snapshot.latest_exceed_alpha_0_50 == 0.205
    assert snapshot.best_exceed_alpha_0_50 == 0.198
    assert snapshot.best_objective == 0.279543
    assert snapshot.tokens_seen == 45_000_000
    assert snapshot.delta_best_exceed == 0.0


def test_extract_trial_snapshot_accepts_bracketed_hookpoint_prefix(tmp_path: Path):
    log_path = tmp_path / "train.log"
    log_path.write_text("[cost][k=80] total_ratio=0.081543x\n")
    metrics_path = tmp_path / "metrics.jsonl"
    rows = [
        {
            "type": "step",
            "step": 1,
            "total_tokens": 15_000_000,
            "layers.17.self_attn.q_proj/exceed_alpha_0.50": 0.210,
            "layers.17.self_attn.q_proj/fvu": 0.320,
        },
        {
            "type": "step",
            "step": 2,
            "total_tokens": 30_000_000,
            "layers.17.self_attn.q_proj/exceed_alpha_0.50": 0.198,
            "layers.17.self_attn.q_proj/fvu": 0.305,
        },
    ]
    metrics_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    snapshot = extract_trial_snapshot(
        log_path=log_path,
        metrics_path=metrics_path,
        hook_metric_prefix="layers.[17].self_attn.q_proj",
    )

    assert snapshot.total_cost_ratio == 0.081543
    assert snapshot.latest_exceed_alpha_0_50 == 0.198
    assert snapshot.best_exceed_alpha_0_50 == 0.198
    assert snapshot.best_objective == 0.279543
