from research.deploy_objective_autoresearch.agent_interface import parse_agent_decision
from research.deploy_objective_autoresearch.config import SearchConfig
from research.deploy_objective_autoresearch.proposal_policy import (
    normalize_candidate_params,
    propose_next_params,
)
from research.deploy_objective_autoresearch.scheduler import should_force_stop


def test_normalize_candidate_params_clamps_to_dynamic_bounds():
    params = normalize_candidate_params(
        {
            "K": 81,
            "NUM_EXPERTS": 10000,
            "LATENTS_PER_EXPERT": 13,
            "ACTIVE_EXPERTS": 9,
            "LR": 0.1,
            "AUXK_ALPHA": -1,
        }
    )

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


def test_parse_agent_decision_accepts_supported_actions():
    decision = parse_agent_decision(
        {
            "action": "stop_current_and_spawn_next",
            "rationale": "plateaued",
            "next_params": {"K": 64},
        }
    )

    assert decision.action == "stop_current_and_spawn_next"
    assert decision.next_params == {"K": 64}


def test_first_candidate_uses_baseline_when_no_history(tmp_path):
    cfg = SearchConfig.default(run_root=tmp_path)
    params = propose_next_params(
        cfg,
        attempted_signatures=[],
        incumbent_trial=None,
        current_trial=None,
    )

    assert params == normalize_candidate_params(cfg.baseline_params(), cfg)
