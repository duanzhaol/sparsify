from research.AutoResearch.target_profile import resolve_target_profile


def test_resolve_qwen3_4b_q_proj_profile_from_model_path():
    profile = resolve_target_profile({
        "MODEL_PATH": "/root/models/Qwen3-4B",
        "HOOKPOINTS": "layers.[17].self_attn.q_proj",
    })

    assert profile.training_hookpoint == "layers.[17].self_attn.q_proj"
    assert profile.d_in == 2560
    assert profile.n_output == 6144
    assert profile.elbow_threshold_path == "thresholds/Qwen3-4B/thresholds_q.json"


def test_resolve_qwen3_4b_up_proj_profile_from_model_path():
    profile = resolve_target_profile({
        "MODEL_PATH": "/root/models/Qwen3-4B",
        "HOOKPOINTS": "layers.[17].mlp.up_proj",
    })

    assert profile.training_hookpoint == "layers.[17].mlp.up_proj"
    assert profile.d_in == 2560
    assert profile.n_output == 19456
    assert profile.elbow_threshold_path == "thresholds/Qwen3-4B/thresholds_up.json"


def test_resolve_profile_prefers_explicit_elbow_threshold_path():
    profile = resolve_target_profile({
        "MODEL_PATH": "/root/models/Qwen3-4B",
        "HOOKPOINTS": "layers.[17].self_attn.q_proj",
        "ELBOW_THRESHOLD_PATH": "/tmp/custom_thresholds.json",
    })

    assert profile.elbow_threshold_path == "/tmp/custom_thresholds.json"


def test_resolve_profile_handles_comma_list_hookpoints_for_qwen3_4b():
    profile = resolve_target_profile({
        "MODEL_PATH": "/root/models/Qwen3-4B",
        "HOOKPOINTS": "layers.[0,5,17].self_attn.q_proj",
        "ELBOW_THRESHOLD_PATH": "/root/sparsify-ascend/thresholds/Qwen3-4B/thresholds_q.json",
    })

    assert profile.training_hookpoint == "layers.[0,5,17].self_attn.q_proj"
    assert profile.d_in == 2560
    assert profile.n_output == 6144
    assert profile.elbow_threshold_path == "/root/sparsify-ascend/thresholds/Qwen3-4B/thresholds_q.json"
