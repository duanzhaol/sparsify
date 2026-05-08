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


def test_resolve_qwen3_5_4b_gdn_qkvz_profile_from_model_path():
    profile = resolve_target_profile({
        "MODEL_PATH": "/root/models/Qwen3.5-4B",
        "HOOKPOINTS": "layers.[0].linear_attn.in_proj_qkv",
    })

    assert profile.training_hookpoint == "layers.[0].linear_attn.in_proj_qkv"
    assert profile.d_in == 2560
    assert profile.n_output == 12288
    assert profile.elbow_threshold_path == "thresholds/Qwen3.5-4B/thresholds_gdn_qkvz.json"


def test_resolve_qwen3_5_4b_q_gatekv_profile_from_model_path():
    profile = resolve_target_profile({
        "MODEL_PATH": "/root/models/Qwen3.5-4B",
        "HOOKPOINTS": "layers.[3].self_attn.q_proj",
    })

    assert profile.training_hookpoint == "layers.[3].self_attn.q_proj"
    assert profile.d_in == 2560
    assert profile.n_output == 10240
    assert profile.elbow_threshold_path == "thresholds/Qwen3.5-4B/thresholds_q_gatekv.json"


def test_resolve_qwen3_5_4b_gate_up_profile_from_model_path():
    profile = resolve_target_profile({
        "MODEL_PATH": "/root/models/Qwen3.5-4B",
        "HOOKPOINTS": "layers.[0].mlp.up_proj",
    })

    assert profile.training_hookpoint == "layers.[0].mlp.up_proj"
    assert profile.d_in == 2560
    assert profile.n_output == 18432
    assert profile.elbow_threshold_path == "thresholds/Qwen3.5-4B/thresholds_gate_up.json"
