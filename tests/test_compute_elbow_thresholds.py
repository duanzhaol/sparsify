from dataclasses import dataclass

from datasets import Dataset

from compute_elbow_thresholds import load_threshold_dataset


@dataclass
class DummyArgs:
    dataset: str
    ctx_len: int = 2048
    num_tokens: int = 4096


def test_load_threshold_dataset_supports_save_to_disk_directory(tmp_path):
    output = tmp_path / "tokenized"
    Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "overflow_to_sample_mapping": [0, 1],
        }
    ).save_to_disk(output)

    dataset, dataset_type = load_threshold_dataset(
        DummyArgs(str(output)),
        tokenizer=None,
    )

    assert dataset_type == "hf_tokenized"
    assert len(dataset) == 2
    assert dataset.column_names == ["input_ids", "overflow_to_sample_mapping"]
