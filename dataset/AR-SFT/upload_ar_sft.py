import argparse

from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Upload local AR-SFT jsonl files to Hugging Face Hub.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/AR-SFT",
        help="Local AR-SFT directory: <dataset-dir>/<subset>/{train,test}.jsonl",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Sculpt-AI/AR-SFT",
        help="Target HF dataset repo id.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update dataset repo as private.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    ds_paths = sorted(Path.glob(dataset_dir, "*/*.jsonl"))
    if not ds_paths:
        raise FileNotFoundError(f"No jsonl files found under: {dataset_dir}")

    for ds_path in ds_paths:
        subset_name = ds_path.parent.stem
        split_name = ds_path.stem
        print(f"Uploading subset={subset_name}, split={split_name}, path={ds_path}")
        ds = load_dataset("json", data_files={split_name: ds_path.as_posix()})
        ds.push_to_hub(args.repo_id, subset_name, private=args.private)


if __name__ == "__main__":
    main()
