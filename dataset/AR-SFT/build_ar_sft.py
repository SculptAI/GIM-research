import argparse
import os
from pathlib import Path

from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import list_repo_files


Q_COLUMN = "q"
A_COLUMN = "a"
OUTPUT_COLUMNS = [Q_COLUMN, A_COLUMN]


def _to_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _load_dataset_with_parquet_fallback(dataset_id: str, split: str, num_proc: int, **kwargs) -> Dataset:
    """Load HF dataset with a parquet fallback for metadata incompatibilities.

    Some repos may contain dataset metadata with feature types unsupported by the
    currently installed `datasets` version (e.g., `List`). In this case we load
    parquet files directly from the dataset repository.
    """

    try:
        return load_dataset(dataset_id, split=split, num_proc=num_proc, **kwargs)
    except ValueError as exc:
        if "Feature type 'List' not found" not in str(exc):
            raise

        repo_files = list_repo_files(dataset_id, repo_type="dataset")
        parquet_files = [
            f for f in repo_files if f.endswith(".parquet") and (f"/{split}-" in f or f.startswith(f"{split}-"))
        ]
        if not parquet_files:
            raise RuntimeError(
                f"Failed to find parquet files for split '{split}' in dataset '{dataset_id}'"
            ) from exc

        data_files = {split: [f"hf://datasets/{dataset_id}/{path}" for path in sorted(parquet_files)]}
        return load_dataset("parquet", data_files=data_files, split=split, num_proc=num_proc)


def _build_cnn_daily_mail(num_proc: int) -> Dataset:
    ds = load_dataset("abisee/cnn_dailymail", name="3.0.0", split="train", num_proc=num_proc)

    def _convert(example: dict) -> dict:
        return {
            Q_COLUMN: _to_text(example["article"]),
            A_COLUMN: _to_text(example["highlights"]),
        }

    return ds.map(_convert, num_proc=num_proc).filter(
        lambda x: bool(x[Q_COLUMN]) and bool(x[A_COLUMN]), num_proc=num_proc
    ).select_columns(OUTPUT_COLUMNS)


def _build_gsm8k_reasoning(num_proc: int) -> Dataset:
    ds = _load_dataset_with_parquet_fallback("thesven/gsm8k-reasoning", split="train", num_proc=num_proc)

    def _convert(example: dict) -> dict:
        return {
            Q_COLUMN: _to_text(example["question"]),
            A_COLUMN: _to_text(example["generation"]),
        }

    return ds.map(_convert, num_proc=num_proc).filter(
        lambda x: bool(x[Q_COLUMN]) and bool(x[A_COLUMN]), num_proc=num_proc
    ).select_columns(OUTPUT_COLUMNS)


def _build_hk_o1aw(num_proc: int) -> Dataset:
    ds = load_dataset("HKAIR-Lab/HK-O1aw-SFT-16K", split="train", num_proc=num_proc)

    def _convert(example: dict) -> dict:
        thinking = _to_text(example.get("thinking"))
        answer = _to_text(example.get("answer"))
        if thinking and answer:
            a_value = f"{thinking}\n\nFinal Answer:\n{answer}"
        else:
            a_value = thinking or answer
        return {
            Q_COLUMN: _to_text(example.get("prompt")),
            A_COLUMN: a_value,
        }

    return ds.map(_convert, num_proc=num_proc).filter(
        lambda x: bool(x[Q_COLUMN]) and bool(x[A_COLUMN]), num_proc=num_proc
    ).select_columns(OUTPUT_COLUMNS)


def _build_kaist_cot(num_proc: int) -> Dataset:
    ds = load_dataset(
        "kaist-ai/CoT-Collection",
        split="train",
        trust_remote_code=True,
        num_proc=num_proc,
    )

    def _convert(example: dict) -> dict:
        rationale = _to_text(example.get("rationale"))
        target = _to_text(example.get("target"))
        if rationale and target:
            a_value = f"{rationale}\n\nFinal Answer:\n{target}"
        else:
            a_value = rationale or target
        return {
            Q_COLUMN: _to_text(example.get("source")),
            A_COLUMN: a_value,
        }

    return ds.map(_convert, num_proc=num_proc).filter(
        lambda x: bool(x[Q_COLUMN]) and bool(x[A_COLUMN]), num_proc=num_proc
    ).select_columns(OUTPUT_COLUMNS)


def _build_lima(_: int) -> Dataset:
    ds = load_dataset("Ki-Seki/GAIR_lima", split="train")
    rows = []
    for example in ds:
        conversations = example.get("conversations") or []
        conv_len = len(conversations) - (len(conversations) % 2)
        for i in range(0, conv_len, 2):
            q_value = _to_text(conversations[i])
            a_value = _to_text(conversations[i + 1])
            if q_value and a_value:
                rows.append({Q_COLUMN: q_value, A_COLUMN: a_value})
    return Dataset.from_list(rows)


def _build_magpie_reasoning(num_proc: int) -> Dataset:
    ds = load_dataset("Magpie-Align/Magpie-Reasoning-150K", split="train", num_proc=num_proc)

    def _convert(example: dict) -> dict:
        return {
            Q_COLUMN: _to_text(example.get("instruction")),
            A_COLUMN: _to_text(example.get("response")),
        }

    return ds.map(_convert, num_proc=num_proc).filter(
        lambda x: bool(x[Q_COLUMN]) and bool(x[A_COLUMN]), num_proc=num_proc
    ).select_columns(OUTPUT_COLUMNS)


def _build_numina_math(num_proc: int) -> Dataset:
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", num_proc=num_proc)

    def _convert(example: dict) -> dict:
        return {
            Q_COLUMN: _to_text(example.get("problem")),
            A_COLUMN: _to_text(example.get("solution")),
        }

    return ds.map(_convert, num_proc=num_proc).filter(
        lambda x: bool(x[Q_COLUMN]) and bool(x[A_COLUMN]), num_proc=num_proc
    ).select_columns(OUTPUT_COLUMNS)


def _build_o1_journey(num_proc: int) -> Dataset:
    ds = load_dataset("GAIR/o1-journey", split="train", num_proc=num_proc)

    def _convert(example: dict) -> dict:
        long_cot = _to_text(example.get("longCOT"))
        if "####" in long_cot:
            long_cot = long_cot.split("####", maxsplit=1)[0].strip()
        answer = _to_text(example.get("answer"))
        if long_cot and answer:
            a_value = f"{long_cot}\n\nFinal Answer:\n{answer}"
        else:
            a_value = long_cot or answer
        return {
            Q_COLUMN: _to_text(example.get("question")),
            A_COLUMN: a_value,
        }

    return ds.map(_convert, num_proc=num_proc).filter(
        lambda x: bool(x[Q_COLUMN]) and bool(x[A_COLUMN]), num_proc=num_proc
    ).select_columns(OUTPUT_COLUMNS)


def _build_process_bench(num_proc: int) -> Dataset:
    ds_dict = load_dataset("Qwen/ProcessBench", num_proc=num_proc)
    ds = concatenate_datasets([ds_dict[split_name] for split_name in ds_dict.keys()])

    # Keep only high-quality correct processes, same rule as GIM-SFT script.
    ds = ds.filter(lambda x: x["label"] == -1 and bool(x["final_answer_correct"]), num_proc=num_proc)

    def _convert(example: dict) -> dict:
        steps = example.get("steps") or []
        a_value = "\n".join([_to_text(step) for step in steps if _to_text(step)])
        return {
            Q_COLUMN: _to_text(example.get("problem")),
            A_COLUMN: a_value,
        }

    return ds.map(_convert, num_proc=num_proc).filter(
        lambda x: bool(x[Q_COLUMN]) and bool(x[A_COLUMN]), num_proc=num_proc
    ).select_columns(OUTPUT_COLUMNS)


def _build_uhgeval(num_proc: int) -> Dataset:
    ds = load_dataset("Ki-Seki/UHGEvalDataset", name="full", split="validation", num_proc=num_proc)

    def _convert(example: dict) -> dict:
        q_value = "\n".join(
            [
                _to_text(example.get("headLine")),
                _to_text(example.get("broadcastDate")),
                _to_text(example.get("newsBeginning")),
            ]
        ).strip()
        return {
            Q_COLUMN: q_value,
            A_COLUMN: _to_text(example.get("newsRemainder")),
        }

    return ds.map(_convert, num_proc=num_proc).filter(
        lambda x: bool(x[Q_COLUMN]) and bool(x[A_COLUMN]), num_proc=num_proc
    ).select_columns(OUTPUT_COLUMNS)


BUILDERS = {
    "cnn_daily_mail": _build_cnn_daily_mail,
    "gsm8k_reasoning": _build_gsm8k_reasoning,
    "hk_o1aw": _build_hk_o1aw,
    "kaist_cot": _build_kaist_cot,
    "lima": _build_lima,
    "magpie_reasoning": _build_magpie_reasoning,
    "numina_math": _build_numina_math,
    "o1_journey": _build_o1_journey,
    "process_bench": _build_process_bench,
    "uhgeval": _build_uhgeval,
}


def _save_subset(ds: Dataset, subset_name: str, output_root: Path, seed: int):
    split = ds.train_test_split(test_size=0.1, shuffle=True, seed=seed)
    subset_dir = output_root / subset_name
    subset_dir.mkdir(parents=True, exist_ok=True)
    split["train"].to_json((subset_dir / "train.jsonl").as_posix(), force_ascii=False)
    split["test"].to_json((subset_dir / "test.jsonl").as_posix(), force_ascii=False)
    print(f"[{subset_name}] train={len(split['train'])}, test={len(split['test'])}")


def main():
    parser = argparse.ArgumentParser(description="Build Sculpt-AI/AR-SFT dataset from source subsets.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/AR-SFT",
        help="Output root directory. Each subset is saved under <output-dir>/<subset>/{train,test}.jsonl",
    )
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=list(BUILDERS.keys()),
        choices=list(BUILDERS.keys()),
        help="Subset names to build. Default: build all.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used by train/test split.")
    parser.add_argument("--num-proc", type=int, default=os.cpu_count(), help="Workers for datasets.map/filter.")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for subset in args.subsets:
        print(f"Building subset: {subset}")
        ds = BUILDERS[subset](args.num_proc)
        _save_subset(ds, subset, output_root, args.seed)


if __name__ == "__main__":
    main()