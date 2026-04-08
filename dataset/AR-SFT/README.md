# AR-SFT Dataset Builder

This folder contains a single script to build an auto-regressive SFT dataset (`q` -> `a`) from the same source subsets used by GIM-SFT.

## Output Format

Each example has two fields:

- `q`: prompt / question
- `a`: target answer

For every subset, the script performs an independent split:

- 90% `train`
- 10% `test`

Output files are saved as JSONL:

- `<output-dir>/<subset>/train.jsonl`
- `<output-dir>/<subset>/test.jsonl`

## Scripts

- `build_ar_sft.py`
- `upload_ar_sft.py`

## Usage

Build all subsets:

```bash
python dataset/AR-SFT/build_ar_sft.py
```

Build selected subsets:

```bash
python dataset/AR-SFT/build_ar_sft.py --subsets gsm8k_reasoning numina_math o1_journey
```

Custom output directory and seed:

```bash
python dataset/AR-SFT/build_ar_sft.py --output-dir data/AR-SFT --seed 42
```

You can also set worker count:

```bash
python dataset/AR-SFT/build_ar_sft.py --num-proc 16
```

## Upload to Hugging Face

Login first:

```bash
huggingface-cli login
```

Upload all generated subsets/splits to `Sculpt-AI/AR-SFT`:

```bash
python dataset/AR-SFT/upload_ar_sft.py --dataset-dir data/AR-SFT --repo-id Sculpt-AI/AR-SFT
```

Upload as private dataset:

```bash
python dataset/AR-SFT/upload_ar_sft.py --dataset-dir data/AR-SFT --repo-id Sculpt-AI/AR-SFT --private
```

## Supported Subsets

- `cnn_daily_mail`
- `gsm8k_reasoning`
- `hk_o1aw`
- `kaist_cot`
- `lima`
- `magpie_reasoning`
- `numina_math`
- `o1_journey`
- `process_bench`
- `uhgeval`

## Q/A Mapping Summary

- `cnn_daily_mail`: `article` -> `highlights`
- `gsm8k_reasoning`: `question` -> `generation`
- `hk_o1aw`: `prompt` -> `thinking` + final answer
- `kaist_cot`: `source` -> `rationale` + target answer
- `lima`: conversation turns, user turn -> assistant turn
- `magpie_reasoning`: `instruction` -> `response`
- `numina_math`: `problem` -> `solution`
- `o1_journey`: `question` -> `longCOT` + final answer
- `process_bench`: `problem` -> joined reasoning `steps` (only correct processes)
- `uhgeval`: headline/date/beginning -> remainder

## Notes

- Empty `q` or `a` rows are filtered out.
- The script uses Hugging Face Datasets and downloads source datasets from the Hub.
- Splits are reproducible with `--seed`.
