# GIM Research

Research code for **Generative Instruction Masking (GIM)** — a training approach where parts of model responses (thinking, reasoning, reflection, etc.) are masked during supervised fine-tuning.

## Structure

- `dataset/` — scripts to prepare and upload GIM-format SFT datasets from various sources (GSM8K, LIMA, Numina Math, etc.)
- `runs/` — training run configs and scripts
- `ui/` — simple HTML data viewer

## Setup

```bash
make install
```

## Usage

**Lint:**
```bash
make lint
```

**Serve a model with vLLM:**
```bash
make serve model_path=/path/to/model
```
