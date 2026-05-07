# TODO:
# - Modify reward funcs
# - Debug eval and saving

import os


os.environ["UNSLOTH_STABLE_DOWNLOADS"] = (
    "1"  # https://unsloth.ai/docs/basics/troubleshooting-and-faqs#downloading-gets-stuck-at-90-to-95
)
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"  # [NEW] Extra 30% context lengths!

from unsloth import FastLanguageModel  # noqa: I001

import logging
import pathlib
import random

import numpy as np
import torch

from datasets import Dataset, concatenate_datasets, load_dataset
from gimkit import guide
from gimkit.contexts import Query, infill
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams


# ─── Train Configs ────────────────────────────────────────────────────────────
# This class-type configs is easily convertible to python file configs.
# We keep the class name lowercase and use uppercase for the variables.


class configs:  # noqa: N801
    PROJECT_NAME = "GIM-RLVR"
    RUN_NAME = pathlib.Path(__file__).resolve().parent.name

    ARTIFACTS_DIR = pathlib.Path("/mnt/data/artifacts") / RUN_NAME
    FINAL_MODEL_DIR = ARTIFACTS_DIR / "rlvr-gim-model"

    RANDOM_SEED = 42

    BASE_MODEL_NAME = "Sculpt-AI/GIM-1.7B"
    MAX_SEQ_LENGTH = 8192
    QUANT_BITS = 4

    DATASET_NAME = "Sculpt-AI/GIM-SFT"
    DATASET_LEN = 10_000

    HIGH_SUBSETS_PROPORTION = 6_000
    MID_SUBSETS_PROPORTION = (DATASET_LEN - HIGH_SUBSETS_PROPORTION) // 2
    LOW_SUBSETS_PROPORTION = DATASET_LEN - HIGH_SUBSETS_PROPORTION - MID_SUBSETS_PROPORTION

    TRAIN_SPLIT = 0.98
    TRAIN_SIZE = int(DATASET_LEN * TRAIN_SPLIT)

    NUM_GPUS = torch.cuda.device_count()
    MICRO_BSZ = 2
    GRAD_ACCUM = 4
    GLOBAL_BSZ = MICRO_BSZ * GRAD_ACCUM * NUM_GPUS
    NUM_GENERATIONS = 4

    LEARNING_RATE = 5e-6

    ESTIMATED_STEPS = (TRAIN_SIZE * NUM_GENERATIONS // GLOBAL_BSZ) * 1
    WARMUP_STEPS = max(1, ESTIMATED_STEPS // 20)
    SAVE_STEPS = max(1, ESTIMATED_STEPS // 10)
    EVAL_STEPS = SAVE_STEPS
    NO_EVAL = True

    LORA_R = 32
    LORA_ALPHA = 32
    LR_SCHEDULER_TYPE = "cosine"
    WEIGHT_DECAY = 0.001

    SAMPLING_PARAM_TEMPERATURE = 1.0
    SAMPLING_PARAM_TOP_P = 1.0
    SAMPLING_PARAM_MIN_P = 0.1
    SAMPLING_PARAM_TOP_K = -1


configs.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Reward Functions ─────────────────────────────────────────────────────────


def check_format(prompts, completions, solution, **kwargs):
    scores = []
    for i in range(len(prompts)):
        query = prompts[i][-1]["content"]
        response = completions[i][-1]["content"]
        golden_truth = solution[i]  # noqa: F841
        try:
            infill(query, response, strict=True)
            scores.append(1)
        except:  # noqa: E722
            scores.append(0)
    return scores


reward_funcs = [
    check_format,
]


# ─── General Setup ────────────────────────────────────────────────────────────

os.environ["WANDB_PROJECT"] = configs.PROJECT_NAME
os.environ["WANDB_DIR"] = str(configs.ARTIFACTS_DIR)
os.environ["WANDB_LOG_MODEL"] = "end"

logging.basicConfig(
    filename=configs.ARTIFACTS_DIR / "training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

logging.info("Training configurations:")
for key in dir(configs):
    if key.isupper():
        logging.info(f"{key} = {getattr(configs, key)}")


# ─── Load Model And Tokenizer ─────────────────────────────────────────────────

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=configs.BASE_MODEL_NAME,
    max_seq_length=configs.MAX_SEQ_LENGTH,
    load_in_4bit=configs.QUANT_BITS == 4,  # False for LoRA 16bit
    fast_inference=True,  # Enable vllm fast inference
    max_lora_rank=configs.LORA_R,
    gpu_memory_utilization=0.9,  # Reduce if out of memory
    enforce_eager=True,  # important
)

model = FastLanguageModel.get_peft_model(
    model,
    r=configs.LORA_R,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=configs.LORA_ALPHA,  # *2 speeds up training
    use_gradient_checkpointing="unsloth",  # Reduces memory usage
    random_state=configs.RANDOM_SEED,
)


# ─── Load Dataset ─────────────────────────────────────────────────────────────


def _concat_subsets(subsets: list[str]) -> Dataset:
    return concatenate_datasets([load_dataset(configs.DATASET_NAME, subset, split="train") for subset in subsets])


def _build_chat_example(example: dict) -> dict:
    return {
        "prompt": [
            {"role": "user", "content": example["gim_query"]},
        ],
        "solution": example["gim_response"],
    }


# fmt: off
high_subsets = [        # 23294 in total
    "gsm8k_reasoning",  # 1254
    "hk_o1aw",          # 14363
    "lima",             # 1030
    "o1_journey",       # 327
    "process_bench",    # 1179
    "uhgeval",          # 5141
]
mid_subsets = [         # 437113 in total
    "cnn_daily_mail",   # 287113
    "magpie_reasoning", # 150000
]
low_subsets = [         # 2697422 in total
    "kaist_cot",        # 1837928
    "numina_math",      # 859494
]
# fmt: on

total_proportion = configs.HIGH_SUBSETS_PROPORTION + configs.MID_SUBSETS_PROPORTION + configs.LOW_SUBSETS_PROPORTION
num_high = int(configs.DATASET_LEN * configs.HIGH_SUBSETS_PROPORTION / total_proportion)
num_mid = int(configs.DATASET_LEN * configs.MID_SUBSETS_PROPORTION / total_proportion)
num_low = configs.DATASET_LEN - num_high - num_mid

logging.info("Loading and preparing dataset...")
rng = random.Random(configs.RANDOM_SEED)
high_dataset = _concat_subsets(high_subsets).shuffle(seed=configs.RANDOM_SEED)
mid_dataset = _concat_subsets(mid_subsets).shuffle(seed=configs.RANDOM_SEED)
low_dataset = _concat_subsets(low_subsets).shuffle(seed=configs.RANDOM_SEED)
high_dataset = high_dataset.select(rng.choices(range(len(high_dataset)), k=num_high))
mid_dataset = mid_dataset.select(rng.choices(range(len(mid_dataset)), k=num_mid))
low_dataset = low_dataset.select(rng.choices(range(len(low_dataset)), k=num_low))
dataset = concatenate_datasets([high_dataset, mid_dataset, low_dataset]).shuffle(seed=configs.RANDOM_SEED)

assert len(dataset) == configs.DATASET_LEN, f"{len(dataset)=}, {configs.DATASET_LEN=}"
logging.info(f"Number of training samples: high={len(high_dataset)}, mid={len(mid_dataset)}, low={len(low_dataset)}")

dataset = dataset.map(_build_chat_example, num_proc=os.cpu_count() or 1)
logging.info(f"Dataset sample: {dataset[0]=}")


# ─── Remove Long Samples ──────────────────────────────────────────────────────

tokenized = dataset.map(
    lambda x: {
        "prompt_tokens": [
            tokenizer.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True)
            for q in x["gim_query"]
        ],
        "prompt_completion_tokens": [
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": r},
                ],
                add_generation_prompt=False,
                tokenize=True,
            )
            for q, r in zip(x["gim_query"], x["gim_response"], strict=True)
        ],
    },
    batched=True,
)
logging.info(f"{tokenizer.decode(tokenized[0]['prompt_tokens'])=}")
logging.info(f"{tokenizer.decode(tokenized[0]['prompt_completion_tokens'])=}")
tokenized = tokenized.map(
    lambda x: {"len_prompt": len(x["prompt_tokens"]), "len_prompt_completion": len(x["prompt_completion_tokens"])}
)


# We wish prompt + completion <= max_seq_length - 256
dataset = dataset.select(np.where(np.array(tokenized["len_prompt_completion"]) <= configs.MAX_SEQ_LENGTH - 256)[0])
tokenized = tokenized.select(np.where(np.array(tokenized["len_prompt_completion"]) <= configs.MAX_SEQ_LENGTH - 256)[0])

# Get the maximum prompt length and prompt + completion length for logging
max_prompt_length = max(tokenized["len_prompt"])
max_prompt_completion_length = max(tokenized["len_prompt_completion"])
logging.info(f"Maximum prompt length: {max_prompt_length}")
logging.info(f"Maximum prompt + completion length: {max_prompt_completion_length}")

del tokenized


# ─── Training ─────────────────────────────────────────────────────────────────


vllm_sampling_params = SamplingParams(
    min_p=configs.SAMPLING_PARAM_MIN_P,
    top_p=configs.SAMPLING_PARAM_TOP_P,
    top_k=configs.SAMPLING_PARAM_TOP_K,
    seed=configs.RANDOM_SEED,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)


_max_prompt_length = max_prompt_length + 1  # + 1 just in case
_max_completion_length = configs.MAX_SEQ_LENGTH - (_max_prompt_length + 1)
logging.info(
    f"Using max_prompt_length={_max_prompt_length} and max_completion_length={_max_completion_length} for training."
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_funcs,
    train_dataset=dataset.select(range(configs.TRAIN_SIZE)),
    eval_dataset=None if configs.NO_EVAL else dataset.select(range(configs.TRAIN_SIZE, configs.DATASET_LEN)),
    args=GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=configs.SAMPLING_PARAM_TEMPERATURE,
        learning_rate=configs.LEARNING_RATE,
        weight_decay=configs.WEIGHT_DECAY,
        warmup_steps=configs.WARMUP_STEPS,
        lr_scheduler_type=configs.LR_SCHEDULER_TYPE,
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=configs.MICRO_BSZ,
        gradient_accumulation_steps=configs.GRAD_ACCUM,  # Increase to 4 for smoother training
        num_generations=configs.NUM_GENERATIONS,  # Decrease if out of memory
        max_prompt_length=_max_prompt_length,
        max_completion_length=_max_completion_length,
        num_train_epochs=1,  # Set to 1 for a full training run
        max_steps=-1,
        save_steps=configs.SAVE_STEPS,
        report_to="wandb",
        output_dir=configs.ARTIFACTS_DIR,
        run_name=configs.RUN_NAME,
        # For optional evaluation
        fp16_full_eval=True,
        per_device_eval_batch_size=configs.GLOBAL_BSZ,
        eval_accumulation_steps=None if configs.NO_EVAL else configs.GRAD_ACCUM,
        eval_strategy="no" if configs.NO_EVAL else "steps",
        eval_steps=None if configs.NO_EVAL else configs.EVAL_STEPS,
    ),
)
trainer.train()


# ─── Inference ────────────────────────────────────────────────────────────────

text = str(Query(f"This is an {guide()} text."))
logging.info("Request: " + text)

sampling_params = SamplingParams(
    temperature=1.0,
    top_k=50,
    max_tokens=1024,
)
output = (
    model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=None,
    )[0]
    .outputs[0]
    .text
)
logging.info("Response: " + output)


# ─── Save Model ───────────────────────────────────────────────────────────────

model.save_pretrained_merged(configs.FINAL_MODEL_DIR, tokenizer, save_method="merged_16bit")
