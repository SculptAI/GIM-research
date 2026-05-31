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
from gimkit.schemas import validate
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
    MAX_SEQ_LENGTH = 2048
    QUANT_BITS = 4

    DATASET_NAME = "Sculpt-AI/GIM-SFT"
    DATASET_LEN = 5_000
    RESULTS_VERIFIABLE_SUBSETS = 2_000
    HIGH_SUBSETS = 2_000
    MID_SUBSETS = 500
    LOW_SUBSETS = DATASET_LEN - RESULTS_VERIFIABLE_SUBSETS - HIGH_SUBSETS - MID_SUBSETS

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
    LR_SCHEDULER_TYPE = "constant_with_warmup"
    WEIGHT_DECAY = 0.001

    SAMPLING_PARAM_TEMPERATURE = 1.0
    SAMPLING_PARAM_TOP_P = 1.0
    SAMPLING_PARAM_MIN_P = 0.1
    SAMPLING_PARAM_TOP_K = -1


configs.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Reward Functions ─────────────────────────────────────────────────────────


def format_reward(prompts, completions, **kwargs):
    scores = []
    for i in range(len(prompts)):
        query = prompts[i][-1]["content"]
        response = completions[i][-1]["content"]
        try:
            validate(query, response)
            scores.append(2)
            continue
        except:  # noqa: E722
            pass
        try:
            infill(query, response, strict=False)
            scores.append(1)
            continue
        except:  # noqa: E722
            pass
        scores.append(-1)
    return scores


reward_debug_counter = 0

def correctness_length_reward(prompts, completions, solution, **kwargs):
    global reward_debug_counter

    scores = []

    batch_acc = []
    batch_ratio = []
    batch_reward = []
    batch_correctness_reward = []
    batch_length_factor = []
    batch_perfect_match = []

    exception_count = 0

    sample_query = None
    sample_response = None
    sample_golden = None
    sample_acc = None
    sample_ratio = None
    sample_reward = None

    for i in range(len(prompts)):
        query = prompts[i][-1]["content"]
        response = completions[i][-1]["content"]
        golden_truth = solution[i]

        try:
            pred_result = infill(query, response)
            real_result = infill(query, golden_truth)

            # ----- correctness -----
            total_tags = len(real_result.tags)

            correct_tags = sum(
                pred_tag == real_tag
                for pred_tag, real_tag in zip(
                    pred_result.tags,
                    real_result.tags,
                    strict=True,
                )
            )

            acc = correct_tags / max(total_tags, 1)

            # map to [-1, 1]
            correctness_reward = 2 * acc - 1

            # ----- length regularization -----
            response_len = len(response)
            golden_len = len(golden_truth)

            ratio = response_len / max(golden_len, 1)

            # no penalty in reasonable range
            if ratio <= 1.2:
                length_factor = 1.0

            # soft penalty for verbosity
            elif ratio <= 2.0:
                length_factor = 1.0 - 0.3 * (ratio - 1.2) / 0.8

            # stronger penalty for runaway reasoning
            else:
                length_factor = 0.7 * np.exp(-(ratio - 2.0))

            # ----- combine -----
            reward = correctness_reward * length_factor

            # perfect prediction bonus
            if acc == 1.0:
                reward += 0.5

            reward = float(reward)

            scores.append(reward)

            # ----- stats -----
            batch_acc.append(acc)
            batch_ratio.append(ratio)
            batch_reward.append(reward)
            batch_correctness_reward.append(correctness_reward)
            batch_length_factor.append(length_factor)
            batch_perfect_match.append(acc == 1.0)

            # save first sample for debug
            if sample_query is None:
                sample_query = query
                sample_response = response
                sample_golden = golden_truth
                sample_acc = acc
                sample_ratio = ratio
                sample_reward = reward

        except Exception:
            exception_count += 1
            scores.append(-1.0)

    reward_debug_counter += 1

    # ===== aggregate stats =====
    if reward_debug_counter % 10 == 0:
        logging.info(
            "[RewardStats] "
            f"calls={reward_debug_counter} "
            f"acc={np.mean(batch_acc):.4f} "
            f"perfect={np.mean(batch_perfect_match):.4f} "
            f"ratio={np.mean(batch_ratio):.4f} "
            f"corr_reward={np.mean(batch_correctness_reward):.4f} "
            f"len_factor={np.mean(batch_length_factor):.4f} "
            f"final_reward={np.mean(batch_reward):.4f} "
            f"exceptions={exception_count}/{len(prompts)}"
        )

    # ===== sample dump =====
    if reward_debug_counter % 20 == 0 and sample_query is not None:
        logging.info(
            "\n"
            "================ REWARD SAMPLE ================\n"
            f"Acc: {sample_acc:.4f}\n"
            f"Length Ratio: {sample_ratio:.4f}\n"
            f"Reward: {sample_reward:.4f}\n\n"
            f"Query:\n{sample_query[:1000]}\n\n"
            f"Prediction:\n{sample_response[:2000]}\n\n"
            f"Golden:\n{sample_golden[:2000]}\n"
            "================================================"
        )

    return scores


reward_funcs = [
    format_reward,
    correctness_length_reward,
]


# ─── General Setup ────────────────────────────────────────────────────────────

os.environ["WANDB_PROJECT"] = configs.PROJECT_NAME
os.environ["WANDB_DIR"] = str(configs.ARTIFACTS_DIR)
os.environ["WANDB_LOG_MODEL"] = "end"

train_log_path = configs.ARTIFACTS_DIR / "training.log"
logging.basicConfig(
    filename=train_log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
print(f"Training logs will be saved to {train_log_path}")

logging.info("Training configurations:")
for key, value in configs.__dict__.items():
    if key.isupper():
        logging.info(f"{key} = {value}")


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


def _process_subsets(subsets: list[str], num_samples: int, random_seed: int) -> Dataset:
    """Load and concatenate specified subsets, then sample num_samples examples with replacement if needed."""
    ds: Dataset = concatenate_datasets(
        [load_dataset(configs.DATASET_NAME, subset, split="train") for subset in subsets]
    )
    if num_samples <= len(ds):
        ds = ds.select(random.sample(range(len(ds)), k=num_samples))
    else:
        ds = ds.select(random.choices(range(len(ds)), k=num_samples))
    return ds.shuffle(seed=random_seed)


def _build_chat_example(example: dict) -> dict:
    return {
        "prompt": [
            {"role": "user", "content": example["gim_query"]},
        ],
        "solution": example["gim_response"],
    }


# fmt: off
results_verifiable_subsets = [
    "gsm8k_reasoning",  # 1254
    "o1_journey",       # 327
    "o1_journey",       # 327
    "o1_journey",       # 327
    # Repeating o1_journey to make the dataset more balanced
]
high_subsets = [
    "hk_o1aw",          # 14363
    "lima",             # 1030
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

logging.info("Loading and preparing dataset...")

verifiable_dataset = _process_subsets(
    results_verifiable_subsets, configs.RESULTS_VERIFIABLE_SUBSETS, configs.RANDOM_SEED
)
high_dataset = _process_subsets(high_subsets, configs.HIGH_SUBSETS, configs.RANDOM_SEED)
mid_dataset = _process_subsets(mid_subsets, configs.MID_SUBSETS, configs.RANDOM_SEED)
low_dataset = _process_subsets(low_subsets, configs.LOW_SUBSETS, configs.RANDOM_SEED)
dataset = (
    concatenate_datasets([verifiable_dataset, high_dataset, mid_dataset, low_dataset])
    .shuffle(seed=configs.RANDOM_SEED)
    .map(_build_chat_example, num_proc=os.cpu_count() or 1)
)

logging.info(f"Dataset loaded and prepared. Total length: {len(dataset)}")
logging.info(
    f"Number of training samples: verifiable={len(verifiable_dataset)}, high={len(high_dataset)}, mid={len(mid_dataset)}, low={len(low_dataset)}"
)
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

condition = (np.array(tokenized["len_prompt"]) <= configs.MAX_SEQ_LENGTH * 2 // 3) & (
    np.array(tokenized["len_prompt_completion"]) <= configs.MAX_SEQ_LENGTH - 256
)
dataset = dataset.select(np.where(condition)[0])
tokenized = tokenized.select(np.where(condition)[0])
logging.info(f"Dataset length after removing long samples: {len(dataset)}")

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
