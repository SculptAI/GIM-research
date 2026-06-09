# ruff: noqa: PGH004
# ruff: noqa

import os


os.environ["UNSLOTH_STABLE_DOWNLOADS"] = (
    "1"  # https://unsloth.ai/docs/basics/troubleshooting-and-faqs#downloading-gets-stuck-at-90-to-95
)

from unsloth import FastModel  # noqa: I001

import logging
import random

import configs

from datasets import Dataset, concatenate_datasets, load_dataset
from gimkit import guide
from gimkit.contexts import Query
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import get_chat_template, train_on_responses_only


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
for key, value in vars(configs).items():
    if not key.startswith("__"):
        logging.info(f"{key} = {value}")


# ─── Load Model And Tokenizer ─────────────────────────────────────────────────

model, tokenizer = FastModel.from_pretrained(
    model_name=configs.BASE_MODEL_NAME,
    max_seq_length=configs.MAX_SEQ_LENGTH,
    load_in_4bit=configs.QUANT_BITS == 4,
    load_in_8bit=configs.QUANT_BITS == 8,
    full_finetuning=False,
    token=None,
)

model = FastModel.get_peft_model(
    model,
    r=32,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=configs.RANDOM_SEED,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen3-instruct",
)


# ─── Load Dataset ─────────────────────────────────────────────────────────────


def _concat_subsets(subsets: list[str]) -> Dataset:
    return concatenate_datasets([load_dataset(configs.DATASET_NAME, subset, split="train") for subset in subsets])


def _build_chat_example(example: dict) -> dict:
    return {
        "text": tokenizer.apply_chat_template(
            [
                {"role": "user", "content": example["gim_query"]},
                {"role": "assistant", "content": example["gim_response"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
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


def is_long(example):
    return len(example["gim_query"]) > configs.LONG_SAMPLE_CONDITION


def is_short(example):
    return len(example["gim_query"]) <= configs.LONG_SAMPLE_CONDITION


logging.info("Loading raw subsets...")
raw_high = _concat_subsets(high_subsets).shuffle(seed=configs.RANDOM_SEED)
raw_mid = _concat_subsets(mid_subsets).shuffle(seed=configs.RANDOM_SEED)
raw_low = _concat_subsets(low_subsets).shuffle(seed=configs.RANDOM_SEED)

logging.info("Filtering into specific pools...")
rng = random.Random(configs.RANDOM_SEED)

# 1. Split into 6 basic pools (long/short x high/mid/low)
long_high_pool = raw_high.filter(is_long, num_proc=os.cpu_count() or 1)
long_mid_pool = raw_mid.filter(is_long, num_proc=os.cpu_count() or 1)
long_low_pool = raw_low.filter(is_long, num_proc=os.cpu_count() or 1)

short_high_pool = raw_high.filter(is_short, num_proc=os.cpu_count() or 1)
short_mid_pool = raw_mid.filter(is_short, num_proc=os.cpu_count() or 1)
short_low_pool = raw_low.filter(is_short, num_proc=os.cpu_count() or 1)

# 2. Assemble Long Pool (target 2000, High has absolute priority)
# Priority: keep all long_high (but not exceeding the upper limit of 2000 just in case)
num_long_high = min(configs.LONG_SUBSETS_PROPORTION, len(long_high_pool))
sampled_long_high = long_high_pool.select(rng.sample(range(len(long_high_pool)), k=num_long_high))

# Fill: calculate how many slots are left, sample from mid and low long texts
remaining_long_slots = configs.LONG_SUBSETS_PROPORTION - num_long_high
long_rest_pool = concatenate_datasets([long_mid_pool, long_low_pool]).shuffle(seed=configs.RANDOM_SEED)
num_long_rest = min(remaining_long_slots, len(long_rest_pool))
sampled_long_rest = long_rest_pool.select(rng.sample(range(len(long_rest_pool)), k=num_long_rest))

# Merge to get the final 2000 long texts
sampled_long = concatenate_datasets([sampled_long_high, sampled_long_rest])


# 3. Assemble Short Pools (sample as needed)
num_sh_high = min(configs.SHORT_HIGH_SUBSETS_PROPORTION, len(short_high_pool))
sampled_short_high = short_high_pool.select(rng.sample(range(len(short_high_pool)), k=num_sh_high))

num_sh_mid = min(configs.SHORT_MID_SUBSETS_PROPORTION, len(short_mid_pool))
sampled_short_mid = short_mid_pool.select(rng.sample(range(len(short_mid_pool)), k=num_sh_mid))

num_sh_low = min(configs.SHORT_LOW_SUBSETS_PROPORTION, len(short_low_pool))
sampled_short_low = short_low_pool.select(rng.sample(range(len(short_low_pool)), k=num_sh_low))


# 4. Global merge and shuffle
dataset = concatenate_datasets([sampled_long, sampled_short_high, sampled_short_mid, sampled_short_low]).shuffle(
    seed=configs.RANDOM_SEED
)

logging.info(f"Final Count: {len(dataset)}")
logging.info(f"Long Pool Breakdown: High={len(sampled_long_high)}, Mid/Low={len(sampled_long_rest)}")
logging.info(
    f"Short Pool Breakdown: High={len(sampled_short_high)}, Mid={len(sampled_short_mid)}, Low={len(sampled_short_low)}"
)

# Map to Chat Template
dataset = dataset.map(_build_chat_example, num_proc=os.cpu_count() or 1).select_columns(["text"])


# ─── Training ─────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset.select(range(configs.TRAIN_SIZE)),
    eval_dataset=dataset.select(range(configs.TRAIN_SIZE, configs.DATASET_LEN)),
    args=SFTConfig(
        output_dir=configs.ARTIFACTS_DIR,
        dataset_text_field="text",
        per_device_train_batch_size=configs.MICRO_BSZ,
        gradient_accumulation_steps=configs.GRAD_ACCUM,
        eval_strategy="steps",
        eval_steps=configs.EVAL_STEPS,
        num_train_epochs=1,  # Set this for 1 full training run.
        max_steps=-1,
        warmup_steps=configs.WARMUP_STEPS,
        learning_rate=configs.LEARNING_RATE,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=configs.SAVE_STEPS,
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=configs.RANDOM_SEED,
        report_to="wandb",
        run_name=configs.RUN_NAME,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

trainer_stats = trainer.train()

# ─── Inference ────────────────────────────────────────────────────────────────

messages = [{"role": "user", "content": str(Query(f"This is an {guide()} text."))}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,  # Must add for generation
)

response = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=256,  # Increase for longer outputs!
    temperature=0.7,
    top_p=0.8,
    top_k=20,  # For non thinking
)
logging.info("Request: " + text)
logging.info("Response: " + tokenizer.decode(response[0]))

# ─── Save Model ───────────────────────────────────────────────────────────────

model.save_pretrained_merged(configs.FINAL_MODEL_DIR, tokenizer, save_method="merged_16bit")

# ─── Upload Model ─────────────────────────────────────────────────────────────

from huggingface_hub import HfApi


api = HfApi()
api.create_repo(
    repo_id="Sculpt-AI/" + configs.RUN_NAME,
    repo_type="model",
    private=True,
    exist_ok=True,
)
api.upload_folder(folder_path=str(configs.FINAL_MODEL_DIR), repo_id="Sculpt-AI/" + configs.RUN_NAME, repo_type="model")
