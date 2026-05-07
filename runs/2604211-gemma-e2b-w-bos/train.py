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
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=configs.LORA_R,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    lora_alpha=configs.LORA_ALPHA,
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
    chat_template="gemma-4",
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

dataset = dataset.map(_build_chat_example, num_proc=os.cpu_count() or 1).select_columns(["text"])


# ─── Training ─────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset.select(range(configs.TRAIN_SIZE)),
    eval_dataset=None if configs.NO_EVAL else dataset.select(range(configs.TRAIN_SIZE, configs.DATASET_LEN)),
    args=SFTConfig(
        output_dir=configs.ARTIFACTS_DIR,
        dataset_text_field="text",
        per_device_train_batch_size=configs.MICRO_BSZ,
        gradient_accumulation_steps=configs.GRAD_ACCUM,
        eval_strategy="no" if configs.NO_EVAL else "steps",
        eval_steps=None if configs.NO_EVAL else configs.EVAL_STEPS,
        num_train_epochs=1,  # Set this for 1 full training run.
        max_steps=-1,
        warmup_steps=configs.WARMUP_STEPS,
        learning_rate=configs.LEARNING_RATE,
        lr_scheduler_type=configs.LR_SCHEDULER_TYPE,
        logging_steps=1,
        save_steps=configs.SAVE_STEPS,
        optim="adamw_8bit",
        weight_decay=configs.WEIGHT_DECAY,
        seed=configs.RANDOM_SEED,
        report_to="wandb",
        run_name=configs.RUN_NAME,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|turn>user\n",
    response_part="<|turn>model\n",
)
logging.info("Training data example:", tokenizer.decode(trainer.train_dataset[100]["input_ids"]))
logging.info(
    "Training data labels example:",
    tokenizer.decode(
        [tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]
    ).replace(tokenizer.pad_token, " "),
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
    **tokenizer(text=text, return_tensors="pt").to("cuda"),
    max_new_tokens=256,  # Increase for longer outputs!
    temperature=1.0,
    top_p=0.95,
    top_k=64,  # For non thinking
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
