from unsloth import FastModel  # noqa: I001

import logging
import os
import re

import configs
import torch

from datasets import Dataset, concatenate_datasets, load_dataset
from gimkit import Query, guide
from gimkit.contexts import infill
from gimkit.exceptions import InvalidFormatError
from gimkit.schemas import QUERY_PREFIX, QUERY_SUFFIX
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer, create_reference_model
from trl.extras.profiling import profiling_decorator

from unsloth.chat_templates import get_chat_template, train_on_responses_only
import requests


# ─── General Setup ────────────────────────────────────────────────────────────

os.environ["WANDB_PROJECT"] = configs.PROJECT_NAME
os.environ["WANDB_DIR"] = str(configs.ARTIFACTS_DIR / "wandb")
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

logging.basicConfig(
    filename="train.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info("Training configurations:")
for key, value in vars(configs).items():
    if not key.startswith("__"):
        logging.info(f"{key} = {value}")


# ─── Custom PPL Metric ────────────────────────────────────────────────────────

def compute_ppl_via_server(text: str, server_url: str = configs.PPL_URL) -> float:
    payload = {"text": text}

    resp = requests.post(server_url, json=payload)

    if resp.status_code == 200:
        return resp.json().get("ppl", 0.0)
    else:
        logging.error(f"PPL server error {resp.status_code}: {resp.text}")
        return 0.0


class SFTTrainerWithCTP(SFTTrainer):
    @profiling_decorator
    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        # <copied from transformers.trainer>
        # handle multiple eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics
        # </copied from transformers.trainer>

        # run original evaluation
        eval_output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        logging.info("Calculating Composite Text Perplexity (CTP) with reference model...")

        ctps = []
        for example in tqdm(eval_dataset, desc="CTP Calculation"):
            if not (query_match := re.search(f"({re.escape(QUERY_PREFIX)}.*?{re.escape(QUERY_SUFFIX)})", example["text"], re.DOTALL)):
                continue
            query = query_match.group(1).strip()
            self.processing_class: PreTrainedTokenizerBase
            prompt = self.processing_class.apply_chat_template(
                [{"role": "user", "content": query}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processing_class(prompt, return_tensors="pt").to(self.model.device)

            # Generate response with the model being trained
            response_ids = self.model.generate(
                **inputs, max_length=configs.MAX_SEQ_LENGTH, pad_token_id=self.processing_class.eos_token_id
            )
            response = self.processing_class.decode(
                response_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            try:
                infilled = infill(query, response).to_string(fields=[])
            except InvalidFormatError:
                continue

            # Calculate PPL with reference model
            ppl = compute_ppl_via_server(infilled)
            if ppl > 0:
                ctps.append(ppl)

        if ctps:
            avg_ctp = sum(ctps) / len(ctps)
            eval_output[f"{metric_key_prefix}_ctp"] = avg_ctp
            self.log(eval_output)
            logging.info(f"Average CTP with reference model: {avg_ctp:.2f}")
            logging.info(f"Last query: {query}")
            logging.info(f"Last response: {response}")
            logging.info(f"Last infilled: {infilled}")

        return eval_output


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


def _concat_subsets(subsets: list[str]) -> Dataset:
    return concatenate_datasets([load_dataset(configs.DATASET_NAME, subset, split="train") for subset in subsets])


logging.info("Loading and preparing dataset...")
high_dataset = _concat_subsets(high_subsets)
if configs.DATASET_LEN - len(high_dataset) > 0:
    _rest_len = configs.DATASET_LEN - len(high_dataset)
    _mid_len = int(_rest_len * 0.6)
    _low_len = _rest_len - _mid_len
    mid_dataset = _concat_subsets(mid_subsets).shuffle(seed=configs.RANDOM_SEED).select(range(_mid_len))
    low_dataset = _concat_subsets(low_subsets).shuffle(seed=configs.RANDOM_SEED).select(range(_low_len))

    assert len(high_dataset) + len(mid_dataset) + len(low_dataset) == configs.DATASET_LEN
    dataset = concatenate_datasets([high_dataset, mid_dataset, low_dataset])
    logging.info(f"Dataset sizes: high {len(high_dataset)}, mid {len(mid_dataset)}, low {len(low_dataset)}")

else:
    dataset = high_dataset.shuffle(seed=configs.RANDOM_SEED).select(range(configs.DATASET_LEN))
    logging.info(f"Dataset sizes: high {len(dataset)}")

dataset = (
    dataset.shuffle(seed=configs.RANDOM_SEED)
    .map(
        lambda example: {
            "text": tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": example["gim_query"]},
                    {"role": "assistant", "content": example["gim_response"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        },
        num_proc=os.cpu_count(),
    )
    .select_columns(["text"])
)

# ─── Training ─────────────────────────────────────────────────────────────────

trainer = SFTTrainerWithCTP(
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
        learning_rate=2e-4,  # Reduce to 2e-5 for long training runs
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
