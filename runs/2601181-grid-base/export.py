import os
os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"  # https://unsloth.ai/docs/basics/troubleshooting-and-faqs#downloading-gets-stuck-at-90-to-95


from unsloth import FastModel  # noqa: I001

import logging

import configs

from gimkit import guide
from gimkit.contexts import Query
from transformers import AutoModelForCausalLM, AutoTokenizer


CKPT_PATH = "xxx"

# ─── General Setup ────────────────────────────────────────────────────────────

os.environ["WANDB_PROJECT"] = configs.PROJECT_NAME
os.environ["WANDB_DIR"] = str(configs.ARTIFACTS_DIR)
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

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
    model_name=CKPT_PATH,
    max_seq_length=configs.MAX_SEQ_LENGTH,
    load_in_4bit=configs.QUANT_BITS == 4,
    load_in_8bit=configs.QUANT_BITS == 8,
    full_finetuning=False,
    token=None,
)

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

saved_model = AutoModelForCausalLM.from_pretrained(
    configs.FINAL_MODEL_DIR, torch_dtype="auto", device_map="auto"
)
saved_tokenizer = AutoTokenizer.from_pretrained(configs.FINAL_MODEL_DIR)

response = saved_model.generate(
    **saved_tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=256,  # Increase for longer outputs!
    temperature=0.7,
    top_p=0.8,
    top_k=20,  # For non thinking
)
logging.info("Request (from saved model): " + text)
logging.info("Response (from saved model): " + saved_tokenizer.decode(response[0]))

# ─── Upload Model ─────────────────────────────────────────────────────────────

from huggingface_hub import HfApi

api = HfApi()
api.create_repo(
    repo_id="Sculpt-AI/" + configs.RUN_NAME,
    repo_type="model",
    private=True,
    exist_ok=True,
)
api.upload_folder(
    folder_path=str(configs.FINAL_MODEL_DIR),
    repo_id="Sculpt-AI/" + configs.RUN_NAME,
    repo_type="model"
)
