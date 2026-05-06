#TODO：
# 把官方代码里有的内容看看要不要copy过来
# 把之前训练代码里的东西看看要不要copy过来
# 完成debug
# 依赖问题处理完

import os


os.environ["UNSLOTH_STABLE_DOWNLOADS"] = (
    "1"  # https://unsloth.ai/docs/basics/troubleshooting-and-faqs#downloading-gets-stuck-at-90-to-95
)

from unsloth import FastLanguageModel  # noqa: I001
import torch
import logging
import random

import configs



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

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = configs.BASE_MODEL_NAME,
    max_seq_length = configs.MAX_SEQ_LENGTH,
    load_in_4bit = configs.QUANT_BITS == 4, # False for LoRA 16bit
    fast_inference = True, # Enable vllm fast inference
    max_lora_rank = configs.LORA_R,
    gpu_memory_utilization = 0.9, # Reduce if out of memory
    enforce_eager=True,  # important
)

model = FastLanguageModel.get_peft_model(
    model,
    r = configs.LORA_R, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = configs.LORA_ALPHA, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = configs.RANDOM_SEED,
)

# ─── Load Dataset ─────────────────────────────────────────────────────────────

from datasets import Dataset, concatenate_datasets, load_dataset

def _concat_subsets(subsets: list[str]) -> Dataset:
    return concatenate_datasets([load_dataset(configs.DATASET_NAME, subset, split="train") for subset in subsets])


def _build_chat_example(example: dict) -> dict:
    return {
        "prompt": 
            [
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

dataset = dataset.map(_build_chat_example, num_proc=os.cpu_count() or 1).select_columns(["prompt", "solution"])
logging.info(f"Dataset sample: {dataset[0]=}")



# ─── Remove Long Samples ──────────────────────────────────────────────────────

tokenized = dataset.map(
    lambda x: {
        "prompt_tokens" : tokenizer.apply_chat_template([
                {"role": "user", "content": x["gim_query"]},
            ], add_generation_prompt = True, tokenize = True),
        "prompt_completion_tokens" : tokenizer.apply_chat_template([
                {"role": "user", "content": x["gim_query"]},
                {"role": "assistant", "content": x["gim_response"]},
            ], add_generation_prompt = False, tokenize = True),
            },
    batched = True,
)
logging.info(f"{tokenizer.decode(tokenized[0]['prompt_tokens'])=}")
logging.info(f"{tokenizer.decode(tokenized[0]['prompt_completion_tokens'])=}")
tokenized = tokenized.map(lambda x: {"len_prompt" : len(x["prompt_tokens"]), "len_prompt_completion" : len(x["prompt_completion_tokens"])})

import numpy as np

# prompt + completion <= max_seq_length - 256
dataset = dataset.select(np.where(np.array(tokenized["len_prompt_completion"]) <= configs.MAX_SEQ_LENGTH - 256)[0])

# Get the maximum prompt length and prompt + completion length for logging
max_prompt_length = max(tokenized["len_prompt"])
max_prompt_completion_length = max(tokenized["len_prompt_completion"])
logging.info(f"Maximum prompt length: {max_prompt_length}")
logging.info(f"Maximum prompt + completion length: {max_prompt_completion_length}")

del tokenized


# ─── Reward Functions ─────────────────────────────────────────────────────────

def check_format(prompts, completions, solution, **kwargs):
    breakpoint()
    question = prompts[0][-1]["content"]
    response = completions[0][0]["content"]
    answer = solution[0]
    return 3.5 if response.strip() == answer.strip() else -1.5

# def check_numbers(prompts, completions, answer, **kwargs):
#     question = prompts[0][-1]["content"]
#     responses = [completion[0]["content"] for completion in completions]

#     extracted_responses = [
#         guess.group(1)
#         if (guess := match_numbers.search(r)) is not None else None \
#         for r in responses
#     ]

#     scores = []
#     # Print only every few steps
#     global PRINTED_TIMES
#     global PRINT_EVERY_STEPS
#     if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
#         print(
#             '*'*20 + f"Question:\n{question}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}"
#         )
#     PRINTED_TIMES += 1

#     for guess, true_answer in zip(extracted_responses, answer):
#         if guess is None:
#             scores.append(-2.5)
#             continue
#         # Convert to numbers
#         try:
#             true_answer = float(true_answer.strip())
#             # Remove commas like in 123,456
#             guess       = float(guess.strip().replace(",", ""))
#             scores.append(3.5 if guess == true_answer else -1.5)
#         except:
#             scores.append(0)
#             continue
#     return scores

# ─── Training ─────────────────────────────────────────────────────────────────

from vllm import SamplingParams
vllm_sampling_params = SamplingParams(
    min_p = configs.SAMPLING_PARAM_MIN_P,
    top_p = configs.SAMPLING_PARAM_TOP_P,
    top_k = configs.SAMPLING_PARAM_TOP_K,
    seed = configs.RANDOM_SEED,
    stop = [tokenizer.eos_token],
    include_stop_str_in_output = True,
)

from trl import GRPOConfig, GRPOTrainer

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        check_format,
    ],
    train_dataset = dataset.select(range(configs.TRAIN_SIZE)),
    eval_dataset = None if configs.NO_EVAL else dataset.select(range(configs.TRAIN_SIZE, configs.DATASET_LEN)),
    args = GRPOConfig(
    vllm_sampling_params = vllm_sampling_params,
    temperature = configs.SAMPLING_PARAM_TEMPERATURE,
    learning_rate = configs.LEARNING_RATE,
    weight_decay = configs.WEIGHT_DECAY,
    warmup_steps = configs.WARMUP_STEPS,
    lr_scheduler_type = configs.LR_SCHEDULER_TYPE,
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = configs.MICRO_BSZ,
    gradient_accumulation_steps = configs.GRAD_ACCUM, # Increase to 4 for smoother training
    num_generations = configs.NUM_GENERATIONS, # Decrease if out of memory
    max_prompt_length = max_prompt_length + 1,  # + 1 just in case
    max_completion_length = configs.MAX_SEQ_LENGTH - (max_prompt_length+1),
    num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = -1,
    save_steps = configs.SAVE_STEPS,
    report_to="wandb",
    output_dir = configs.ARTIFACTS_DIR,
    run_name = configs.RUN_NAME,

    # For optional evaluation
    fp16_full_eval = True,
    per_device_eval_batch_size = configs.GLOBAL_BSZ,
    eval_accumulation_steps = None if configs.NO_EVAL else configs.GRAD_ACCUM,
    eval_strategy = "no" if configs.NO_EVAL else "steps",
    eval_steps = None if configs.NO_EVAL else configs.EVAL_STEPS,
),
)
trainer.train()


# ─── Inference ────────────────────────────────────────────────────────────────



from gimkit import guide
from gimkit.contexts import Query

text = str(Query(f"This is an {guide()} text."))
logging.info("Request: " + text)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 1.0,
    top_k = 50,
    max_tokens = 1024,
)
output = model.fast_generate(
    [text],
    sampling_params = sampling_params,
    lora_request = None,
)[0].outputs[0].text
logging.info("Response: " + output)


# ─── Save Model ───────────────────────────────────────────────────────────────


"""And now with the LoRA we just trained with GRPO - we first save the LoRA first!"""

model.save_lora("grpo_saved_lora")

"""Verify LoRA is actually trained!"""

from safetensors import safe_open

tensors = {}
with safe_open("grpo_saved_lora/adapter_model.safetensors", framework = "pt") as f:
    # Verify both A and B are non zero
    for key in f.keys():
        tensor = f.get_tensor(key)
        n_zeros = (tensor == 0).sum() / tensor.numel()
        assert(n_zeros.item() != tensor.numel())

"""Now we load the LoRA and test:"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": "What is the sqrt of 101?"},
]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    tokenize = False,
)
from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 1.0,
    top_k = 50,
    max_tokens = 2048,
)
output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text

output

"""Our reasoning model is much better - it's not always correct, since we only trained it for an hour or so - it'll be better if we extend the sequence length and train for longer!

<a name="Save"></a>
### Saving to float16 for VLLM

We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.
"""

# Merge to 16bit
if False: model.save_pretrained_merged("qwen_finetune_16bit", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("HF_USERNAME/qwen_finetune_16bit", tokenizer, save_method = "merged_16bit", token = "YOUR_HF_TOKEN")

# Merge to 4bit
if False: model.save_pretrained_merged("qwen_finetune_4bit", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("HF_USERNAME/qwen_finetune_4bit", tokenizer, save_method = "merged_4bit", token = "YOUR_HF_TOKEN")

# Just LoRA adapters
if False:
    model.save_pretrained("qwen_lora")
    tokenizer.save_pretrained("qwen_lora")
if False:
    model.push_to_hub("HF_USERNAME/qwen_lora", token = "YOUR_HF_TOKEN")
    tokenizer.push_to_hub("HF_USERNAME/qwen_lora", token = "YOUR_HF_TOKEN")

"""### GGUF / llama.cpp Conversion
To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.

Some supported quant methods (full list on our [docs page](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)):
* `q8_0` - Fast conversion. High resource use, but generally acceptable.
* `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
* `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.

[**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
"""

# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("qwen_finetune", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False: model.push_to_hub_gguf("HF_USERNAME/qwen_finetune", tokenizer, token = "YOUR_HF_TOKEN")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("qwen_finetune", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("HF_USERNAME/qwen_finetune", tokenizer, quantization_method = "f16", token = "YOUR_HF_TOKEN")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("qwen_finetune", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("HF_USERNAME/qwen_finetune", tokenizer, quantization_method = "q4_k_m", token = "YOUR_HF_TOKEN")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "HF_USERNAME/qwen_finetune", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "YOUR_HF_TOKEN",
    )

"""Now, use the `qwen_finetune.Q8_0.gguf` file or `qwen_finetune.Q4_K_M.gguf` file in llama.cpp.

And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!

Some other resources:
1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
4. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://unsloth.ai/docs/get-started/unsloth-notebooks)!

<div class="align-center">
  <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
  <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
  <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

  Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
</div>

  This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
"""