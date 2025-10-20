make serve model_path=artifacts/09251-gim-sft-tmp/sft-gim
python -m eval.mcqa.gpqa_diamond --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 2 --num_proc 40 --first_n 198
python -m eval.mcqa.gpqa_diamond --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 4 --num_proc 40 --first_n 198
python -m eval.mcqa.gpqa_diamond --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 6 --num_proc 40 --first_n 198
python -m eval.mcqa.gpqa_diamond --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 8 --num_proc 40 --first_n 198
python -m eval.mcqa.gpqa_diamond --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 10 --num_proc 40 --first_n 198
python -m eval.mcqa.gpqa_diamond --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 12 --num_proc 40 --first_n 198
python -m eval.mcqa.gpqa_diamond --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 14 --num_proc 40 --first_n 198
python -m eval.mcqa.gpqa_diamond --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 16 --num_proc 40 --first_n 198
python -m eval.mcqa.gpqa_diamond --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 18 --num_proc 40 --first_n 198
python -m eval.mcqa.mmlu_pro --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 2 --num_proc 40 --first_n 1000
python -m eval.mcqa.mmlu_pro --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 4 --num_proc 40 --first_n 1000
python -m eval.mcqa.mmlu_pro --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 6 --num_proc 40 --first_n 1000
python -m eval.mcqa.mmlu_pro --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 8 --num_proc 40 --first_n 1000
python -m eval.mcqa.mmlu_pro --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 10 --num_proc 40 --first_n 1000
python -m eval.mcqa.mmlu_pro --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 12 --num_proc 40 --first_n 1000
python -m eval.mcqa.mmlu_pro --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 14 --num_proc 40 --first_n 1000
python -m eval.mcqa.mmlu_pro --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 16 --num_proc 40 --first_n 1000
python -m eval.mcqa.mmlu_pro --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 18 --num_proc 40 --first_n 1000

make serve model_path=unsloth/Qwen3-4B-Instruct-2507
python -m eval.mcqa.gpqa_diamond --model_name unsloth/Qwen3-4B-Instruct-2507 --num_proc 40 --first_n 198
python -m eval.mcqa.mmlu_pro --model_name unsloth/Qwen3-4B-Instruct-2507 --num_proc 40 --first_n 1000
