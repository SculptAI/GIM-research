make serve model_path=artifacts/09251-gim-sft-tmp/sft-gim
python eval/gpqa_diamond.py --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 2 --num_proc 20 --first_n 198
python eval/gpqa_diamond.py --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 4 --num_proc 20 --first_n 198
python eval/gpqa_diamond.py --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 6 --num_proc 20 --first_n 198
python eval/gpqa_diamond.py --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 8 --num_proc 20 --first_n 198
python eval/gpqa_diamond.py --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 10 --num_proc 20 --first_n 198
python eval/gpqa_diamond.py --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 12 --num_proc 20 --first_n 198
python eval/gpqa_diamond.py --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 14 --num_proc 20 --first_n 198
python eval/mmlu_pro.py --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 2 --num_proc 20 --first_n 200
python eval/mmlu_pro.py --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 4 --num_proc 20 --first_n 200
python eval/mmlu_pro.py --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 6 --num_proc 20 --first_n 200
python eval/mmlu_pro.py --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 8 --num_proc 20 --first_n 200
python eval/mmlu_pro.py --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 10 --num_proc 20 --first_n 200
python eval/mmlu_pro.py --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 12 --num_proc 20 --first_n 200
python eval/mmlu_pro.py --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 14 --num_proc 20 --first_n 200

make serve model_path=unsloth/Qwen3-4B-Instruct-2507
python eval/gpqa_diamond.py --model_name unsloth/Qwen3-4B-Instruct-2507 --num_proc 20 --first_n 198
python eval/mmlu_pro.py --model_name unsloth/Qwen3-4B-Instruct-2507 --num_proc 20 --first_n 200
