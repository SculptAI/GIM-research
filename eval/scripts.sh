make serve model_path=artifacts/09251-gim-sft-tmp/sft-gim
python eval/gpqa_diamond.py --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 5 --num_proc 20
python eval/gpqa_diamond.py --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 10 --num_proc 20
python eval/mmlu_pro.py --model_name artifacts/09251-gim-sft-tmp/sft-gim --is_gim --reason_budget 5 --num_proc 20 --first_n 100

make serve model_path=unsloth/Qwen3-4B-Instruct-2507
python eval/gpqa_diamond.py --model_name unsloth/Qwen3-4B-Instruct-2507 --num_proc 20
python eval/mmlu_pro.py --model_name unsloth/Qwen3-4B-Instruct-2507 --num_proc 20 --first_n 100
