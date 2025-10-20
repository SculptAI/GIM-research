cd runs/2510051-tmp-train

nohup python ppl_server.py > ppl_server.log 2>&1 &
PPL_PID=$!

python train.py
# UNSLOTH_COMPILE_DISABLE=1 debugpy train.py  # debug

echo "Stopping ppl_server (PID: $PPL_PID)..."
kill $PPL_PID