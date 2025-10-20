# uvicorn <script>:app --host 0.0.0.0 --port 8000

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import configs
import uvicorn

app = FastAPI(title="CPU Transformers PPL Server")

class PPLRequest(BaseModel):
    text: str

model_name = configs.PPL_MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
model.to("cpu")

torch.set_grad_enabled(False)

@app.post("/ppl")
def compute_ppl(req: PPLRequest):
    text = req.text.strip()
    if not text:
        return {"error": "empty text"}

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]  # [1, seq_len]

    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss.item()
    ppl = math.exp(loss)
    print(f"PPL for '{text}': {ppl}")

    return {"ppl": ppl}

if __name__ == "__main__":
    uvicorn.run(app, host=configs.PPL_SERVER_HOST, port=configs.PPL_SERVER_PORT)