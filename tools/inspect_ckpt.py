import os
import json
import torch
from safetensors.torch import load_file

def read_trainer_state(path):
    f = os.path.join(path, "trainer_state.json")
    if not os.path.exists(f):
        print("trainer_state.json not found")
        return
    with open(f) as fp:
        state = json.load(fp)

    print("\n=== TRAINER STATE ===")
    print("global_step:", state.get("global_step"))
    print("epoch:", state.get("epoch"))
    print("best_model_checkpoint:", state.get("best_model_checkpoint"))

    logs = state.get("log_history", [])
    print("\nLast 5 log entries:")
    for entry in logs[-5:]:
        print(entry)


def read_optimizer(path):
    f = os.path.join(path, "optimizer.pt")
    if not os.path.exists(f):
        print("optimizer.pt not found")
        return
    opt = torch.load(f, map_location="cpu")

    print("\n=== OPTIMIZER PARAM GROUPS ===")
    for i, g in enumerate(opt["param_groups"]):
        print(f"Group {i}: lr={g['lr']}, weight_decay={g['weight_decay']}")

    print("\n=== OPTIMIZER STATE SAMPLE ===")
    # Print one param state to confirm updates
    for k, v in opt["state"].items():
        print("Param:", k)
        for sname, sval in list(v.items())[:5]:  # Print only first 5 for brevity
            if torch.is_tensor(sval):
                print(f"  {sname}: mean={sval.float().mean().item():.6f}, std={sval.float().std().item():.6f}")
            else:
                print(f"  {sname}: {sval}")
        break


def read_scheduler(path):
    f = os.path.join(path, "scheduler.pt")
    if not os.path.exists(f):
        print("scheduler.pt not found")
        return
    sched = torch.load(f, map_location="cpu")

    print("\n=== SCHEDULER STATE ===")
    for k, v in sched.items():
        print(k, ":", v)


def read_lora(path):
    f = os.path.join(path, "adapter_model.safetensors")
    if not os.path.exists(f):
        print("adapter_model.safetensors not found")
        return
    lora = load_file(f)

    print("\n=== LORA WEIGHTS SUMMARY ===")
    for k, v in list(lora.items())[:5]:  # Print only first 5 for brevity
        print(f"{k}: mean={v.mean().item():.6f}, std={v.std().item():.6f}")


def compare_checkpoints(path1, path2):
    f1 = os.path.join(path1, "adapter_model.safetensors")
    f2 = os.path.join(path2, "adapter_model.safetensors")

    if not (os.path.exists(f1) and os.path.exists(f2)):
        print("Both checkpoints must contain adapter_model.safetensors")
        return

    a = load_file(f1)
    b = load_file(f2)

    print("\n=== CHECKPOINT DIFF (mean abs diff) ===")
    for k in list(a.keys())[:5]:  # Compare only first 5 keys for brevity
        diff = (a[k] - b[k]).abs().mean().item()
        print(f"{k}: {diff:.8f}")


def inspect_checkpoint(path, compare_to=None):
    print(f"\n########## INSPECTING {path} ##########")
    read_trainer_state(path)
    read_optimizer(path)
    read_scheduler(path)
    read_lora(path)

    if compare_to:
        print(f"\n########## COMPARING TO {compare_to} ##########")
        compare_checkpoints(path, compare_to)


# -------------------------
# Example usage:
# -------------------------

# Single checkpoint inspection
inspect_checkpoint("/mnt/data/artifacts/2605061-gim1_7b-rl/checkpoint-2")

# Compare two checkpoints
inspect_checkpoint(
    "/mnt/data/artifacts/2605061-gim1_7b-rl/checkpoint-2",
    compare_to="/mnt/data/artifacts/2605061-gim1_7b-rl/checkpoint-4"
)
