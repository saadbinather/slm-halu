from pathlib import Path
import re
import torch

ROOT = Path(__file__).parent

KEYWORDS = [
    "lr", "learning_rate", "dropout", "weight_decay",
    "batch_size", "epochs", "max_epochs", "patience",
    "hidden", "hidden_dim", "hidden_channels",
    "optimizer", "temperature", "top_p", "top_k",
    "do_sample", "max_new_tokens"
]

print("\n=== HYPERPARAMETERS FOUND IN SCRIPTS ===\n")

for file in sorted(ROOT.glob("*.py")):
    if file.name == Path(__file__).name:
        continue

    text = file.read_text(encoding="utf-8", errors="ignore")
    matches = []

    for i, line in enumerate(text.splitlines(), 1):
        low = line.lower()
        if any(k in low for k in KEYWORDS):
            if "=" in line or "optimizer" in low:
                matches.append((i, line.strip()))

    if matches:
        print(f"\n--- {file.name} ---")
        for line_no, line in matches:
            print(f"{line_no}: {line}")

print("\n\n=== VALIDATION LABEL COUNTS ===\n")

val_dir = ROOT / "graphs" / "val"

if val_dir.exists():
    correct = 0
    hallucinated = 0
    total = 0

    for path in val_dir.glob("*.pt"):
        data = torch.load(path, map_location="cpu", weights_only=False)

        y = data.y
        if torch.is_tensor(y):
            y = int(y.view(-1)[0].item())
        else:
            y = int(y)

        total += 1
        if y == 0:
            correct += 1
        elif y == 1:
            hallucinated += 1

    print(f"Total:        {total}")
    print(f"Correct:      {correct}")
    print(f"Hallucinated: {hallucinated}")
else:
    print("graphs/val not found")

print("\n\n=== HARDWARE ===\n")
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print(f"GPU memory: {props.total_memory / 1024**3:.1f} GB")

print("\n=== DONE ===")