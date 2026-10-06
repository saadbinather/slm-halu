from pathlib import Path
import torch

VAL_DIR = Path("graphs/val")

correct = 0
hallucinated = 0

for path in VAL_DIR.glob("*.pt"):
    data = torch.load(path, map_location="cpu", weights_only=False)

    y = data.y
    if torch.is_tensor(y):
        y = int(y.view(-1)[0].item())
    else:
        y = int(y)

    if y == 0:
        correct += 1
    elif y == 1:
        hallucinated += 1

print("Validation total:", correct + hallucinated)
print("Correct:", correct)
print("Hallucinated:", hallucinated)