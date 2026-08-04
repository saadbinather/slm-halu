import torch
import os
import random

folder = "movie_traces_train2"

# Folder se saari .pt files ki list banayen
all_files = [f for f in os.listdir(folder) if f.endswith('.pt')]

# 5 random files select karein
random_files = random.sample(all_files, min(5, len(all_files)))

print("--- 5 Random Samples (Generated vs Gold) ---\n")
for file in random_files:
    # File load karein
    data = torch.load(os.path.join(folder, file), map_location="cpu")
    
    print(f"File: {file}")
    print(f"Question: {data['question']}")
    print(f"Gold Ans: {data['gold_answer']}")
    print(f"Gen Ans:  {data['generated_answer']}")
    print("-" * 50)




