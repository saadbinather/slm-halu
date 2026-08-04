import pandas as pd

# Sahi raw URLs
train_url = "https://raw.githubusercontent.com/technion-cs-nlp/LLMsKnow/main/data/movie_qa_train.csv"
test_url = "https://raw.githubusercontent.com/technion-cs-nlp/LLMsKnow/main/data/movie_qa_test.csv"

print("Loading Train and Test datasets...")

# Load into Pandas DataFrames
df_train = pd.read_csv(train_url)
df_test = pd.read_csv(test_url)

print(f"✅ Train samples: {len(df_train)} | Test samples: {len(df_test)}")
print("\n--- First Train Sample ---")
print(df_train.head(1))
print(f"Train: {len(df_train)} samples, Test: {len(df_test)} samples")

# 2000 samples for training, 500 for testing
df_train_small = df_train.sample(n=2000, random_state=42).reset_index(drop=True)
df_test_small = df_test.sample(n=500, random_state=42).reset_index(drop=True)

print(f"✅ Final Split - Train: {len(df_train_small)} | Test: {len(df_test_small)}")