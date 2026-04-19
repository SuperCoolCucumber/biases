from datasets import load_dataset
import pandas as pd
import numpy as np

# --------------------------------------------------
# Reproducibility: set global random seed
# --------------------------------------------------
np.random.seed(42)

# --------------------------------------------------
# Load dataset (human judgments split)
# --------------------------------------------------
dataset = load_dataset("lmsys/mt_bench_human_judgments")
df = pd.DataFrame(dataset['human'])

# --------------------------------------------------
# Inspect label distribution
# --------------------------------------------------
print("Original label distribution:\n", df['winner'].value_counts())

# --------------------------------------------------
# Stratified sampling setup
# --------------------------------------------------
num_labels = df['winner'].nunique()
samples_per_label = 200 // num_labels  # ensures equal distribution

# Perform stratified sampling (balanced across labels)
df_sampled = (
    df.groupby('winner', group_keys=False)[df.columns]
      .apply(lambda x: x.sample(n=samples_per_label, random_state=42))
)

# --------------------------------------------------
# Shuffle final dataset
# --------------------------------------------------
df_final = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

# --------------------------------------------------
# Keep only relevant columns
# --------------------------------------------------
columns_to_keep = [
    'question_id',
    'model_a',
    'model_b',
    'winner',
    'turn',
    'conversation_a',
    'conversation_b'
]
df_final = df_final[columns_to_keep]

# --------------------------------------------------
# Report models present in the split
# --------------------------------------------------
models_a = set(df_final['model_a'])
models_b = set(df_final['model_b'])
all_models = sorted(models_a.union(models_b))

print("\nModels in the split:")
for model in all_models:
    print(model)

print("\nTotal unique models:", len(all_models))

# --------------------------------------------------
# Final validation checks
# --------------------------------------------------
print("\nFinal label distribution:\n", df_final['winner'].value_counts())
print("Total samples:", len(df_final))

# --------------------------------------------------
# Model frequency analysis (basic data analysis)
# --------------------------------------------------
model_counts = pd.concat([
    df_final['model_a'],
    df_final['model_b']
]).value_counts()

print("\nModel frequency:\n", model_counts)

# --------------------------------------------------
# Save final dataset
# --------------------------------------------------
df_final.to_csv("mtbench_stratified_198.csv", index=False)