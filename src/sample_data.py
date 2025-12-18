import pandas as pd

INPUT_FILE = "data/processed/social_posts_processed.csv"
OUTPUT_FILE = "data/processed/social_posts_sampled.csv"

SAMPLES_PER_EMOTION = 1000  # safe & fast

df = pd.read_csv(INPUT_FILE)

sampled_df = (
    df.groupby("emotion", group_keys=False)
      .apply(lambda x: x.sample(min(len(x), SAMPLES_PER_EMOTION), random_state=42))
)

sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Original rows:", len(df))
print("Sampled rows:", len(sampled_df))
print(sampled_df["emotion"].value_counts())

sampled_df.to_csv(OUTPUT_FILE, index=False)
