import pandas as pd

df = pd.read_csv("data/processed/social_posts_sampled.csv")

emotion_map = {
    "joy": "positive",
    "love": "positive",
    "surprise": "positive",
    "anger": "negative",
    "disgust": "negative",
    "sadness": "negative",
    "fear": "mixed",
    "neutral": "neutral"
}

df["emotion"] = df["emotion"].map(emotion_map)

df.to_csv("data/processed/social_posts_4class.csv", index=False)
print("Classes distribution:")
print(df["emotion"].value_counts())
