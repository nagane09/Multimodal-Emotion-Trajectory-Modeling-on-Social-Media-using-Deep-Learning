import pandas as pd
import numpy as np
import torch
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

INPUT_FILE = "data/processed/social_posts_4class.csv"
OUTPUT_X_FILE = "data/processed/X.npy"
OUTPUT_Y_FILE = "data/processed/y.npy"
OUTPUT_LE_FILE = "data/processed/le.pkl"
OUTPUT_CLASS_WEIGHTS_FILE = "data/processed/class_weights.npy"

EMBEDDING_DIM = 16
emoji_dict = {}  

def get_emoji_embedding(emoji_str):
    emb = np.zeros(EMBEDDING_DIM)
    for c in emoji_str:
        if c not in emoji_dict:
            emoji_dict[c] = np.random.rand(EMBEDDING_DIM)
        emb += emoji_dict[c]
    return emb

def extract_features(input_file):
    df = pd.read_csv(input_file)
    df = df.dropna(subset=["text", "emotion"])
    texts = df["text"].astype(str).tolist()
    labels = df["emotion"].tolist()

    print("Generating sentence embeddings...")
    model = SentenceTransformer("all-MiniLM-L12-v2")
    X_text = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    if "emojis" in df.columns:
        df["emojis"] = df["emojis"].fillna("")
        X_emoji = np.vstack(df["emojis"].apply(get_emoji_embedding).values)
    else:
        X_emoji = np.zeros((len(df), EMBEDDING_DIM))

    if "created_at" in df.columns:
        hour = pd.to_datetime(df["created_at"], errors="coerce").dt.hour.fillna(0)
        X_meta_hour = hour.values.reshape(-1, 1) / 23.0
    else:
        X_meta_hour = np.zeros((len(df), 1))

    X_meta_len = np.array([len(t.split()) for t in texts]).reshape(-1, 1) / 100.0
    X_meta = np.hstack([X_meta_hour, X_meta_len])

    X = np.hstack([X_text, X_emoji, X_meta])

    le = LabelEncoder()
    y = le.fit_transform(labels)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    np.save(OUTPUT_X_FILE, X)
    np.save(OUTPUT_Y_FILE, y)
    with open(OUTPUT_LE_FILE, "wb") as f:
        pickle.dump(le, f)
    np.save(OUTPUT_CLASS_WEIGHTS_FILE, class_weights.numpy())

    print("Features saved successfully!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Class weights:", class_weights)

if __name__ == "__main__":
    extract_features(INPUT_FILE)
