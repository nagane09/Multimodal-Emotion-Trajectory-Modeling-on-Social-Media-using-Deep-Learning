import streamlit as st
import torch
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import pandas as pd

# ----------------- Load trained model -----------------
class EmotionClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Load features to get input_dim
X = np.load("data/processed/X.npy")
input_dim = X.shape[1]

# Load label encoder
with open("data/processed/le.pkl", "rb") as f:
    le = pickle.load(f)

num_classes = len(le.classes_)

# Load model
model = EmotionClassifier(input_dim, num_classes)
model.load_state_dict(torch.load("models/emotion_classifier.pt", map_location="cpu"))
model.eval()

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L12-v2")

# ----------------- Streamlit UI -----------------
st.title("Multimodal Emotion Predictor")
st.write("Enter a text (with emojis if you want) to predict its emotion:")

text_input = st.text_area("Text here:")

if st.button("Predict"):
    if not text_input.strip():
        st.warning("Please enter some text!")
    else:
        # --------- Text embeddings ---------
        text_emb = embedding_model.encode([text_input], normalize_embeddings=True)

        # --------- Emoji embedding ---------
        EMBEDDING_DIM = 16
        def get_emoji_embedding(emoji_str):
            emb = np.zeros(EMBEDDING_DIM)
            for c in emoji_str:
                emb += np.random.rand(EMBEDDING_DIM)
            return emb
        import re
        emojis = ''.join(re.findall(r'[\U00010000-\U0010ffff]', text_input))
        emoji_emb = get_emoji_embedding(emojis).reshape(1, -1)

        # --------- Metadata (simple example: text length) ---------
        meta_hour = np.array([[0]])  # unknown posting hour
        meta_len = np.array([[len(text_input.split()) / 100.0]])
        meta_features = np.hstack([meta_hour, meta_len])

        # --------- Combine all features ---------
        X_input = np.hstack([text_emb, emoji_emb, meta_features])
        X_input = torch.tensor(X_input, dtype=torch.float32)

        # --------- Predict ---------
        with torch.no_grad():
            logits = model(X_input)
            pred_class = torch.argmax(logits, dim=1).item()
            emotion = le.inverse_transform([pred_class])[0]

        st.success(f"Predicted Emotion: **{emotion}**")
