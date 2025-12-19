# Multimodal-Emotion-Trajectory-Modeling-on-Social-Media-using-Deep-Learning

## **Live Demo**

You can try the project online using the deployed Streamlit app:  

[Open Live Demo](https://your-deployed-link.com)

* Enter text (with emojis if desired) to see real-time emotion predictions.


# Social Media Emotion Analysis

This project analyzes emotions expressed in social media posts using text, emojis, and metadata. It provides a **pipeline from raw data to real-time emotion prediction** using deep learning.

---

## **Features**

* Preprocesses raw Reddit posts: cleans text, extracts emojis, and metadata (posting hour, text length).
* Maps 28 GoEmotions labels to **4 broad classes**: `positive`, `negative`, `mixed`, `neutral`.
* Converts text + emojis + metadata into numerical embeddings using **SentenceTransformer**.
* Trains a **PyTorch neural network** for emotion classification.
* Streamlit web app for **real-time emotion prediction** from user input.

---

## **Project Structure**

data/
├─ raw/
│ ├─ goemotions_1.csv
│ ├─ goemotions_2.csv
│ └─ goemotions_3.csv
└─ processed/
├─ social_posts.csv
├─ social_posts_processed.csv
├─ social_posts_sampled.csv
├─ social_posts_4class.csv
├─ X.npy
├─ y.npy
├─ le.pkl
└─ class_weights.npy

models/
└─ emotion_classifier.pt

src/
├─ preprocess.py # Data cleaning & preprocessing
├─ extract_features.py # Feature extraction (text, emoji, metadata)
├─ train_model.py # Train emotion classification model
└─ app.py # Streamlit app for predictions


## **Technologies Used**

* Python, Pandas, NumPy
* PyTorch
* Sentence Transformers (`all-MiniLM-L12-v2`)
* Streamlit
* NLTK for text preprocessing

---

## **Output**

* **Processed CSVs**: clean, preprocessed social media posts.
* **Numpy arrays**: `X.npy`, `y.npy` for features and labels.
* **Trained model**: `emotion_classifier.pt`.
* **Streamlit app**: real-time emotion prediction.

