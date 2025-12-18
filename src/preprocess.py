# src/preprocess.py

import pandas as pd
import numpy as np
import re
import emoji
from nltk.corpus import stopwords
import os

import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def load_data(file_path):
    """
    Load raw CSV data
    """
    df = pd.read_csv(file_path)
    return df


def clean_text(text):
    """
    Lowercase, remove URLs, mentions, hashtags, punctuation, and stopwords
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text) 
    text = re.sub(r"@\w+|#\w+", "", text)      # remove mentions/hashtags
    text = re.sub(r"[^a-z\s]", "", text)       # remove punctuation
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text


def extract_emojis(text):
    """
    Extract emojis from text using the updated emoji library
    """
    return ''.join(c for c in str(text) if c in emoji.EMOJI_DATA)

def extract_metadata(df):
    """
    Extract metadata features
    """
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['hour'] = df['created_at'].dt.hour
    return df

def preprocess_pipeline(input_file, output_file):
    df = load_data(input_file)
    
    df['clean_text'] = df['text'].apply(clean_text)
    
    df['emojis'] = df['text'].apply(extract_emojis)
    
    df = extract_metadata(df)
    
    df_processed = df[['text', 'clean_text', 'emojis', 'created_at', 'hour', 'emotion']]
    
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_processed.to_csv(output_file, index=False)
    
    print(f"Preprocessing complete. Saved processed data to {output_file}")
    return df_processed


if __name__ == "__main__":
    input_file = "data/processed/social_posts.csv"
    output_file = "data/processed/social_posts_processed.csv"
    preprocess_pipeline(input_file, output_file)
