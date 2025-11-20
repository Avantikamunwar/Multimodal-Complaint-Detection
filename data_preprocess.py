import pandas as pd
import re
import pickle
from sklearn.preprocessing import LabelEncoder
from config.config import ENCODER_SAVE_PATH

def clean_text(s):
    if pd.isna(s): return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def map_complaint(v):
    if pd.isna(v): return 0
    v = str(v).lower().strip()
    if v in ["1", "true", "yes", "complaint"]: return 1
    return 0

def preprocess_dataset(df):
    df = df.dropna(subset=["Review"]).reset_index(drop=True)
    df["Review"] = df["Review"].apply(clean_text)
    df["Complaint_bin"] = df["Complaint"].apply(map_complaint)

    le_sent = LabelEncoder()
    le_emo = LabelEncoder()

    df["Sentiment_label"] = le_sent.fit_transform(df["Sentiment"].astype(str))
    df["Emotion_label"] = le_emo.fit_transform(df["Emotion"].astype(str))

    pickle.dump(
        {"sent": le_sent, "emo": le_emo},
        open(ENCODER_SAVE_PATH, "wb")
    )
    return df
