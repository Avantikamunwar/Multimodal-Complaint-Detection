import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["Review"].tolist()
        self.sentiments = df["Sentiment_label"].tolist()
        self.emotions = df["Emotion_label"].tolist()
        self.complaints = df["Complaint_bin"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "sentiment": torch.tensor(self.sentiments[idx]),
            "emotion": torch.tensor(self.emotions[idx]),
            "complaint": torch.tensor(self.complaints[idx]),
        }

    def __len__(self):
        return len(self.texts)
