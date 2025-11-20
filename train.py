import torch
import numpy as np
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch import optim, nn
import pickle
import pandas as pd

from config.config import *
from src.dataset import TextDataset
from src.model import MultiTaskModel
from src.evaluate import evaluate
from src.data_preprocess import preprocess_dataset

def seed_everything(seed):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed_everything(SEED)

def train():
    df = pd.read_csv(DATA_PATH)
    df = preprocess_dataset(df)

    train_df = df.sample(frac=0.85, random_state=SEED)
    val_df = df.drop(train_df.index)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    train_ds = TextDataset(train_df, tokenizer, MAX_LENGTH)
    val_ds = TextDataset(val_df, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = MultiTaskModel(
        num_sent=df["Sentiment_label"].nunique(),
        num_emo=df["Emotion_label"].nunique(),
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * NUM_EPOCHS
    )

    criterion = nn.CrossEntropyLoss()
    best = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        for b in train_loader:
            optimizer.zero_grad()
            ids = b["input_ids"].to(DEVICE)
            att = b["attention_mask"].to(DEVICE)
            y_s = b["sentiment"].to(DEVICE)
            y_e = b["emotion"].to(DEVICE)
            y_c = b["complaint"].to(DEVICE)

            out = model(ids, att)

            loss = (
                criterion(out["sentiment"], y_s) +
                criterion(out["emotion"], y_e) +
                criterion(out["complaint"], y_c)
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

        metrics = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch+1} → {metrics}")

        if metrics["combined"] > best:
            best = metrics["combined"]
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Saved best model!")

if __name__ == "__main__":
    train()
