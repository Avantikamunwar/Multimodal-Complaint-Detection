import torch
from sklearn.metrics import accuracy_score

def evaluate(model, loader, device):
    model.eval()
    preds_s, preds_e, preds_c = [], [], []
    truths_s, truths_e, truths_c = [], [], []

    with torch.no_grad():
        for b in loader:
            ids = b["input_ids"].to(device)
            att = b["attention_mask"].to(device)

            out = model(ids, att)

            preds_s.extend(out["sentiment"].argmax(1).cpu().numpy())
            preds_e.extend(out["emotion"].argmax(1).cpu().numpy())
            preds_c.extend(out["complaint"].argmax(1).cpu().numpy())

            truths_s.extend(b["sentiment"].numpy())
            truths_e.extend(b["emotion"].numpy())
            truths_c.extend(b["complaint"].numpy())

    acc_sent = accuracy_score(truths_s, preds_s)
    acc_emo = accuracy_score(truths_e, preds_e)
    acc_comp = accuracy_score(truths_c, preds_c)

    return {
        "acc_sent": acc_sent,
        "acc_emo": acc_emo,
        "acc_comp": acc_comp,
        "combined": (acc_sent + acc_emo + acc_comp) / 3
    }
