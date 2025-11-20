import torch.nn as nn
from transformers import RobertaModel

class MultiTaskModel(nn.Module):
    def __init__(self, num_sent, num_emo):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        hidden = self.roberta.config.hidden_size

        self.drop = nn.Dropout(0.3)
        self.sent_head = nn.Linear(hidden, num_sent)
        self.emo_head = nn.Linear(hidden, num_emo)
        self.comp_head = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        x = out.last_hidden_state[:, 0, :]
        x = self.drop(x)

        return {
            "sentiment": self.sent_head(x),
            "emotion": self.emo_head(x),
            "complaint": self.comp_head(x)
        }
