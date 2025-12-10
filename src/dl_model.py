# src/dl_model.py
import torch
import torch.nn as nn

class CTR_MLP(nn.Module):
    def __init__(self, emb_dim, hidden=[256,128], dropout=0.2):
        super().__init__()
        input_dim = emb_dim * 2
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, user_emb, item_emb):
        # user_emb/item_emb: (batch, emb_dim)
        x = torch.cat([user_emb, item_emb], dim=1)
        logits = self.net(x).squeeze(1)
        probs = torch.sigmoid(logits)
        return probs
