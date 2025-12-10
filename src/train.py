# src/train.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from joblib import dump

from src.embedding import load_news_embeddings
from src.user_profile import load_user_history, build_user_vector_from_history
from src.dl_model import CTR_MLP

# Config
CTR_SAMPLES = "data/precompute/ctr_dataset.csv"
NEWS_EMB = "data/precompute/news_embeddings.npy"
NEWS_META = "data/precompute/news_metadata.csv"
USER_HISTORY = "data/precompute/user_history.pkl"
OUT_MODEL = "models/ctr_model.pt"

BATCH_SIZE = 256
EPOCHS = 8
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------- Dataset -----------------------
class CTRDataset(Dataset):
    def __init__(self, df, news_emb, news_meta, user_history):
        self.df = df.reset_index(drop=True)
        self.news_emb = news_emb
        self.news_meta = news_meta
        self.user_history = user_history

        self.id2idx = {nid: i for i, nid in enumerate(news_meta["news_id"].astype(str).tolist())}

        # Pre-build sample list
        self.rows = [
            (r["user_id"], r["news_id"], int(r["label"]))
            for _, r in df.iterrows()
            if str(r["news_id"]) in self.id2idx
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        uid, nid, label = self.rows[idx]
        item_idx = self.id2idx[nid]

        item_vec = self.news_emb[item_idx]

        # Build user vector
        user_vec = build_user_vector_from_history(uid, self.user_history, self.news_emb, self.news_meta)
        if user_vec is None:
            user_vec = self.news_emb.mean(axis=0)

        return (
            user_vec.astype("float32"),
            item_vec.astype("float32"),
            np.float32(label),
        )


def collate_fn(batch):
    users = np.stack([b[0] for b in batch])
    items = np.stack([b[1] for b in batch])
    labels = np.array([b[2] for b in batch])
    return torch.tensor(users), torch.tensor(items), torch.tensor(labels)


# ----------------------- Trainer -----------------------
def train():
    os.makedirs("models", exist_ok=True)

    print("📥 Loading embeddings + metadata...")
    news_emb, news_meta = load_news_embeddings()
    user_hist = load_user_history()

    df = pd.read_csv(CTR_SAMPLES)

    # Optional subsample
    if len(df) > 150000:
        df = df.sample(n=150000, random_state=42).reset_index(drop=True)
        print("⚠ Using subsample 150k for speed")

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

    train_ds = CTRDataset(train_df, news_emb, news_meta, user_hist)
    val_ds = CTRDataset(val_df, news_emb, news_meta, user_hist)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    emb_dim = news_emb.shape[1]
    model = CTR_MLP(emb_dim).to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("\n🚀 Start training...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [TRAIN]", colour="green")
        total_loss = 0

        for users, items, labels in train_bar:
            users, items, labels = users.to(DEVICE), items.to(DEVICE), labels.to(DEVICE)

            preds = model(users, items)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            train_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"🟢 Train Loss = {avg_loss:.4f}")

        # Validation
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [VAL]", colour="yellow")
            for users, items, labels in val_bar:
                users, items = users.to(DEVICE), items.to(DEVICE)
                out = model(users, items).cpu().numpy()

                val_preds.extend(out.tolist())
                val_labels.extend(labels.numpy().tolist())

        from sklearn.metrics import roc_auc_score

        try:
            auc = roc_auc_score(val_labels, val_preds)
        except:
            auc = 0.0

        print(f"🔵 Validation AUC = {auc:.4f}\n")

    torch.save(model.state_dict(), OUT_MODEL)
    print("🎉 Model saved at:", OUT_MODEL)


if __name__ == "__main__":
    train()
