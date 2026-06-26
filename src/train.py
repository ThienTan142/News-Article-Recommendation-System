import argparse
import os
import sys
from time import perf_counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.artifacts import ProjectArtifactError, ensure_file
from src.config import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    PATHS,
    RANDOM_SEED,
    VALIDATION_SIZE,
)
from src.user_profile import load_user_history, build_user_vector_from_history
from src.dl_model import CTR_MLP
from src.embedding import load_news_embeddings


# ----------------------- Dataset -----------------------
class CTRDataset(Dataset):
    def __init__(self, df, news_emb, news_meta, user_history):
        self.df = df.reset_index(drop=True)
        self.news_emb = news_emb
        self.news_meta = news_meta
        self.user_history = user_history

        self.id2idx = {nid: i for i, nid in enumerate(news_meta["news_id"].astype(str).tolist())}
        self.mean_user_vec = self.news_emb.mean(axis=0).astype("float32")
        self.user_vector_cache = {}

        # Pre-build sample list
        self.rows = [
            (str(r["user_id"]), str(r["news_id"]), int(r["label"]))
            for _, r in df.iterrows()
            if str(r["news_id"]) in self.id2idx
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        uid, nid, label = self.rows[idx]
        item_idx = self.id2idx[nid]

        item_vec = self.news_emb[item_idx]
        user_vec = self._get_user_vector(uid)

        return (
            user_vec.astype("float32"),
            item_vec.astype("float32"),
            np.float32(label),
        )

    def _get_user_vector(self, uid):
        if uid not in self.user_vector_cache:
            user_vec = build_user_vector_from_history(
                uid,
                self.user_history,
                self.news_emb,
                self.news_meta,
                id_to_index=self.id2idx,
            )
            if user_vec is None:
                user_vec = self.mean_user_vec
            self.user_vector_cache[uid] = user_vec.astype("float32")
        return self.user_vector_cache[uid]


def collate_fn(batch):
    users = np.stack([b[0] for b in batch])
    items = np.stack([b[1] for b in batch])
    labels = np.array([b[2] for b in batch])
    return torch.tensor(users), torch.tensor(items), torch.tensor(labels)


def build_tensor_dataset(df, news_emb, news_meta, user_history):
    id_to_index = {nid: i for i, nid in enumerate(news_meta["news_id"].astype(str).tolist())}
    working = df[["user_id", "news_id", "label"]].copy()
    working["user_id"] = working["user_id"].astype(str)
    working["news_id"] = working["news_id"].astype(str)
    working["news_idx"] = working["news_id"].map(id_to_index)
    working = working.dropna(subset=["news_idx"]).reset_index(drop=True)
    if working.empty:
        raise ValueError("CTR dataset has no trainable rows after filtering by news metadata")

    mean_user_vec = news_emb.mean(axis=0).astype("float32")
    unique_users = working["user_id"].drop_duplicates().tolist()
    user_vectors = []
    for uid in unique_users:
        user_vec = build_user_vector_from_history(
            uid,
            user_history,
            news_emb,
            news_meta,
            id_to_index=id_to_index,
        )
        if user_vec is None:
            user_vec = mean_user_vec
        user_vectors.append(user_vec.astype("float32"))

    user_to_row = {uid: i for i, uid in enumerate(unique_users)}
    user_rows = working["user_id"].map(user_to_row).to_numpy(dtype=np.int64)
    user_matrix = np.stack(user_vectors).astype("float32")
    user_tensor = torch.from_numpy(np.ascontiguousarray(user_matrix[user_rows]))

    news_rows = working["news_idx"].to_numpy(dtype=np.int64)
    item_tensor = torch.from_numpy(np.ascontiguousarray(news_emb[news_rows].astype("float32")))
    label_tensor = torch.from_numpy(
        np.ascontiguousarray(working["label"].to_numpy(dtype=np.float32))
    )
    return TensorDataset(user_tensor, item_tensor, label_tensor)


# ----------------------- Trainer -----------------------
def positive_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the CTR reranker.")
    parser.add_argument(
        "--max-rows",
        type=positive_int,
        default=None,
        help="Optional row limit for fast CPU/debug runs. Default trains on all rows.",
    )
    parser.add_argument("--epochs", type=positive_int, default=EPOCHS)
    parser.add_argument("--batch-size", type=positive_int, default=BATCH_SIZE)
    parser.add_argument("--model-path", type=Path, default=PATHS.ctr_model_path)
    parser.add_argument("--torch-threads", type=positive_int, default=None)
    parser.add_argument(
        "--lazy-dataset",
        action="store_true",
        help="Use lower-memory per-sample dataset instead of faster materialized tensors.",
    )
    return parser.parse_args(argv)


def train(
    max_rows=None,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    model_path=PATHS.ctr_model_path,
    torch_threads=None,
    lazy_dataset=False,
):
    model_path = Path(model_path)
    os.makedirs(model_path.parent, exist_ok=True)
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)
        print(f"Using torch CPU threads: {torch_threads}")
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("Loading embeddings + metadata...", flush=True)
    news_emb, news_meta = load_news_embeddings()
    user_hist = load_user_history()

    ctr_dataset_path = ensure_file(
        PATHS.ctr_dataset_path,
        "CTR dataset",
        "python scripts/build_ctr_dataset.py",
    )
    df = pd.read_csv(ctr_dataset_path)

    # Optional subsample
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"Using subsample {max_rows} for speed")

    train_df, val_df = train_test_split(df, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)

    dataset_start = perf_counter()
    if lazy_dataset:
        train_ds = CTRDataset(train_df, news_emb, news_meta, user_hist)
        val_ds = CTRDataset(val_df, news_emb, news_meta, user_hist)
    else:
        print("Materializing train/validation tensors for faster CPU training...", flush=True)
        train_ds = build_tensor_dataset(train_df, news_emb, news_meta, user_hist)
        val_ds = build_tensor_dataset(val_df, news_emb, news_meta, user_hist)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("CTR dataset has no trainable rows after filtering by news metadata")
    print(
        f"Prepared datasets: train={len(train_ds)} val={len(val_ds)} "
        f"in {perf_counter() - dataset_start:.1f}s"
    )

    collate = collate_fn if lazy_dataset else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    emb_dim = news_emb.shape[1]
    model = CTR_MLP(emb_dim).to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStart training...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [TRAIN]", colour="green")
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
        print(f"Train Loss = {avg_loss:.4f}")

        # Validation
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [VAL]", colour="yellow")
            for users, items, labels in val_bar:
                users, items = users.to(DEVICE), items.to(DEVICE)
                out = model(users, items).cpu().numpy()

                val_preds.extend(out.tolist())
                val_labels.extend(labels.numpy().tolist())

        from sklearn.metrics import roc_auc_score

        try:
            auc = roc_auc_score(val_labels, val_preds)
        except ValueError:
            auc = 0.0

        print(f"Validation AUC = {auc:.4f}\n")

    torch.save(model.state_dict(), model_path)
    print("Model saved at:", model_path)


if __name__ == "__main__":
    try:
        args = parse_args()
        train(
            max_rows=args.max_rows,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_path=args.model_path,
            torch_threads=args.torch_threads,
            lazy_dataset=args.lazy_dataset,
        )
    except (ProjectArtifactError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
