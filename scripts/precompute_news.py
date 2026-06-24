import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DEFAULT_EMBEDDING_MODEL, PATHS
from src.artifacts import ProjectArtifactError
from src.mind_dataset import resolve_mindsmall_paths
from src.preprocess import load_news


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mind-dir", default=None, help="Folder containing MINDsmall_train or news.tsv/behaviors.tsv")
    parser.add_argument("--news-path", default=None, help="Explicit path to MINDsmall train news.tsv")
    parser.add_argument("--out-dir", default=str(PATHS.precompute_dir))
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_emb = out_dir / "news_embeddings.npy"
    out_meta = out_dir / "news_metadata.csv"
    manifest_path = out_dir / "manifest.json"
    news_path = Path(args.news_path) if args.news_path else resolve_mindsmall_paths(args.mind_dir).news_path

    os.makedirs(out_dir, exist_ok=True)
    print("Loading news...")
    df = load_news(news_path)
    texts = df["text"].tolist()
    print("Loading SBERT model:", args.model_name)
    model = SentenceTransformer(args.model_name)
    print("Encoding news...")
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # L2 normalize rows
    norms = (emb**2).sum(axis=1) ** 0.5
    norms = np.maximum(norms, 1e-12)
    emb = emb / norms[:, None]
    np.save(out_emb, emb)
    df[["news_id","title","category","text"]].to_csv(out_meta, index=False)

    manifest = {
        "embedding_model": args.model_name,
        "news_path": str(news_path),
        "news_count": int(len(df)),
        "embedding_shape": list(emb.shape),
        "news_embeddings_sha256": _sha256(out_emb),
        "news_metadata_sha256": _sha256(out_meta),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Saved:", out_emb, out_meta, manifest_path)

if __name__ == "__main__":
    try:
        main()
    except ProjectArtifactError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
