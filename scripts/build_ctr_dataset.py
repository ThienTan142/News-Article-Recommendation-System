import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.artifacts import ProjectArtifactError
from src.config import PATHS
from src.ctr_dataset import build_ctr_samples
from src.embedding import load_news_embeddings
from src.mind_dataset import resolve_mindsmall_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mind-dir", default=None, help="Folder containing MINDsmall_train or news.tsv/behaviors.tsv")
    parser.add_argument("--behaviors-path", default=None, help="Explicit path to MINDsmall train behaviors.tsv")
    parser.add_argument("--neg-ratio", type=int, default=4, help="Negative samples per positive impression")
    parser.add_argument("--out-csv", default=str(PATHS.ctr_dataset_path), help="Output CTR dataset CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    _, news_meta = load_news_embeddings()
    behaviors_path = (
        Path(args.behaviors_path)
        if args.behaviors_path
        else resolve_mindsmall_paths(args.mind_dir).behaviors_path
    )
    build_ctr_samples(
        news_meta,
        behaviors_path=behaviors_path,
        neg_ratio=args.neg_ratio,
        out_csv=Path(args.out_csv),
    )


if __name__ == "__main__":
    try:
        main()
    except (ProjectArtifactError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
