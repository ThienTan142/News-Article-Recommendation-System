import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.artifacts import ProjectArtifactError
from src.config import PATHS
from src.mind_dataset import resolve_mindsmall_paths
from src.preprocess import load_behaviors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mind-dir", default=None, help="Folder containing MINDsmall_train or news.tsv/behaviors.tsv")
    parser.add_argument("--behaviors-path", default=None, help="Explicit path to MINDsmall train behaviors.tsv")
    parser.add_argument("--out-path", default=str(PATHS.user_history_path))
    return parser.parse_args()

def main():
    args = parse_args()
    out_path = Path(args.out_path)
    behaviors_path = Path(args.behaviors_path) if args.behaviors_path else resolve_mindsmall_paths(args.mind_dir).behaviors_path
    os.makedirs(out_path.parent, exist_ok=True)
    print("Load behaviors...")
    df = load_behaviors(behaviors_path)
    history = {}
    for _, r in df.iterrows():
        uid = r["user_id"]
        hist = r["history"]
        if not hist or hist == "nan": continue
        news_list = [x for x in hist.split() if x]
        if uid not in history:
            history[uid] = set()
        history[uid].update(news_list)

    serializable_history = {uid: sorted(news_ids) for uid, news_ids in history.items()}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable_history, f)
    print("Saved user_history for", len(history), "users ->", out_path)

if __name__ == "__main__":
    try:
        main()
    except ProjectArtifactError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
