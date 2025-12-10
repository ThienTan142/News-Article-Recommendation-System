# scripts/precompute_user_history.py
import os, pickle
from src.preprocess import load_behaviors

BEH_PATH = "data/MINDsmall/MINDsmall_train/behaviors.tsv"
OUT_PATH = "data/precompute/user_history.pkl"

def main():
    os.makedirs("data/precompute", exist_ok=True)
    print("Load behaviors...")
    df = load_behaviors(BEH_PATH)
    history = {}
    for _, r in df.iterrows():
        uid = r["user_id"]
        hist = r["history"]
        if not hist or hist == "nan": continue
        news_list = [x for x in hist.split() if x]
        if uid not in history:
            history[uid] = set()
        history[uid].update(news_list)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(history, f)
    print("Saved user_history for", len(history), "users ->", OUT_PATH)

if __name__ == "__main__":
    main()
