# scripts/preprocess_news_robust.py
import pandas as pd
import os
import re

INPUT = "data/processed/news_master.csv"
OUTPUT = "data/processed/news_clean.csv"

# các tên cột khả dĩ chứa text (theo thứ tự ưu tiên)
POSSIBLE_TEXT_COLS = [
    "title", "headline", "name",
    "description", "summary", "abstract",
    "content", "body", "text", "full_text",
    "article", "detail"
]

# các cột id khả dĩ
POSSIBLE_ID_COLS = ["news_id", "id", "article_id", "nid"]

def clean_text_fn(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)               # remove links
    text = re.sub(r"[^a-z0-9\s]", " ", text)           # keep alphanum + spaces
    text = re.sub(r"\s+", " ", text).strip()           # normalize spaces
    return text

def find_id_column(df):
    for c in POSSIBLE_ID_COLS:
        if c in df.columns:
            return c
    # fallback: use first column
    return df.columns[0]

def find_text_columns(df):
    found = []
    for c in POSSIBLE_TEXT_COLS:
        if c in df.columns:
            found.append(c)
    # also include any column that likely contains long text by dtype/object and avg length > 20
    for c in df.columns:
        if c not in found and df[c].dtype == object:
            # sample few values safely
            sample = df[c].dropna().astype(str).head(50).tolist()
            avg_len = sum(len(s) for s in sample) / (len(sample) if sample else 1)
            if avg_len > 30:
                found.append(c)
    return found

def main():
    if not os.path.exists(INPUT):
        print(f"[ERROR] Input file not found: {INPUT}")
        return

    print("[INFO] Loading:", INPUT)
    df = pd.read_csv(INPUT)

    print("[INFO] Columns in file:")
    for i, c in enumerate(df.columns):
        print(f"  {i+1}. {c}")

    id_col = find_id_column(df)
    text_cols = find_text_columns(df)

    print(f"[INFO] Using id column: {id_col}")
    if text_cols:
        print("[INFO] Detected text columns (will be concatenated):", text_cols)
    else:
        print("[WARN] No obvious text columns detected. Will try concatenating all string columns.")
        # fallback: use all object dtype columns
        text_cols = [c for c in df.columns if df[c].dtype == object]
        print("[INFO] Fallback text columns:", text_cols)

    # create combined text
    def combine_row(r):
        parts = []
        for c in text_cols:
            val = r.get(c, "")
            if pd.isna(val):
                continue
            s = str(val).strip()
            if s:
                parts.append(s)
        return " ".join(parts)

    print("[INFO] Combining text columns into 'combined_text' (may take a while)...")
    df["combined_text"] = df.apply(combine_row, axis=1)

    print("[INFO] Cleaning text...")
    df["clean_text"] = df["combined_text"].apply(clean_text_fn)

    out_df = df[[id_col, "clean_text"]].rename(columns={id_col: "news_id"})

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    out_df.to_csv(OUTPUT, index=False)
    print(f"[DONE] Saved cleaned news to: {OUTPUT}")
    print("[SAMPLE]")
    print(out_df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
