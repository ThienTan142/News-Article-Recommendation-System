import re
import pandas as pd

def advanced_clean(text):
    text = str(text)

    # Remove wikidata patterns: wikidataid q12345
    text = re.sub(r"wikidataid\s+q\d+", " ", text)

    # Remove words like "label something"
    text = re.sub(r"label\s+\w+", " ", text)

    # Remove "confidence N" patterns
    text = re.sub(r"confidence\s+\d+", " ", text)

    # Remove occurrenceoffsets
    text = re.sub(r"occurrenceoffsets\s+\d+", " ", text)

    # Remove extra numbers
    text = re.sub(r"\b\d+\b", " ", text)

    # Remove repeated spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("[INFO] Loading news_master...")
    df = pd.read_csv("data/processed/news_master.csv")

    # Tạo clean_text từ các cột hiện có
    print("[INFO] Creating clean_text...")

    # Kiểm tra các cột có sẵn
    text_cols = []
    for col in ["title", "summary", "content", "short_description"]:
        if col in df.columns:
            text_cols.append(col)

    if len(text_cols) == 0:
        raise ValueError("Không tìm thấy cột nội dung nào để tạo clean_text!")

    print(f"[INFO] Using columns: {text_cols}")

    # Ghép các cột text thành một đoạn
    df["clean_text"] = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

    print("[INFO] Applying advanced_clean...")
    df["clean_text"] = df["clean_text"].apply(advanced_clean)

    df.to_csv("data/processed/news_clean.csv", index=False)
    print("[DONE] Cleaning completed! File saved: data/processed/news_clean.csv")


if __name__ == "__main__":
    main()
