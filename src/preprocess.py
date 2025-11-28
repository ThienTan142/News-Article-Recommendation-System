import pandas as pd
import os

TRAIN_NEWS = "data/raw/MINDsmall_train/news.tsv"
TRAIN_BEHAVIORS = "data/raw/MINDsmall_train/behaviors.tsv"

VAL_NEWS = "data/raw/MINDsmall_dev/news.tsv"
VAL_BEHAVIORS = "data/raw/MINDsmall_dev/behaviors.tsv"


def load_news(path):
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=[
            "news_id", "category", "subcategory",
            "title", "abstract", "url",
            "title_entities", "abstract_entities"
        ]
    )

def load_behaviors(path):
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "time", "history", "impressions"]
    )

def build_news_master():
    train_news = load_news(TRAIN_NEWS)
    val_news = load_news(VAL_NEWS)

    news = pd.concat([train_news, val_news]).drop_duplicates("news_id")
    os.makedirs("data/processed", exist_ok=True)
    news.to_csv("data/processed/news_master.csv", index=False)

    print("✔ Saved: data/processed/news_master.csv")

def build_interactions(behaviors, output_path):
    rows = []

    for _, row in behaviors.iterrows():
        # impressions dạng "N12345-1 N67893-0 ..."
        items = str(row["impressions"]).split()
        for item in items:
            nid, clicked = item.split("-")
            rows.append([row["user_id"], nid, int(clicked)])

    df = pd.DataFrame(rows, columns=["user_id", "news_id", "clicked"])
    df.to_csv(output_path, index=False)

    print(f"✔ Saved: {output_path}")
def process_behaviors():
    os.makedirs("data/processed", exist_ok=True)

    train_beh = load_behaviors(TRAIN_BEHAVIORS)
    val_beh = load_behaviors(VAL_BEHAVIORS)

    build_interactions(train_beh, "data/processed/interactions_train.csv")
    build_interactions(val_beh, "data/processed/interactions_val.csv")
if __name__ == "__main__":
    build_news_master()
    process_behaviors()

