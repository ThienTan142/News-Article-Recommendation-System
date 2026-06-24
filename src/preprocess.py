import pandas as pd

from src.artifacts import ensure_file, require_columns


def load_news(news_path):
    """
    Load MIND news.tsv into DataFrame with columns:
    news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities
    """
    news_file = ensure_file(news_path, "MIND news file")
    df = pd.read_csv(news_file, sep="\t", header=None,
                     names=["news_id","category","subcategory","title","abstract","url","title_entities","abstract_entities"],
                     quoting=3, dtype=str).fillna("")
    require_columns(df, ["news_id", "title", "abstract"], "MIND news file")
    # create a single text field
    df["text"] = df["title"].astype(str) + ". " + df["abstract"].astype(str)
    return df

def load_behaviors(beh_path):
    behavior_file = ensure_file(beh_path, "MIND behaviors file")
    df = pd.read_csv(behavior_file, sep="\t", header=None,
                     names=["impression_id","user_id","time","history","impressions"],
                     quoting=3, dtype=str).fillna("")
    require_columns(df, ["user_id", "history", "impressions"], "MIND behaviors file")
    return df
