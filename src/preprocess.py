# src/preprocess.py
import pandas as pd

def load_news(news_path):
    """
    Load MIND news.tsv into DataFrame with columns:
    news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities
    """
    df = pd.read_csv(news_path, sep="\t", header=None,
                     names=["news_id","category","subcategory","title","abstract","url","title_entities","abstract_entities"],
                     quoting=3, dtype=str).fillna("")
    # create a single text field
    df["text"] = df["title"].astype(str) + ". " + df["abstract"].astype(str)
    return df

def load_behaviors(beh_path):
    df = pd.read_csv(beh_path, sep="\t", header=None,
                     names=["impression_id","user_id","time","history","impressions"],
                     quoting=3, dtype=str).fillna("")
    return df
