# Data Directory

This directory is intentionally kept out of Git except for this README.

Expected raw MINDsmall train input:

```text
data/MINDsmall/MINDsmall_train/news.tsv
data/MINDsmall/MINDsmall_train/behaviors.tsv
```

The code also accepts common extracted Google/Drive layouts:

```text
data/MINDsmall_train/news.tsv
data/MINDsmall_train/behaviors.tsv

data/MINDsmall/train/news.tsv
data/MINDsmall/train/behaviors.tsv
```

If your dataset is outside `data/`, pass `--mind-dir <folder>` to the precompute scripts.

Expected generated artifacts:

```text
data/precompute/news_embeddings.npy
data/precompute/news_metadata.csv
data/precompute/manifest.json
data/precompute/user_history.json
data/precompute/ctr_dataset.csv
```

Generate the first three precompute artifacts with:

```powershell
python scripts/precompute_news.py
python scripts/precompute_user_history.py
```

Generate CTR training data after news embeddings exist:

```powershell
python scripts/build_ctr_dataset.py
```
