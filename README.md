# News-Article-Recommendation-System

Prototype he thong goi y bai bao dung MINDsmall dataset. Pipeline hien tai ket hop content-based retrieval bang article embeddings, user profile averaging, CTR neural reranking, va MMR diversity reranking.

Project hien da bo demo Streamlit; cach chay chinh la CLI/offline pipeline.

Xem pipeline chi tiet trong [PIPELINE.md](PIPELINE.md).

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## MINDsmall Dataset

Neu ban tai MINDsmall tu Google Drive/Colab/Kaggle mirror, giai nen sao cho co 2 file train:

```text
news.tsv
behaviors.tsv
```

Project tu dong tim cac layout pho bien sau:

```text
data/MINDsmall/MINDsmall_train/news.tsv
data/MINDsmall/MINDsmall_train/behaviors.tsv

data/MINDsmall_train/news.tsv
data/MINDsmall_train/behaviors.tsv

data/MINDsmall/train/news.tsv
data/MINDsmall/train/behaviors.tsv
```

Khuyen nghi dat du lieu theo layout dau tien:

```text
data/MINDsmall/MINDsmall_train/news.tsv
data/MINDsmall/MINDsmall_train/behaviors.tsv
```

Neu dataset cua ban nam o folder khac, truyen truc tiep:

```powershell
python scripts/precompute_news.py --mind-dir "F:\path\to\MINDsmall"
python scripts/precompute_user_history.py --mind-dir "F:\path\to\MINDsmall"
```

`--mind-dir` co the tro vao folder chua truc tiep `news.tsv`/`behaviors.tsv`, hoac folder cha co `MINDsmall_train`.

## Build Artifacts

Tao embeddings va user history:

```powershell
python scripts/precompute_news.py
python scripts/precompute_user_history.py
```

Neu muon train CTR reranker, tao `data/precompute/ctr_dataset.csv`:

```powershell
python scripts/build_ctr_dataset.py
```

Sau do train:

```powershell
python -m src.train
```

Neu chi can smoke test nhanh pipeline tren CPU, dung tap con nho:

```powershell
python -m src.train --max-rows 5000 --epochs 1 --batch-size 512
```

## Train on Google Colab

Neu muon train nhanh hon bang GPU Colab, dung notebook:

```text
notebooks/colab_train.ipynb
```

Trong Colab, doi runtime sang GPU, sua bien `MIND_DIR` toi folder MINDsmall tren Google Drive, roi chay cac cell tu tren xuong duoi. Notebook se clone repo, tao virtualenv sach tai `/content/news-rec-venv`, cai dependencies, build artifact, train CTR reranker, chay recommendation mau, va copy artifact ve Google Drive.

Neu gap loi `numpy.dtype size changed`, chon `Runtime` -> `Disconnect and delete runtime`, mo lai notebook moi, va chay lai tu dau. Cell cai dependencies phai in ra Python path la:

```text
/content/news-rec-venv/bin/python
```

Mo truc tiep tren Colab:

```text
https://colab.research.google.com/github/ThienTan142/News-Article-Recommendation-System/blob/codex/mindsmall-pipeline-cleanup/notebooks/colab_train.ipynb
```

## Run Recommendations

Chay CLI:

```powershell
python -m src.run_recommend_cli --user U8125 --topk 10
```

Output JSON:

```powershell
python -m src.run_recommend_cli --user U8125 --topk 10 --json
```

## Demo UI

Project khong dung Streamlit. Demo UI hien la prototype web tinh trong `demo-ui/`.

Mo file sau trong trinh duyet:

```text
demo-ui/index.html
```

UI nay cho phep:

- Load sample recommendation result de demo nhanh.
- Paste JSON output tu lenh CLI `--json`.
- Trinh bay ranking source, cold-start status, scores, article cards, va pipeline explanation.

Khong can build tool, khong can server, khong goi backend.

## Project Structure

```text
demo-ui/                         Static UX/UI prototype for demo presentation
src/config.py                     Central paths and hyperparameters
src/artifacts.py                  Artifact validation and custom errors
src/mind_dataset.py               MINDsmall dataset path resolver
src/recommender.py                Shared recommendation service for CLI/API use
src/preprocess.py                 MIND news/behavior readers
src/embedding.py                  Precomputed news embedding loader
src/user_profile.py               User history loader and user vector builder
src/candidate_generation.py       Cosine-similarity candidate retrieval
src/ranking.py                    CTR model loading and reranking
src/diversity.py                  MMR diversity reranking
src/dl_model.py                   PyTorch CTR MLP model
src/ctr_dataset.py                CTR sample generation from impressions
src/train.py                      CTR model training script with configurable quick runs
src/run_recommend_cli.py          CLI entry point
scripts/precompute_news.py        Offline article embedding generation
scripts/precompute_user_history.py Offline user history generation
scripts/build_ctr_dataset.py      CTR training dataset generation
models/                           Local trained model artifacts
data/                             Local raw/precomputed data, not committed
```

## Data Flow

1. `src.mind_dataset.resolve_mindsmall_paths()` locates MINDsmall train files.
2. `src/preprocess.py` reads raw MIND `news.tsv` and `behaviors.tsv`.
3. `scripts/precompute_news.py` creates `news_embeddings.npy`, `news_metadata.csv`, and `manifest.json`.
4. `scripts/precompute_user_history.py` creates `user_history.json`.
5. `src/ctr_dataset.py` can build `ctr_dataset.csv` from impression labels.
6. `src/train.py` trains `models/ctr_model.pt`.
7. `src/recommender.py` loads artifacts, builds a user vector, retrieves candidates, reranks with CTR when available, then applies MMR.
8. `src/run_recommend_cli.py` prints recommendation results.

## Expected Artifacts

| Artifact | Produced by | Consumed by |
|---|---|---|
| `data/precompute/news_embeddings.npy` | `scripts/precompute_news.py` | CLI, training |
| `data/precompute/news_metadata.csv` | `scripts/precompute_news.py` | CLI, training |
| `data/precompute/manifest.json` | `scripts/precompute_news.py` | humans/review |
| `data/precompute/user_history.json` | `scripts/precompute_user_history.py` | CLI, training |
| `data/precompute/ctr_dataset.csv` | `src.ctr_dataset.build_ctr_samples` | training |
| `models/ctr_model.pt` | `python -m src.train` | CTR reranking |

## Notes

- `data/` and generated model artifacts are local outputs and are ignored by Git.
- Legacy `user_history.pkl` can still be loaded if present, but JSON is preferred because pickle should only be used for trusted local artifacts.
- If the CTR model is missing, the recommender can fall back to similarity ranking and reports the fallback reason.
