# Roadmap Update Chi Tiet: News-Article-Recommendation-System

**Ngay cap nhat:** 2026-06-17
**Huong hien tai:** CLI/offline pipeline cho MINDsmall dataset. Streamlit demo da duoc bo.

## 1. Trang Thai Hien Tai

Project hien tap trung vao pipeline:

1. Doc MINDsmall `news.tsv` va `behaviors.tsv`.
2. Tao embedding cho news.
3. Tao user history.
4. Tao CTR dataset.
5. Train CTR MLP.
6. Recommend bang CLI.

## 2. Nhung Phan Da Chuyen Huong

### Bo Streamlit demo

**Da bo:**
- `app.py`
- `streamlit` trong `requirements.txt`
- Huong dan `streamlit run app.py` trong README

**Ly do:** User muon dung MINDsmall/offline pipeline, khong can UI demo.

### Bo sung MINDsmall resolver

**File:** `src/mind_dataset.py`

**Ho tro cac layout:**

```text
data/MINDsmall/MINDsmall_train/news.tsv
data/MINDsmall/MINDsmall_train/behaviors.tsv

data/MINDsmall_train/news.tsv
data/MINDsmall_train/behaviors.tsv

data/MINDsmall/train/news.tsv
data/MINDsmall/train/behaviors.tsv
```

Hoac truyen:

```powershell
python scripts/precompute_news.py --mind-dir "F:\path\to\MINDsmall"
python scripts/precompute_user_history.py --mind-dir "F:\path\to\MINDsmall"
```

### Them static demo UI

**File:** `demo-ui/index.html`, `demo-ui/styles.css`, `demo-ui/demo.js`

**Vai tro:** Trinh bay recommendation output cho demo bao cao ma khong can Streamlit.

**Cach dung:**

```text
Mo demo-ui/index.html trong trinh duyet
```

UI co the load sample result hoac paste JSON tu:

```powershell
python -m src.run_recommend_cli --user U8125 --topk 10 --json
```

## 3. Roadmap Tiep Theo

### Phase 1 - Hoan thien CLI pipeline

#### Task 1.1: Them script build CTR dataset

**Lam gi:** Tao `scripts/build_ctr_dataset.py` de thay cho Python one-liner trong README.

**Vi sao:** Pipeline se co du 4 lenh ro rang: precompute news, precompute history, build CTR dataset, train.

**Tieu chi xong:**
- Chay duoc `python scripts/build_ctr_dataset.py`.
- Co args `--mind-dir`, `--neg-ratio`, `--out-csv`.
- In class balance.

#### Task 1.2: Them artifact check command

**Lam gi:** Tao script `scripts/check_artifacts.py`.

**Vi sao:** Truoc khi recommend/train, user biet thieu artifact nao.

**Tieu chi xong:**
- Bao OK/thieu cho `news_embeddings.npy`, `news_metadata.csv`, `user_history.json`, `ctr_dataset.csv`, `ctr_model.pt`.

### Phase 2 - ML evaluation

#### Task 2.1: Luu train metrics

**Lam gi:** `src/train.py` ghi `models/metrics.json` va `models/model_config.json`.

**Vi sao:** De so sanh model giua cac lan train.

#### Task 2.2: Them ranking metrics

**Lam gi:** Tao `src/evaluate.py` tinh Recall@K, NDCG@K, MRR.

**Vi sao:** AUC khong du de danh gia chat luong top-K recommendation.

### Phase 3 - Performance

#### Task 3.1: Toi uu candidate retrieval

**Lam gi:** Dung partial top-K hoac ANN index.

**Vi sao:** Full scan embedding se cham khi corpus lon.

#### Task 3.2: Cache/precompute user vectors

**Lam gi:** Tao artifact user vectors tu user history.

**Vi sao:** Training hien co the lap lai user-vector computation nhieu lan.

### Phase 4 - Engineering hygiene

#### Task 4.1: CI

**Lam gi:** Them GitHub Actions chay compile + unit tests.

#### Task 4.2: Packaging

**Lam gi:** Chuyen `setup.py` sang `pyproject.toml` khi pipeline on dinh.

#### Task 4.3: Artifact policy

**Lam gi:** Quyet dinh model/data dung Git LFS, release asset, DVC, hay regenerate.

## 4. Thu Tu Nen Lam

1. `scripts/build_ctr_dataset.py`
2. `scripts/check_artifacts.py`
3. Luu `metrics.json` va `model_config.json`
4. Ranking metrics
5. CI
6. ANN retrieval neu dataset lon
7. Docker/FastAPI neu can deploy/tich hop

## 5. Checklist Nhanh Theo File

| File/Folder | Viec tiep theo |
|---|---|
| `README.md` | Cap nhat khi them script build CTR dataset |
| `src/mind_dataset.py` | Them layout moi neu Google Drive dataset cua ban co cau truc khac |
| `scripts/precompute_news.py` | OK cho `--mind-dir` va `--news-path` |
| `scripts/precompute_user_history.py` | OK cho `--mind-dir` va `--behaviors-path` |
| `src/ctr_dataset.py` | Nen co CLI wrapper |
| `src/train.py` | Luu metrics/config |
| `src/run_recommend_cli.py` | CLI chinh cua project |
| `tests/` | Them integration test nho |
