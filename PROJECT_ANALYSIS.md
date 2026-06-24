# Project Analysis: News-Article-Recommendation-System
**Analyzed:** 2026-06-17
**Current direction:** MINDsmall CLI/offline recommendation pipeline

## 1. Project Overview
This project is now oriented around a CLI/offline ML pipeline for MINDsmall news recommendation. Streamlit demo UI has been removed; the primary flow is dataset preparation, artifact precompute, optional CTR training, then CLI recommendation output.

## 2. Current Structure
`.gitignore` -> Ignores local data, model artifacts, Python caches, and build metadata.
`README.md` -> Main usage guide for MINDsmall layout, precompute, training, and CLI recommendation.
`requirements.txt` -> Runtime dependencies for data processing, embeddings, PyTorch model training, and CLI utilities.
`setup.py` -> Package metadata and `news-recommend` console script entry point.
`demo-ui/` -> Static browser demo for presenting CLI recommendation JSON without Streamlit or backend calls.
`demo-ui/index.html` -> Main demo interface with pipeline status, JSON input, results, and explanation panel.
`demo-ui/styles.css` -> Responsive visual system for the static demo.
`demo-ui/demo.js` -> Sample data, JSON parsing, result rendering, copy command, and mode switching.
`data/README.md` -> Documents expected MINDsmall raw data layouts and generated artifacts.
`models/README.md` -> Documents local CTR model artifact strategy.
`scripts/precompute_news.py` -> Resolves MINDsmall `news.tsv`, generates article embeddings and metadata.
`scripts/precompute_user_history.py` -> Resolves MINDsmall `behaviors.tsv`, generates user history JSON.
`src/config.py` -> Central path, hyperparameter, and device defaults.
`src/artifacts.py` -> Artifact validation helpers and custom missing/invalid artifact errors.
`src/mind_dataset.py` -> Resolves common MINDsmall train layouts, including Google/Drive extracted folders.
`src/preprocess.py` -> Reads MIND news and behavior TSV files.
`src/embedding.py` -> Loads and validates precomputed news embeddings and metadata.
`src/user_profile.py` -> Loads user history and builds normalized user vectors.
`src/candidate_generation.py` -> Generates cosine-similarity top-K candidates.
`src/ranking.py` -> Loads cached CTR MLP ranker and scores candidates.
`src/diversity.py` -> Applies MMR diversification.
`src/dl_model.py` -> Defines the CTR MLP model.
`src/ctr_dataset.py` -> Builds CTR training samples from behavior impressions.
`src/train.py` -> Trains the CTR MLP and saves `models/ctr_model.pt`.
`src/run_recommend_cli.py` -> CLI entry point for recommendations.
`tests/test_core_logic.py` -> Unit tests for candidate generation, user profile, MMR, and MINDsmall path resolution.

## 3. Data Flow
1. Place MINDsmall train files in one supported layout, preferably `data/MINDsmall/MINDsmall_train/news.tsv` and `data/MINDsmall/MINDsmall_train/behaviors.tsv`.
2. `src.mind_dataset.resolve_mindsmall_paths()` finds the raw files. You can override with `--mind-dir`.
3. `scripts/precompute_news.py` creates `news_embeddings.npy`, `news_metadata.csv`, and `manifest.json`.
4. `scripts/precompute_user_history.py` creates `user_history.json`.
5. `src.ctr_dataset.build_ctr_samples(...)` creates `ctr_dataset.csv` for CTR training.
6. `python -m src.train` trains the CTR model and writes `models/ctr_model.pt`.
7. `python -m src.run_recommend_cli --user <id>` loads artifacts, builds the user vector, retrieves candidates, optionally reranks by CTR, applies MMR, and prints results.

## 4. Algorithms Implemented
- **Content-based embeddings** -> SentenceTransformer article embeddings from title plus abstract.
- **User profile averaging** -> Mean of historical article embeddings for known users; mean news embedding for cold start.
- **Cosine candidate retrieval** -> Top-K semantic similarity over article embeddings.
- **CTR MLP reranking** -> PyTorch MLP over concatenated user/item embeddings.
- **MMR diversification** -> Balances relevance with diversity in the final ranked list.
- **Negative sampling** -> Random seeded negative sampling from non-clicked impressions for CTR training.

## 5. Current Gaps
- Real recommendation output still requires local MINDsmall data and generated artifacts.
- CTR dataset generation is still invoked via Python one-liner rather than a dedicated script.
- Model training does not yet persist `metrics.json` or `model_config.json`.
- No API service, Dockerfile, or CI pipeline yet.
- Approximate nearest-neighbor retrieval is not implemented; candidate generation still scans all embeddings.
- Historical audit docs may mention the old Streamlit state, but active source/dependencies/README are now CLI-first.
- Demo UI is static and presentational; it does not call the Python CLI or backend automatically.

## 6. Update Roadmap
### Priority 1 — Critical
- Add a dedicated `scripts/build_ctr_dataset.py` -> makes the train pipeline fully command-driven.
- Add tiny fixture integration test -> proves precompute/artifact/recommend contracts without heavy model download.
- Persist training metrics/config -> makes model outputs reproducible and comparable.

### Priority 2 — Important
- Add ANN retrieval option -> improves candidate generation when article corpus grows.
- Add `src/evaluate.py` -> report Recall@K, NDCG@K, MRR, and cosine baseline comparison.
- Add CI with compile + unit tests -> protects future edits.

### Priority 3 — Nice to have
- Add FastAPI service only if integration is needed -> CLI remains the primary path for now.
- Add Dockerfile after artifact paths are stable.
- Add DVC/Git LFS/release artifact policy for model and data versioning.
