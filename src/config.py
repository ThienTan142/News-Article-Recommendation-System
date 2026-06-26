from dataclasses import dataclass
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MIND_TRAIN_DIR = DATA_DIR / "MINDsmall" / "MINDsmall_train"
PRECOMPUTE_DIR = DATA_DIR / "precompute"
MODELS_DIR = ROOT_DIR / "models"


@dataclass(frozen=True)
class ProjectPaths:
    mind_train_dir: Path = MIND_TRAIN_DIR
    news_path: Path = MIND_TRAIN_DIR / "news.tsv"
    behaviors_path: Path = MIND_TRAIN_DIR / "behaviors.tsv"
    precompute_dir: Path = PRECOMPUTE_DIR
    news_embeddings_path: Path = PRECOMPUTE_DIR / "news_embeddings.npy"
    news_metadata_path: Path = PRECOMPUTE_DIR / "news_metadata.csv"
    user_history_path: Path = PRECOMPUTE_DIR / "user_history.json"
    legacy_user_history_pickle_path: Path = PRECOMPUTE_DIR / "user_history.pkl"
    ctr_dataset_path: Path = PRECOMPUTE_DIR / "ctr_dataset.csv"
    precompute_manifest_path: Path = PRECOMPUTE_DIR / "manifest.json"
    ctr_model_path: Path = MODELS_DIR / "ctr_model.pt"
    training_report_path: Path = MODELS_DIR / "training_report.json"


PATHS = ProjectPaths()

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_K = 200
DEFAULT_MMR_LAMBDA = 0.7

BATCH_SIZE = 256
EPOCHS = 8
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
VALIDATION_SIZE = 0.15
DEVICE = os.getenv("NEWS_REC_DEVICE", "cpu")
