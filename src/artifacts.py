from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class ProjectArtifactError(RuntimeError):
    """Raised when a required project artifact is missing or invalid."""


class MissingArtifactError(ProjectArtifactError):
    """Raised when a required artifact file cannot be found."""


class InvalidArtifactError(ProjectArtifactError):
    """Raised when an artifact exists but has an invalid schema or shape."""


def ensure_file(path: str | Path, purpose: str, producer_hint: str = "") -> Path:
    resolved = Path(path)
    if not resolved.exists():
        hint = f" Run {producer_hint} first." if producer_hint else ""
        raise MissingArtifactError(f"Missing {purpose}: {resolved}.{hint}")
    if not resolved.is_file():
        raise InvalidArtifactError(f"Expected {purpose} to be a file: {resolved}")
    return resolved


def require_columns(frame: pd.DataFrame, columns: Iterable[str], artifact_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise InvalidArtifactError(
            f"{artifact_name} is missing required columns: {', '.join(missing)}"
        )


def validate_news_artifacts(news_emb: np.ndarray, news_meta: pd.DataFrame) -> None:
    if news_emb.ndim != 2:
        raise InvalidArtifactError(
            f"news embeddings must be a 2D matrix, got shape {news_emb.shape}"
        )
    require_columns(news_meta, ["news_id"], "news metadata")
    if len(news_meta) != news_emb.shape[0]:
        raise InvalidArtifactError(
            "news metadata row count must match embedding rows: "
            f"{len(news_meta)} rows vs {news_emb.shape[0]} embeddings"
        )
    if news_meta["news_id"].astype(str).duplicated().any():
        raise InvalidArtifactError("news metadata contains duplicate news_id values")


def build_id_to_index(news_meta: pd.DataFrame) -> dict[str, int]:
    require_columns(news_meta, ["news_id"], "news metadata")
    return {str(news_id): index for index, news_id in enumerate(news_meta["news_id"])}
