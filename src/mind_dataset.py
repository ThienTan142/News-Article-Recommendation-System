from dataclasses import dataclass
from pathlib import Path

from src.artifacts import MissingArtifactError
from src.config import DATA_DIR, PATHS


@dataclass(frozen=True)
class MindDatasetPaths:
    root: Path
    news_path: Path
    behaviors_path: Path


def _candidate_roots(dataset_dir: str | Path | None = None) -> list[Path]:
    if dataset_dir:
        base = Path(dataset_dir)
        return [
            base,
            base / "MINDsmall_train",
            base / "MINDsmall" / "MINDsmall_train",
        ]

    return [
        PATHS.mind_train_dir,
        DATA_DIR / "MINDsmall_train",
        DATA_DIR / "MINDsmall" / "train",
    ]


def resolve_mindsmall_paths(dataset_dir: str | Path | None = None) -> MindDatasetPaths:
    checked = []
    for root in _candidate_roots(dataset_dir):
        news_path = root / "news.tsv"
        behaviors_path = root / "behaviors.tsv"
        checked.append(root)
        if news_path.exists() and behaviors_path.exists():
            return MindDatasetPaths(
                root=root,
                news_path=news_path,
                behaviors_path=behaviors_path,
            )

    expected = "\n".join(f"- {root}" for root in checked)
    raise MissingArtifactError(
        "Missing MINDsmall train dataset. Expected both news.tsv and behaviors.tsv in one of:\n"
        f"{expected}\n"
        "If your Google/Drive download is elsewhere, pass --mind-dir <folder>."
    )
