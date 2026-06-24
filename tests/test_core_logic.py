import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Install project requirements to run core logic tests") from exc

from src.candidate_generation import get_topk_candidates
from src.diversity import mmr_rerank
from src.mind_dataset import resolve_mindsmall_paths
from src.user_profile import build_user_vector_from_history


class CandidateGenerationTests(unittest.TestCase):
    def test_returns_top_candidates_and_excludes_read_items(self):
        user_vec = np.array([1.0, 0.0])
        news_emb = np.array(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
            ]
        )
        out = get_topk_candidates(
            user_vec,
            news_emb,
            ["N1", "N2", "N3"],
            k=2,
            exclude_set={"N1"},
        )

        self.assertEqual([row[0] for row in out], ["N2", "N3"])

    def test_rejects_non_positive_k(self):
        with self.assertRaises(ValueError):
            get_topk_candidates(np.array([1.0]), np.array([[1.0]]), ["N1"], k=0)


class UserProfileTests(unittest.TestCase):
    def test_builds_normalized_average_user_vector(self):
        news_meta = pd.DataFrame({"news_id": ["N1", "N2", "N3"]})
        news_emb = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        user_history = {"U1": {"N1", "N2"}}

        vec = build_user_vector_from_history("U1", user_history, news_emb, news_meta)

        expected = np.array([1.0, 1.0]) / np.sqrt(2)
        np.testing.assert_allclose(vec, expected, atol=1e-8)

    def test_returns_none_for_unknown_user(self):
        news_meta = pd.DataFrame({"news_id": ["N1"]})
        news_emb = np.array([[1.0, 0.0]])

        self.assertIsNone(build_user_vector_from_history("missing", {}, news_emb, news_meta))


class DiversityTests(unittest.TestCase):
    def test_mmr_returns_requested_number_of_unique_items(self):
        query = np.array([1.0, 0.0])
        docs = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ]
        )

        out = mmr_rerank(query, docs, ["N1", "N2", "N3"], top_k=2, lambda_param=0.7)

        self.assertEqual(len(out), 2)
        self.assertEqual(len({news_id for news_id, _ in out}), 2)

    def test_mmr_rejects_invalid_lambda(self):
        with self.assertRaises(ValueError):
            mmr_rerank(np.array([1.0]), np.array([[1.0]]), ["N1"], lambda_param=1.5)


class MindDatasetPathTests(unittest.TestCase):
    def test_resolves_direct_google_drive_layout(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "news.tsv").write_text("", encoding="utf-8")
            (root / "behaviors.tsv").write_text("", encoding="utf-8")

            paths = resolve_mindsmall_paths(root)

            self.assertEqual(paths.root, root)
            self.assertEqual(paths.news_path, root / "news.tsv")
            self.assertEqual(paths.behaviors_path, root / "behaviors.tsv")

    def test_resolves_nested_mindsmall_train_layout(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_dir = root / "MINDsmall_train"
            train_dir.mkdir()
            (train_dir / "news.tsv").write_text("", encoding="utf-8")
            (train_dir / "behaviors.tsv").write_text("", encoding="utf-8")

            paths = resolve_mindsmall_paths(root)

            self.assertEqual(paths.root, train_dir)


if __name__ == "__main__":
    unittest.main()
