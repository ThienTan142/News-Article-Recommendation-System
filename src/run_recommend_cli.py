import argparse
import json
import sys

from src.artifacts import ProjectArtifactError
from src.config import DEFAULT_CANDIDATE_K, DEFAULT_TOP_K
from src.recommender import recommend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", "-u", required=True)
    parser.add_argument("--topk", "-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--json", action="store_true", help="Output recommendations as JSON")
    args = parser.parse_args()

    try:
        result = recommend(args.user, top_k=args.topk, candidate_k=args.candidate_k)
    except (ProjectArtifactError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if args.json:
        print(
            json.dumps(
                {
                    "ranking_source": result.ranking_source,
                    "cold_start": result.cold_start,
                    "fallback_reason": result.fallback_reason,
                    "recommendations": [item.__dict__ for item in result.recommendations],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if result.cold_start:
        print("Cold-start: using mean news embedding")
    if result.fallback_reason:
        print("CTR ranking failed, fallback to similarity:", result.fallback_reason)
    print("Top recommendations:")
    for item in result.recommendations:
        print(f"{item.news_id} | score={item.score:.4f} | title={item.title or ''}")


if __name__ == "__main__":
    main()
