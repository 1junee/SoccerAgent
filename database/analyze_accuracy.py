from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_results(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a list of QA entries")
    return data


def summarize(entries: list[dict]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for item in entries:
        category = item.get("category", "unknown")
        summary_entry = summary[category]
        summary_entry["total"] += 1
        if item.get("answer") == item.get("closeA"):
            summary_entry["correct"] += 1
    return summary


def format_row(category: str, total: int, correct: int) -> str:
    accuracy = (correct / total * 100) if total else 0.0
    return f"{category:>5}  {correct:>3}/{total:<3}  {accuracy:6.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute accuracy per question category")
    parser.add_argument("json_path", type=Path, help="Path to the result JSON file")
    args = parser.parse_args()

    entries = load_results(args.json_path)
    summary = summarize(entries)

    print("Category  Correct  Accuracy")
    print("--------------------------------")
    for category in sorted(summary):
        stats = summary[category]
        print(format_row(category, int(stats["total"]), int(stats["correct"])))


if __name__ == "__main__":
    main()
