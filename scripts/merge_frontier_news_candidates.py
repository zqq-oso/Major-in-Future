from __future__ import annotations

import argparse
import csv
from pathlib import Path


BASE_FIELDS = ["时间", "领域", "事件类别", "新闻标题", "新闻摘要", "原文链接"]


def read_existing(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def candidate_to_base(row: dict[str, str]) -> dict[str, str]:
    return {
        "时间": row["date"],
        "领域": row["domain"],
        "事件类别": row["category"],
        "新闻标题": row["title_zh"],
        "新闻摘要": row["summary_zh"],
        "原文链接": row["source_url"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge reviewed frontier-news candidates into a new CSV.")
    parser.add_argument("--base", default="data/base_seed_news.csv")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-status",
        default="candidate_verified_primary",
        help="Comma-separated review_status values allowed into the merged output.",
    )
    args = parser.parse_args()

    base_path = Path(args.base)
    candidates_path = Path(args.candidates)
    output_path = Path(args.output)
    allowed = {item.strip() for item in args.allow_status.split(",") if item.strip()}

    existing = read_existing(base_path)
    existing_keys = {
        (row.get("新闻标题", "").strip(), row.get("原文链接", "").strip())
        for row in existing
    }
    existing_titles = {row.get("新闻标题", "").strip() for row in existing}

    added: list[dict[str, str]] = []
    skipped: list[tuple[str, str]] = []
    for row in read_candidates(candidates_path):
        if row.get("review_status", "") not in allowed:
            skipped.append((row.get("record_id", ""), "review_status_not_allowed"))
            continue
        mapped = candidate_to_base(row)
        key = (mapped["新闻标题"].strip(), mapped["原文链接"].strip())
        if key in existing_keys or mapped["新闻标题"].strip() in existing_titles:
            skipped.append((row.get("record_id", ""), "duplicate_existing"))
            continue
        added.append(mapped)
        existing_keys.add(key)
        existing_titles.add(mapped["新闻标题"].strip())

    merged = existing + added
    merged.sort(key=lambda row: row.get("时间", ""))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASE_FIELDS)
        writer.writeheader()
        writer.writerows(merged)

    print(f"base_rows={len(existing)}")
    print(f"candidate_rows={len(read_candidates(candidates_path))}")
    print(f"added_rows={len(added)}")
    print(f"skipped_rows={len(skipped)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
