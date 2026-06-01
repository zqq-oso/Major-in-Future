from __future__ import annotations

import argparse
import calendar
import csv
import datetime as dt
import json
import re
import shutil
from pathlib import Path


CN_DATE_RE = re.compile(r"(20\d{2})年\s*(\d{1,2})月(?:\s*(\d{1,2})日)?")
ISO_DATE_RE = re.compile(r"\b(20\d{2})([-/])(\d{1,2})(?:\2(\d{1,2}))?\b")
BARE_MONTH_RE = re.compile(r"(?<![\d年-])([1-9]|1[0-2])月(?:\s*(\d{1,2})日)?(?![\d个份])")
CN_QUARTER_RE = re.compile(r"(20\d{2})年\s*第?([一二三四1234])季度")
ISO_QUARTER_RE = re.compile(r"\b(20\d{2})\s*Q([1-4])\b", re.IGNORECASE)
CN_HALF_RE = re.compile(r"(20\d{2})年\s*(上半年|下半年)")
QUARTER_TO_INT = {"一": 1, "二": 2, "三": 3, "四": 4, "1": 1, "2": 2, "3": 3, "4": 4}
INT_TO_QUARTER = {1: "一", 2: "二", 3: "三", 4: "四"}


def half_year_bounds(period: str) -> tuple[int, int, int]:
    year_text, half_text = period.split("-S", 1)
    year = int(year_text)
    half = int(half_text)
    if half == 1:
        return year, 1, 6
    return year, 7, 12


def is_history(period: str, history_end: str) -> bool:
    return period <= history_end


def month_fits(period: str, year: int, month: int) -> bool:
    target_year, start_month, end_month = half_year_bounds(period)
    return year == target_year and start_month <= month <= end_month


def mapped_month(period: str, old_month: int, row_index: int) -> int:
    _, start_month, end_month = half_year_bounds(period)
    if start_month == 7 and 1 <= old_month <= 6:
        return old_month + 6
    if end_month == 6 and 7 <= old_month <= 12:
        return old_month - 6
    months = list(range(start_month, end_month + 1))
    return months[(row_index - 1) % len(months)]


def mapped_quarter(period: str, old_quarter: int) -> int:
    _, start_month, end_month = half_year_bounds(period)
    if end_month == 6:
        return old_quarter if old_quarter in (1, 2) else old_quarter - 2
    return old_quarter if old_quarter in (3, 4) else old_quarter + 2


def target_half_text(period: str) -> str:
    _, _, end_month = half_year_bounds(period)
    return "上半年" if end_month == 6 else "下半年"


def capped_day(year: int, month: int, day_text: str | None) -> int | None:
    if not day_text:
        return None
    day = int(day_text)
    return min(day, calendar.monthrange(year, month)[1])


def fix_text_dates(text: str, period: str, row_index: int) -> tuple[str, int]:
    target_year, _, _ = half_year_bounds(period)
    replacements = 0

    def replace_cn(match: re.Match[str]) -> str:
        nonlocal replacements
        year = int(match.group(1))
        month = int(match.group(2))
        day_text = match.group(3)
        if month_fits(period, year, month):
            return match.group(0)
        new_month = mapped_month(period, month, row_index)
        new_day = capped_day(target_year, new_month, day_text)
        replacements += 1
        if new_day is None:
            return f"{target_year}年{new_month}月"
        return f"{target_year}年{new_month}月{new_day}日"

    def replace_iso(match: re.Match[str]) -> str:
        nonlocal replacements
        year = int(match.group(1))
        sep = match.group(2)
        month = int(match.group(3))
        day_text = match.group(4)
        if month_fits(period, year, month):
            return match.group(0)
        new_month = mapped_month(period, month, row_index)
        new_day = capped_day(target_year, new_month, day_text)
        replacements += 1
        if new_day is None:
            return f"{target_year}{sep}{new_month:02d}"
        return f"{target_year}{sep}{new_month:02d}{sep}{new_day:02d}"

    fixed = CN_DATE_RE.sub(replace_cn, text)
    fixed = ISO_DATE_RE.sub(replace_iso, fixed)

    def replace_bare_month(match: re.Match[str]) -> str:
        nonlocal replacements
        month = int(match.group(1))
        day_text = match.group(2)
        _, start_month, end_month = half_year_bounds(period)
        if start_month <= month <= end_month:
            return match.group(0)
        new_month = mapped_month(period, month, row_index)
        new_day = capped_day(target_year, new_month, day_text)
        replacements += 1
        if new_day is None:
            return f"{new_month}月"
        return f"{new_month}月{new_day}日"

    fixed = BARE_MONTH_RE.sub(replace_bare_month, fixed)

    def replace_cn_quarter(match: re.Match[str]) -> str:
        nonlocal replacements
        year = int(match.group(1))
        quarter = QUARTER_TO_INT[match.group(2)]
        new_quarter = mapped_quarter(period, quarter)
        target_half = 1 if half_year_bounds(period)[2] == 6 else 2
        mentioned_half = 1 if quarter <= 2 else 2
        if year == target_year and mentioned_half == target_half:
            return match.group(0)
        replacements += 1
        return f"{target_year}年第{INT_TO_QUARTER[new_quarter]}季度"

    def replace_iso_quarter(match: re.Match[str]) -> str:
        nonlocal replacements
        year = int(match.group(1))
        quarter = int(match.group(2))
        new_quarter = mapped_quarter(period, quarter)
        target_half = 1 if half_year_bounds(period)[2] == 6 else 2
        mentioned_half = 1 if quarter <= 2 else 2
        if year == target_year and mentioned_half == target_half:
            return match.group(0)
        replacements += 1
        return f"{target_year} Q{new_quarter}"

    def replace_cn_half(match: re.Match[str]) -> str:
        nonlocal replacements
        year = int(match.group(1))
        half_text = match.group(2)
        new_half_text = target_half_text(period)
        if year == target_year and half_text == new_half_text:
            return match.group(0)
        replacements += 1
        return f"{target_year}年{new_half_text}"

    fixed = CN_QUARTER_RE.sub(replace_cn_quarter, fixed)
    fixed = ISO_QUARTER_RE.sub(replace_iso_quarter, fixed)
    fixed = CN_HALF_RE.sub(replace_cn_half, fixed)
    return fixed, replacements


def mismatch_count(text: str, period: str) -> int:
    count = 0
    target_year, _, end_month = half_year_bounds(period)
    target_half = 1 if end_month == 6 else 2
    for match in CN_DATE_RE.finditer(text):
        if not month_fits(period, int(match.group(1)), int(match.group(2))):
            count += 1
    for match in ISO_DATE_RE.finditer(text):
        if not month_fits(period, int(match.group(1)), int(match.group(3))):
            count += 1
    _, start_month, end_month = half_year_bounds(period)
    for match in BARE_MONTH_RE.finditer(text):
        mentioned_month = int(match.group(1))
        if not (start_month <= mentioned_month <= end_month):
            count += 1
    for match in CN_QUARTER_RE.finditer(text):
        mentioned_year = int(match.group(1))
        mentioned_quarter = QUARTER_TO_INT[match.group(2)]
        mentioned_half = 1 if mentioned_quarter <= 2 else 2
        if mentioned_year != target_year or mentioned_half != target_half:
            count += 1
    for match in ISO_QUARTER_RE.finditer(text):
        mentioned_year = int(match.group(1))
        mentioned_quarter = int(match.group(2))
        mentioned_half = 1 if mentioned_quarter <= 2 else 2
        if mentioned_year != target_year or mentioned_half != target_half:
            count += 1
    for match in CN_HALF_RE.finditer(text):
        mentioned_year = int(match.group(1))
        mentioned_half = 1 if match.group(2) == "上半年" else 2
        if mentioned_year != target_year or mentioned_half != target_half:
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def backup_run_news(run_dir: Path) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = run_dir / f"news_month_mismatch_backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(run_dir / "news", backup_dir / "news")
    timeline = run_dir / "news_timeline.json"
    if timeline.exists():
        shutil.copy2(timeline, backup_dir / "news_timeline.json")
    return backup_dir


def rebuild_news_timeline(run_dir: Path) -> None:
    timeline: dict[str, list[dict]] = {}
    for path in sorted((run_dir / "news").glob("*.jsonl")):
        timeline[path.stem] = read_jsonl(path)
    (run_dir / "news_timeline.json").write_text(
        json.dumps(timeline, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit or fix explicit month mismatches in generated future-news files.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--history-end", default="2026-S1")
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--audit-csv", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    news_dir = run_dir / "news"
    if not news_dir.exists():
        raise SystemExit(f"news directory not found: {news_dir}")

    backup_dir = backup_run_news(run_dir) if args.fix else None
    audit_rows: list[dict[str, str]] = []
    summary: dict[str, dict[str, int]] = {}

    for path in sorted(news_dir.glob("*.jsonl")):
        period = path.stem
        if is_history(period, args.history_end):
            continue
        rows = read_jsonl(path)
        period_mismatch_before = 0
        period_mismatch_after = 0
        period_replacements = 0
        fixed_rows: list[dict] = []

        for idx, row in enumerate(rows, start=1):
            fixed_row = dict(row)
            for field in ("title", "summary"):
                before = str(fixed_row.get(field, ""))
                before_mismatch = mismatch_count(before, period)
                period_mismatch_before += before_mismatch
                after, replacements = fix_text_dates(before, period, idx)
                after_mismatch = mismatch_count(after, period)
                period_mismatch_after += after_mismatch
                period_replacements += replacements
                if before != after or before_mismatch:
                    audit_rows.append(
                        {
                            "period": period,
                            "idx": str(idx),
                            "field": field,
                            "mismatch_before": str(before_mismatch),
                            "mismatch_after": str(after_mismatch),
                            "replacements": str(replacements),
                            "before": before,
                            "after": after,
                        }
                    )
                fixed_row[field] = after
            fixed_rows.append(fixed_row)

        summary[period] = {
            "rows": len(rows),
            "mismatch_before": period_mismatch_before,
            "mismatch_after": period_mismatch_after,
            "replacements": period_replacements,
        }
        if args.fix:
            write_jsonl(path, fixed_rows)

    if args.fix:
        rebuild_news_timeline(run_dir)

    audit_csv = Path(args.audit_csv) if args.audit_csv else run_dir / "future_news_month_fix_audit.csv"
    with audit_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "period",
                "idx",
                "field",
                "mismatch_before",
                "mismatch_after",
                "replacements",
                "before",
                "after",
            ],
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    print(json.dumps({
        "run_dir": str(run_dir),
        "mode": "fix" if args.fix else "audit",
        "backup_dir": str(backup_dir) if backup_dir else "",
        "audit_csv": str(audit_csv),
        "summary": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
