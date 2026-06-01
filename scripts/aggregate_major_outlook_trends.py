from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PERIODS = [
    "2026-S1",
    "2026-S2",
    "2027-S1",
    "2027-S2",
    "2028-S1",
    "2028-S2",
    "2029-S1",
    "2029-S2",
    "2030-S1",
    "2030-S2",
]


@dataclass
class GroupStats:
    selection_id: str
    door: str
    major_category: str
    n: int = 0
    employment_sum: float = 0.0
    salary_sum: float = 0.0

    def add(self, employment: float, salary: float) -> None:
        self.n += 1
        self.employment_sum += employment
        self.salary_sum += salary

    @property
    def employment_mean(self) -> float:
        return self.employment_sum / self.n if self.n else 0.0

    @property
    def salary_mean(self) -> float:
        return self.salary_sum / self.n if self.n else 0.0

    @property
    def combined_mean(self) -> float:
        return (self.employment_mean + self.salary_mean) / 2


def round_score(value: float) -> str:
    return f"{value:.4f}"


def rank_desc(rows: list[dict[str, object]], key: str, out_key: str) -> None:
    ordered = sorted(rows, key=lambda row: (-float(row[key]), str(row["major_category"])))
    last_value: float | None = None
    last_rank = 0
    for index, row in enumerate(ordered, start=1):
        value = float(row[key])
        if last_value is None or value != last_value:
            last_rank = index
            last_value = value
        row[out_key] = last_rank


def slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    xs = list(range(len(values)))
    x_mean = sum(xs) / len(xs)
    y_mean = sum(values) / len(values)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denom


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate major outlook scores by major and half-year period.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("output/major_outlook_survey/full_zgc_20260521_2x100"),
        help="Formal survey run directory containing parsed_ratings.csv and metadata.json.",
    )
    parser.add_argument("--output-subdir", default="analysis")
    args = parser.parse_args()

    run_dir = args.run_dir
    parsed_path = run_dir / "parsed_ratings.csv"
    metadata_path = run_dir / "metadata.json"
    if not parsed_path.exists():
        raise SystemExit(f"missing parsed ratings: {parsed_path}")

    expected_profiles = 1000
    periods = PERIODS
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_profiles = int(metadata.get("profile_count", expected_profiles))
        periods = list(metadata.get("survey_periods", periods))

    period_order = {period: index for index, period in enumerate(periods)}
    groups: dict[tuple[str, str], GroupStats] = {}

    with parsed_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            period = row["period"]
            major_category = row["major_category"]
            key = (period, major_category)
            if key not in groups:
                groups[key] = GroupStats(
                    selection_id=row["selection_id"],
                    door=row["door"],
                    major_category=major_category,
                )
            try:
                employment = float(row["employment_score"])
                salary = float(row["salary_score"])
            except ValueError:
                continue
            if 1 <= employment <= 10 and 1 <= salary <= 10:
                groups[key].add(employment, salary)

    long_rows: list[dict[str, object]] = []
    for (period, _major), stats in sorted(groups.items(), key=lambda item: (period_order.get(item[0][0], 999), item[1].selection_id)):
        long_rows.append(
            {
                "period": period,
                "period_order": period_order.get(period, 999),
                "selection_id": stats.selection_id,
                "door": stats.door,
                "major_category": stats.major_category,
                "n_profiles": stats.n,
                "expected_profiles": expected_profiles,
                "coverage_pct": round_score(stats.n / expected_profiles * 100),
                "employment_mean": round_score(stats.employment_mean),
                "salary_mean": round_score(stats.salary_mean),
                "combined_mean": round_score(stats.combined_mean),
            }
        )

    by_period: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in long_rows:
        by_period[str(row["period"])].append(dict(row))

    ranking_rows: list[dict[str, object]] = []
    for period in periods:
        rows = by_period.get(period, [])
        rank_desc(rows, "employment_mean", "employment_rank")
        rank_desc(rows, "salary_mean", "salary_rank")
        rank_desc(rows, "combined_mean", "combined_rank")
        ranking_rows.extend(sorted(rows, key=lambda row: (int(row["combined_rank"]), str(row["selection_id"]))))

    majors: dict[str, dict[str, object]] = {}
    for row in long_rows:
        major = str(row["major_category"])
        majors.setdefault(
            major,
            {
                "selection_id": row["selection_id"],
                "door": row["door"],
                "major_category": major,
                "period_rows": {},
            },
        )
        majors[major]["period_rows"][str(row["period"])] = row

    wide_rows: list[dict[str, object]] = []
    trend_rows: list[dict[str, object]] = []
    for major_info in sorted(majors.values(), key=lambda item: str(item["selection_id"])):
        period_rows: dict[str, dict[str, object]] = major_info["period_rows"]  # type: ignore[assignment]
        wide: dict[str, object] = {
            "selection_id": major_info["selection_id"],
            "door": major_info["door"],
            "major_category": major_info["major_category"],
        }
        emp_values: list[float] = []
        sal_values: list[float] = []
        combined_values: list[float] = []
        counts: list[int] = []
        for period in periods:
            row = period_rows.get(period)
            if row:
                emp = float(row["employment_mean"])
                sal = float(row["salary_mean"])
                combined = float(row["combined_mean"])
                count = int(row["n_profiles"])
            else:
                emp = sal = combined = 0.0
                count = 0
            emp_values.append(emp)
            sal_values.append(sal)
            combined_values.append(combined)
            counts.append(count)
            wide[f"{period}_employment_mean"] = round_score(emp)
            wide[f"{period}_salary_mean"] = round_score(sal)
            wide[f"{period}_combined_mean"] = round_score(combined)
            wide[f"{period}_n_profiles"] = count
        wide_rows.append(wide)

        trend_rows.append(
            {
                "selection_id": major_info["selection_id"],
                "door": major_info["door"],
                "major_category": major_info["major_category"],
                "min_n_profiles": min(counts) if counts else 0,
                "employment_mean_all_periods": round_score(sum(emp_values) / len(emp_values)),
                "salary_mean_all_periods": round_score(sum(sal_values) / len(sal_values)),
                "combined_mean_all_periods": round_score(sum(combined_values) / len(combined_values)),
                "employment_2026S1": round_score(emp_values[0]),
                "employment_2030S2": round_score(emp_values[-1]),
                "employment_delta_2030S2_minus_2026S1": round_score(emp_values[-1] - emp_values[0]),
                "employment_slope_per_half_year": round_score(slope(emp_values)),
                "salary_2026S1": round_score(sal_values[0]),
                "salary_2030S2": round_score(sal_values[-1]),
                "salary_delta_2030S2_minus_2026S1": round_score(sal_values[-1] - sal_values[0]),
                "salary_slope_per_half_year": round_score(slope(sal_values)),
                "combined_2026S1": round_score(combined_values[0]),
                "combined_2030S2": round_score(combined_values[-1]),
                "combined_delta_2030S2_minus_2026S1": round_score(combined_values[-1] - combined_values[0]),
                "combined_slope_per_half_year": round_score(slope(combined_values)),
            }
        )

    for key, rank_key in [
        ("combined_mean_all_periods", "combined_all_periods_rank"),
        ("combined_2030S2", "combined_2030S2_rank"),
        ("combined_delta_2030S2_minus_2026S1", "combined_delta_rank"),
    ]:
        rank_desc(trend_rows, key, rank_key)
    trend_rows = sorted(trend_rows, key=lambda row: (int(row["combined_2030S2_rank"]), str(row["selection_id"])))

    out_dir = run_dir / args.output_subdir
    write_csv(
        out_dir / "major_period_mean_scores.csv",
        long_rows,
        [
            "period",
            "period_order",
            "selection_id",
            "door",
            "major_category",
            "n_profiles",
            "expected_profiles",
            "coverage_pct",
            "employment_mean",
            "salary_mean",
            "combined_mean",
        ],
    )

    wide_fields = ["selection_id", "door", "major_category"]
    for period in periods:
        wide_fields.extend(
            [
                f"{period}_employment_mean",
                f"{period}_salary_mean",
                f"{period}_combined_mean",
                f"{period}_n_profiles",
            ]
        )
    write_csv(out_dir / "major_period_mean_scores_wide.csv", wide_rows, wide_fields)

    ranking_fields = [
        "period",
        "period_order",
        "selection_id",
        "door",
        "major_category",
        "n_profiles",
        "expected_profiles",
        "coverage_pct",
        "employment_mean",
        "salary_mean",
        "combined_mean",
        "employment_rank",
        "salary_rank",
        "combined_rank",
    ]
    write_csv(out_dir / "period_major_rankings.csv", ranking_rows, ranking_fields)

    trend_fields = [
        "selection_id",
        "door",
        "major_category",
        "min_n_profiles",
        "employment_mean_all_periods",
        "salary_mean_all_periods",
        "combined_mean_all_periods",
        "employment_2026S1",
        "employment_2030S2",
        "employment_delta_2030S2_minus_2026S1",
        "employment_slope_per_half_year",
        "salary_2026S1",
        "salary_2030S2",
        "salary_delta_2030S2_minus_2026S1",
        "salary_slope_per_half_year",
        "combined_2026S1",
        "combined_2030S2",
        "combined_delta_2030S2_minus_2026S1",
        "combined_slope_per_half_year",
        "combined_all_periods_rank",
        "combined_2030S2_rank",
        "combined_delta_rank",
    ]
    write_csv(out_dir / "major_trend_summary.csv", trend_rows, trend_fields)

    top_2030 = sorted(trend_rows, key=lambda row: int(row["combined_2030S2_rank"]))[:8]
    top_risers = sorted(trend_rows, key=lambda row: int(row["combined_delta_rank"]))[:8]
    lines = [
        "# Major Outlook Trend Summary",
        "",
        f"- Source: `{parsed_path}`",
        f"- Periods: {', '.join(periods)}",
        f"- Major-period rows: {len(long_rows)}",
        f"- Expected profiles per major-period: {expected_profiles}",
        "",
        "## Top Combined Score In 2030-S2",
        "",
        "| Rank | Major | Door | Employment | Salary | Combined | n |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in top_2030:
        period_row = majors[str(row["major_category"])]["period_rows"]["2030-S2"]  # type: ignore[index]
        lines.append(
            f"| {row['combined_2030S2_rank']} | {row['major_category']} | {row['door']} | "
            f"{period_row['employment_mean']} | {period_row['salary_mean']} | {period_row['combined_mean']} | {period_row['n_profiles']} |"
        )
    lines.extend(
        [
            "",
            "## Largest Combined Score Increase",
            "",
            "| Rank | Major | Door | 2026-S1 | 2030-S2 | Delta | Slope/Half-Year |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in top_risers:
        lines.append(
            f"| {row['combined_delta_rank']} | {row['major_category']} | {row['door']} | "
            f"{row['combined_2026S1']} | {row['combined_2030S2']} | "
            f"{row['combined_delta_2030S2_minus_2026S1']} | {row['combined_slope_per_half_year']} |"
        )
    (out_dir / "trend_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "major_period_rows": len(long_rows),
                "wide_rows": len(wide_rows),
                "trend_rows": len(trend_rows),
                "ranking_rows": len(ranking_rows),
                "min_n_profiles": min(int(row["n_profiles"]) for row in long_rows),
                "max_n_profiles": max(int(row["n_profiles"]) for row in long_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
