from __future__ import annotations

import argparse
import colorsys
import csv
import html
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_PERIODS = [
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
class FineGroupInfo:
    fine_group_id: str
    fine_group_name: str
    door: str
    major_categories: set[str] = field(default_factory=set)


@dataclass
class FineGroupPeriodStats:
    fine_group_id: str
    fine_group_name: str
    door: str
    period: str
    major_categories: set[str] = field(default_factory=set)
    n_ratings: int = 0
    employment_sum: float = 0.0
    salary_sum: float = 0.0

    def add(self, major_category: str, employment_score: float, salary_score: float) -> None:
        self.major_categories.add(major_category)
        self.n_ratings += 1
        self.employment_sum += employment_score
        self.salary_sum += salary_score

    @property
    def employment_mean(self) -> float:
        return self.employment_sum / self.n_ratings if self.n_ratings else 0.0

    @property
    def salary_mean(self) -> float:
        return self.salary_sum / self.n_ratings if self.n_ratings else 0.0

    @property
    def combined_mean(self) -> float:
        return (self.employment_mean + self.salary_mean) / 2


def fmt(value: float) -> str:
    return f"{value:.4f}"


def fmt_short(value: float) -> str:
    return f"{value:.2f}"


def fmt_signed(value: float) -> str:
    return f"{value:+.2f}"


def period_sort_key(period: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{4})-?S([12])", period)
    if not match:
        return (9999, 9)
    return (int(match.group(1)), int(match.group(2)))


def color_for_index(index: int, total: int) -> str:
    hue = 0.58 if total <= 1 else (index * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.62, 0.70)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


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


def read_periods(run_dir: Path) -> list[str]:
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        periods = metadata.get("survey_periods")
        if isinstance(periods, list) and periods:
            return [str(period) for period in periods]
    return DEFAULT_PERIODS


def read_fine_groups(path: Path) -> tuple[dict[str, FineGroupInfo], dict[str, FineGroupInfo]]:
    by_major: dict[str, FineGroupInfo] = {}
    by_group: dict[str, FineGroupInfo] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            group_id = row["fine_group_id"]
            group = by_group.setdefault(
                group_id,
                FineGroupInfo(
                    fine_group_id=group_id,
                    fine_group_name=row["fine_group_name"],
                    door=row["door"],
                ),
            )
            if group.fine_group_name != row["fine_group_name"] or group.door != row["door"]:
                raise ValueError(f"inconsistent fine group metadata for {group_id}")
            group.major_categories.add(row["major_category"])
            by_major[row["major_category"]] = group
    return by_major, by_group


def aggregate(
    parsed_path: Path,
    fine_group_map: dict[str, FineGroupInfo],
    periods: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    period_order = {period: index for index, period in enumerate(periods)}
    groups: dict[tuple[str, str], FineGroupPeriodStats] = {}
    missing_majors: set[str] = set()

    with parsed_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            major_category = row["major_category"]
            group_info = fine_group_map.get(major_category)
            if group_info is None:
                missing_majors.add(major_category)
                continue
            period = row["period"]
            key = (group_info.fine_group_id, period)
            if key not in groups:
                groups[key] = FineGroupPeriodStats(
                    fine_group_id=group_info.fine_group_id,
                    fine_group_name=group_info.fine_group_name,
                    door=group_info.door,
                    period=period,
                )
            try:
                employment_score = float(row["employment_score"])
                salary_score = float(row["salary_score"])
            except ValueError:
                continue
            if 1.0 <= employment_score <= 10.0 and 1.0 <= salary_score <= 10.0:
                groups[key].add(major_category, employment_score, salary_score)

    if missing_majors:
        raise ValueError(f"fine group mapping missing majors: {', '.join(sorted(missing_majors))}")

    long_rows: list[dict[str, object]] = []
    for (_group_id, period), stats in sorted(
        groups.items(),
        key=lambda item: (period_order.get(item[0][1], 999), item[1].fine_group_id),
    ):
        major_count = len(stats.major_categories)
        profiles_per_major = stats.n_ratings / major_count if major_count else 0.0
        long_rows.append(
            {
                "period": period,
                "period_order": period_order.get(period, 999),
                "fine_group_id": stats.fine_group_id,
                "door": stats.door,
                "fine_group_name": stats.fine_group_name,
                "major_count": major_count,
                "n_ratings": stats.n_ratings,
                "profiles_per_major": fmt(profiles_per_major),
                "employment_mean": fmt(stats.employment_mean),
                "salary_mean": fmt(stats.salary_mean),
                "combined_mean": fmt(stats.combined_mean),
            }
        )

    rows_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in long_rows:
        rows_by_group[str(row["fine_group_id"])].append(row)

    trend_rows: list[dict[str, object]] = []
    for group_id, rows in rows_by_group.items():
        rows.sort(key=lambda row: int(row["period_order"]))
        employment_values = [float(row["employment_mean"]) for row in rows]
        salary_values = [float(row["salary_mean"]) for row in rows]
        combined_values = [float(row["combined_mean"]) for row in rows]
        trend_rows.append(
            {
                "fine_group_id": group_id,
                "door": rows[0]["door"],
                "fine_group_name": rows[0]["fine_group_name"],
                "major_count": rows[0]["major_count"],
                "min_n_ratings": min(int(row["n_ratings"]) for row in rows),
                "first_period": rows[0]["period"],
                "last_period": rows[-1]["period"],
                "employment_first": fmt(employment_values[0]),
                "employment_last": fmt(employment_values[-1]),
                "employment_delta": fmt(employment_values[-1] - employment_values[0]),
                "employment_slope_per_half_year": fmt(slope(employment_values)),
                "salary_first": fmt(salary_values[0]),
                "salary_last": fmt(salary_values[-1]),
                "salary_delta": fmt(salary_values[-1] - salary_values[0]),
                "salary_slope_per_half_year": fmt(slope(salary_values)),
                "combined_first": fmt(combined_values[0]),
                "combined_last": fmt(combined_values[-1]),
                "combined_delta": fmt(combined_values[-1] - combined_values[0]),
                "combined_slope_per_half_year": fmt(slope(combined_values)),
            }
        )

    trend_rows.sort(key=lambda row: (-float(row["combined_last"]), str(row["fine_group_id"])))
    for rank, row in enumerate(trend_rows, start=1):
        row["combined_last_rank"] = rank

    return long_rows, trend_rows


def rows_to_series(rows: list[dict[str, object]], periods: list[str], score_field: str) -> dict[str, dict[str, object]]:
    by_group: dict[str, dict[str, object]] = {}
    for row in rows:
        group_id = str(row["fine_group_id"])
        by_group.setdefault(
            group_id,
            {
                "fine_group_id": group_id,
                "door": row["door"],
                "fine_group_name": row["fine_group_name"],
                "major_count": int(row["major_count"]),
                "n_ratings": [],
                "rows": {},
            },
        )
        by_group[group_id]["rows"][str(row["period"])] = row

    series: dict[str, dict[str, object]] = {}
    for group_id, item in sorted(by_group.items(), key=lambda pair: pair[0]):
        values = []
        n_ratings = []
        row_map: dict[str, dict[str, object]] = item["rows"]  # type: ignore[assignment]
        for period in periods:
            row = row_map[period]
            values.append(float(row[score_field]))
            n_ratings.append(int(row["n_ratings"]))
        series[group_id] = {
            "fine_group_id": group_id,
            "door": item["door"],
            "fine_group_name": item["fine_group_name"],
            "major_count": item["major_count"],
            "values": values,
            "n_ratings": n_ratings,
        }
    return series


def make_svg(title: str, subtitle: str, periods: list[str], series: dict[str, dict[str, object]], y_label: str) -> str:
    group_count = len(series)
    width = 1680
    height = max(920, 124 + group_count * 36)
    left = 82
    right = 620
    top = 92
    bottom = 92
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_pos(index: int) -> float:
        return left + plot_w * index / (len(periods) - 1)

    def y_pos(value: float) -> float:
        value = max(1.0, min(10.0, value))
        return top + plot_h * (10.0 - value) / 9.0

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:'Noto Sans CJK SC','Microsoft YaHei','PingFang SC','WenQuanYi Zen Hei',Arial,sans-serif;fill:#172033}",
        ".title{font-size:28px;font-weight:700}",
        ".subtitle{font-size:14px;fill:#5f6b7a}",
        ".axis{stroke:#253247;stroke-width:1.2}",
        ".grid{stroke:#d9e0ea;stroke-width:1}",
        ".tick{font-size:13px;fill:#5f6b7a}",
        ".xlabel{font-size:13px;fill:#344054}",
        ".ylabel{font-size:14px;fill:#344054;font-weight:600}",
        ".legend{font-size:13px;fill:#172033}",
        ".legend-small{font-size:11px;fill:#6b7785}",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="42" class="title">{html.escape(title)}</text>',
        f'<text x="{left}" y="66" class="subtitle">{html.escape(subtitle)}</text>',
    ]

    for tick in range(1, 11):
        y = y_pos(float(tick))
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 4:.2f}" text-anchor="end" class="tick">{tick}</text>')

    for index, period in enumerate(periods):
        x = x_pos(index)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" class="grid"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" class="xlabel">{html.escape(period)}</text>')

    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" class="axis"/>',
            f'<text x="{left - 58}" y="{top + plot_h / 2}" transform="rotate(-90 {left - 58} {top + plot_h / 2})" text-anchor="middle" class="ylabel">{html.escape(y_label)}</text>',
        ]
    )

    plot_items: list[dict[str, object]] = []
    for index, (group_id, item) in enumerate(series.items()):
        values: list[float] = item["values"]  # type: ignore[assignment]
        points = [(x_pos(i), y_pos(value)) for i, value in enumerate(values)]
        path = " ".join(("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}" for i, (x, y) in enumerate(points))
        plot_items.append(
            {
                "fine_group_id": group_id,
                "door": item["door"],
                "fine_group_name": item["fine_group_name"],
                "major_count": item["major_count"],
                "n_ratings": item["n_ratings"],
                "values": values,
                "path": path,
                "color": color_for_index(index, group_count),
                "end_x": points[-1][0],
                "end_y": points[-1][1],
                "end_value": values[-1],
                "delta": values[-1] - values[0],
            }
        )

    for item in sorted(plot_items, key=lambda row: float(row["end_value"])):
        parts.append(
            f'<path d="{item["path"]}" fill="none" stroke="{item["color"]}" stroke-width="2.7" '
            'stroke-linecap="round" stroke-linejoin="round" opacity="0.78"/>'
        )
    for item in plot_items:
        parts.append(
            f'<circle cx="{float(item["end_x"]):.2f}" cy="{float(item["end_y"]):.2f}" r="4.0" '
            f'fill="#ffffff" stroke="{item["color"]}" stroke-width="1.7"/>'
        )

    legend_x = width - right + 34
    legend_y = top
    parts.append(f'<text x="{legend_x}" y="{legend_y - 24}" class="ylabel">图例（按{html.escape(periods[-1])}排序）</text>')
    for rank, item in enumerate(sorted(plot_items, key=lambda row: (-float(row["end_value"]), str(row["fine_group_id"]))), start=1):
        y = legend_y + (rank - 1) * 36
        value = fmt_short(float(item["end_value"]))
        delta = fmt_signed(float(item["delta"]))
        label = f'{rank}. {item["fine_group_name"]}：{value} | 变化 {delta}'
        major_count = int(item["major_count"])
        min_n = min(item["n_ratings"])  # type: ignore[arg-type]
        parts.append(f'<line x1="{legend_x}" y1="{y:.2f}" x2="{legend_x + 26}" y2="{y:.2f}" stroke="{item["color"]}" stroke-width="3.5" stroke-linecap="round"/>')
        parts.append(f'<text x="{legend_x + 36}" y="{y + 4:.2f}" class="legend">{html.escape(label)}</text>')
        parts.append(f'<text x="{legend_x + 36}" y="{y + 19:.2f}" class="legend-small">{html.escape(str(item["fine_group_id"]))} | 门类 {html.escape(str(item["door"]))} | 专业数 {major_count} | 每期评分数 >= {min_n}</text>')

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def table_html(rows: list[dict[str, object]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['combined_last_rank']))}</td>"
            f"<td>{html.escape(str(row['fine_group_id']))}</td>"
            f"<td>{html.escape(str(row['door']))}</td>"
            f"<td>{html.escape(str(row['fine_group_name']))}</td>"
            f"<td>{html.escape(str(row['major_count']))}</td>"
            f"<td>{fmt_short(float(row['employment_last']))}</td>"
            f"<td>{fmt_short(float(row['employment_delta']))}</td>"
            f"<td>{fmt_short(float(row['salary_last']))}</td>"
            f"<td>{fmt_short(float(row['salary_delta']))}</td>"
            f"<td>{html.escape(str(row['min_n_ratings']))}</td>"
            "</tr>"
        )
    return "\n".join(body)


def write_index(output_dir: Path, trend_rows: list[dict[str, object]]) -> None:
    html_parts = [
        "<!doctype html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8"/>',
        "<title>Fine-Group Outlook Trend Charts</title>",
        "<style>",
        "body{font-family:'Noto Sans CJK SC','Microsoft YaHei','PingFang SC',Arial,sans-serif;margin:28px;background:#f5f7fb;color:#172033}",
        "h1{font-size:28px;margin:0 0 8px}",
        "p{color:#5f6b7a;margin:0 0 20px}",
        "section{background:#fff;border:1px solid #d9e0ea;border-radius:8px;margin:0 0 22px;padding:18px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(640px,1fr));gap:16px}",
        "img{width:100%;height:auto;border:1px solid #e5eaf2;border-radius:6px;background:#fff}",
        "table{border-collapse:collapse;width:100%;font-size:13px;margin-top:12px}",
        "th,td{border-bottom:1px solid #e5eaf2;padding:7px 8px;text-align:left;white-space:nowrap}",
        "th{background:#f8fafc;color:#344054}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>专业细分群组前景趋势图</h1>",
        "<p>每条折线代表一个新划分的细分群组；细分群组均分按该群组在对应时期的全部个体-专业评分记录聚合。</p>",
        "<section>",
        '<div class="grid">',
        '<a href="fine_group_employment_mean_trends.svg"><img src="fine_group_employment_mean_trends.svg" alt="细分群组就业前景均分趋势"/></a>',
        '<a href="fine_group_salary_mean_trends.svg"><img src="fine_group_salary_mean_trends.svg" alt="细分群组薪酬前景均分趋势"/></a>',
        "</div>",
        "</section>",
        "<section>",
        "<h2>2030-S2 排序摘要</h2>",
        "<table><thead><tr><th>Rank</th><th>ID</th><th>门类</th><th>细分群组</th><th>专业数</th><th>就业2030-S2</th><th>就业变化</th><th>薪酬2030-S2</th><th>薪酬变化</th><th>最小每期评分数</th></tr></thead><tbody>",
        table_html(trend_rows),
        "</tbody></table>",
        "</section>",
        "</body>",
        "</html>",
    ]
    (output_dir / "index.html").write_text("\n".join(html_parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate and plot fine-group outlook trends.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("output/your_run"),
        help="Survey run directory containing parsed_ratings.csv and optional metadata.json.",
    )
    parser.add_argument(
        "--fine-groups",
        type=Path,
        default=Path("major/major_category_fine_groups.csv"),
        help="CSV mapping major categories to fine groups.",
    )
    parser.add_argument("--analysis-subdir", default="analysis")
    parser.add_argument("--output-subdir", default="plots/fine_groups")
    args = parser.parse_args()

    run_dir = args.run_dir
    parsed_path = run_dir / "parsed_ratings.csv"
    if not parsed_path.exists():
        raise SystemExit(f"missing parsed ratings: {parsed_path}")
    if not args.fine_groups.exists():
        raise SystemExit(f"missing fine group mapping: {args.fine_groups}")

    periods = sorted(read_periods(run_dir), key=period_sort_key)
    fine_group_map, by_group = read_fine_groups(args.fine_groups)
    long_rows, trend_rows = aggregate(parsed_path, fine_group_map, periods)

    analysis_dir = run_dir / args.analysis_subdir
    output_dir = analysis_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        analysis_dir / "fine_group_period_mean_scores.csv",
        long_rows,
        [
            "period",
            "period_order",
            "fine_group_id",
            "door",
            "fine_group_name",
            "major_count",
            "n_ratings",
            "profiles_per_major",
            "employment_mean",
            "salary_mean",
            "combined_mean",
        ],
    )
    write_csv(
        analysis_dir / "fine_group_trend_summary.csv",
        trend_rows,
        [
            "combined_last_rank",
            "fine_group_id",
            "door",
            "fine_group_name",
            "major_count",
            "min_n_ratings",
            "first_period",
            "last_period",
            "employment_first",
            "employment_last",
            "employment_delta",
            "employment_slope_per_half_year",
            "salary_first",
            "salary_last",
            "salary_delta",
            "salary_slope_per_half_year",
            "combined_first",
            "combined_last",
            "combined_delta",
            "combined_slope_per_half_year",
        ],
    )

    employment_series = rows_to_series(long_rows, periods, "employment_mean")
    salary_series = rows_to_series(long_rows, periods, "salary_mean")
    subtitle = (
        f"映射：major_category_fine_groups.csv | 细分群组数：{len(by_group)} | "
        "横轴为半年期，纵轴固定为 1-10 分"
    )
    (output_dir / "fine_group_employment_mean_trends.svg").write_text(
        make_svg(
            title="各专业细分群组就业前景均分趋势",
            subtitle=subtitle,
            periods=periods,
            series=employment_series,
            y_label="就业前景均分",
        ),
        encoding="utf-8",
    )
    (output_dir / "fine_group_salary_mean_trends.svg").write_text(
        make_svg(
            title="各专业细分群组薪酬前景均分趋势",
            subtitle=subtitle,
            periods=periods,
            series=salary_series,
            y_label="薪酬前景均分",
        ),
        encoding="utf-8",
    )
    write_index(output_dir, trend_rows)

    print(
        {
            "run_dir": str(run_dir),
            "fine_group_period_rows": len(long_rows),
            "fine_groups": len(by_group),
            "periods": len(periods),
            "csv": str(analysis_dir / "fine_group_period_mean_scores.csv"),
            "summary_csv": str(analysis_dir / "fine_group_trend_summary.csv"),
            "plot_dir": str(output_dir),
            "charts": 2,
        }
    )


if __name__ == "__main__":
    main()
