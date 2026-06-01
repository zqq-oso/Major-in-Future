from __future__ import annotations

import argparse
import csv
import html
import re
from collections import defaultdict
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


def slugify(text: str) -> str:
    text = re.sub(r"[\\/:*?\"<>|\s]+", "_", text.strip())
    return text.strip("_") or "major"


def fmt_score(value: float) -> str:
    return f"{value:.2f}"


def make_svg(
    title: str,
    subtitle: str,
    periods: list[str],
    values: list[float],
    counts: list[int],
    color: str,
    y_label: str,
) -> str:
    width = 980
    height = 560
    left = 84
    right = 44
    top = 92
    bottom = 92
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_pos(index: int) -> float:
        if len(periods) == 1:
            return left + plot_w / 2
        return left + plot_w * index / (len(periods) - 1)

    def y_pos(value: float) -> float:
        value = max(1.0, min(10.0, value))
        return top + plot_h * (10.0 - value) / 9.0

    points = [(x_pos(i), y_pos(v)) for i, v in enumerate(values)]
    path_d = " ".join(("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}" for i, (x, y) in enumerate(points))

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:'Noto Sans CJK SC','Microsoft YaHei','PingFang SC','WenQuanYi Zen Hei',Arial,sans-serif;fill:#172033}",
        ".title{font-size:26px;font-weight:700}",
        ".subtitle{font-size:14px;fill:#5f6b7a}",
        ".axis{stroke:#253247;stroke-width:1.2}",
        ".grid{stroke:#d9e0ea;stroke-width:1}",
        ".tick{font-size:13px;fill:#5f6b7a}",
        ".xlabel{font-size:13px;fill:#344054}",
        ".ylabel{font-size:14px;fill:#344054;font-weight:600}",
        ".value{font-size:12px;fill:#172033;font-weight:600}",
        ".count{font-size:11px;fill:#6b7785}",
        "</style>",
        '<rect x="0" y="0" width="980" height="560" fill="#ffffff"/>',
        f'<text x="{left}" y="42" class="title">{html.escape(title)}</text>',
        f'<text x="{left}" y="66" class="subtitle">{html.escape(subtitle)}</text>',
    ]

    for tick in range(1, 11):
        y = y_pos(float(tick))
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 4:.2f}" text-anchor="end" class="tick">{tick}</text>')

    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" class="axis"/>',
            f'<text x="{left - 58}" y="{top + plot_h / 2}" transform="rotate(-90 {left - 58} {top + plot_h / 2})" text-anchor="middle" class="ylabel">{html.escape(y_label)}</text>',
            f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>',
        ]
    )

    for i, ((x, y), period, value, count) in enumerate(zip(points, periods, values, counts)):
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" class="axis"/>')
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" class="xlabel">{html.escape(period)}</text>'
        )
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6.5" fill="#ffffff" stroke="{color}" stroke-width="3"/>')
        label_y = y - 12 if i % 2 == 0 else y + 24
        parts.append(f'<text x="{x:.2f}" y="{label_y:.2f}" text-anchor="middle" class="value">{fmt_score(value)}</text>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 48}" text-anchor="middle" class="count">n={count}</text>')

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render per-major employment and salary trend SVG charts.")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("output/major_outlook_survey/full_zgc_20260521_2x100/analysis"),
    )
    parser.add_argument("--output-subdir", default="plots/by_major")
    args = parser.parse_args()

    analysis_dir = args.analysis_dir
    input_path = analysis_dir / "major_period_mean_scores.csv"
    if not input_path.exists():
        raise SystemExit(f"missing aggregated score table: {input_path}")

    rows_by_major: dict[str, list[dict[str, str]]] = defaultdict(list)
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_major[row["major_category"]].append(row)

    output_dir = analysis_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    index_items: list[dict[str, str]] = []
    for major, rows in sorted(rows_by_major.items(), key=lambda item: item[1][0]["selection_id"]):
        by_period = {row["period"]: row for row in rows}
        periods = [period for period in PERIODS if period in by_period]
        selection_id = rows[0]["selection_id"]
        door = rows[0]["door"]
        counts = [int(by_period[period]["n_profiles"]) for period in periods]
        emp_values = [float(by_period[period]["employment_mean"]) for period in periods]
        sal_values = [float(by_period[period]["salary_mean"]) for period in periods]
        min_n = min(counts)
        max_n = max(counts)
        subtitle = f"门类：{door} | 样本数范围：{min_n}-{max_n} | 分数范围固定为 1-10"

        base_name = f"{selection_id}_{slugify(major)}"
        employment_file = f"{base_name}_employment.svg"
        salary_file = f"{base_name}_salary.svg"

        (output_dir / employment_file).write_text(
            make_svg(
                title=f"{major}：就业前景均分趋势",
                subtitle=subtitle,
                periods=periods,
                values=emp_values,
                counts=counts,
                color="#2563eb",
                y_label="就业前景均分",
            ),
            encoding="utf-8",
        )
        (output_dir / salary_file).write_text(
            make_svg(
                title=f"{major}：薪资前景均分趋势",
                subtitle=subtitle,
                periods=periods,
                values=sal_values,
                counts=counts,
                color="#dc2626",
                y_label="薪资前景均分",
            ),
            encoding="utf-8",
        )
        index_items.append(
            {
                "selection_id": selection_id,
                "major": major,
                "door": door,
                "employment_file": employment_file,
                "salary_file": salary_file,
            }
        )

    index_path = output_dir.parent / "index.html"
    html_parts = [
        "<!doctype html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8"/>',
        "<title>Major Outlook Trend Charts</title>",
        "<style>",
        "body{font-family:'Noto Sans CJK SC','Microsoft YaHei','PingFang SC',Arial,sans-serif;margin:28px;background:#f5f7fb;color:#172033}",
        "h1{font-size:28px;margin:0 0 8px}",
        "p{color:#5f6b7a;margin:0 0 22px}",
        ".major{background:#fff;border:1px solid #d9e0ea;border-radius:8px;margin:0 0 22px;padding:18px}",
        ".major h2{font-size:20px;margin:0 0 14px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px}",
        "img{width:100%;height:auto;border:1px solid #e5eaf2;border-radius:6px;background:#fff}",
        "a{color:#2563eb;text-decoration:none}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>未来专业前景趋势图</h1>",
        "<p>每个专业两张图：就业前景均分、薪资前景均分。横轴为半年期，纵轴固定为 1-10 分。</p>",
    ]
    for item in index_items:
        html_parts.extend(
            [
                '<section class="major">',
                f"<h2>{html.escape(item['selection_id'])} | {html.escape(item['major'])} | {html.escape(item['door'])}</h2>",
                '<div class="grid">',
                f'<a href="by_major/{html.escape(item["employment_file"])}"><img src="by_major/{html.escape(item["employment_file"])}" alt="{html.escape(item["major"])} 就业前景"/></a>',
                f'<a href="by_major/{html.escape(item["salary_file"])}"><img src="by_major/{html.escape(item["salary_file"])}" alt="{html.escape(item["major"])} 薪资前景"/></a>',
                "</div>",
                "</section>",
            ]
        )
    html_parts.extend(["</body>", "</html>"])
    index_path.write_text("\n".join(html_parts) + "\n", encoding="utf-8")

    print(
        {
            "output_dir": str(output_dir),
            "index": str(index_path),
            "majors": len(index_items),
            "charts": len(index_items) * 2,
        }
    )


if __name__ == "__main__":
    main()
