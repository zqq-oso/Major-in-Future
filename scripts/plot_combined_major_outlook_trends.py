from __future__ import annotations

import argparse
import csv
import html
import colorsys
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

def fmt(value: float) -> str:
    return f"{value:.2f}"


def color_for_index(index: int, total: int) -> str:
    if total <= 1:
        hue = 0.58
    else:
        hue = (index * 0.61803398875) % 1.0
    saturation = 0.58
    value = 0.72
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def read_scores(path: Path) -> dict[str, dict[str, object]]:
    majors: dict[str, dict[str, object]] = {}
    rows_by_major: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_major[row["major_category"]].append(row)

    for major, rows in rows_by_major.items():
        rows.sort(key=lambda row: int(row["period_order"]))
        majors[major] = {
            "selection_id": rows[0]["selection_id"],
            "door": rows[0]["door"],
            "rows": rows,
        }
    return dict(sorted(majors.items(), key=lambda item: str(item[1]["selection_id"])))


def make_svg(
    title: str,
    subtitle: str,
    majors: dict[str, dict[str, object]],
    score_field: str,
    y_label: str,
    show_inline_legend: bool,
) -> str:
    major_count = len(majors)
    width = 1500 if major_count > 40 else 1280
    height = 860 if major_count > 40 else 820
    left = 82
    right = 80 if major_count > 40 else 360
    top = 92
    bottom = 92
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_pos(index: int) -> float:
        return left + plot_w * index / (len(PERIODS) - 1)

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

    for i, period in enumerate(PERIODS):
        x = x_pos(i)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" class="grid"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" class="xlabel">{period}</text>')

    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" class="axis"/>',
            f'<text x="{left - 58}" y="{top + plot_h / 2}" transform="rotate(-90 {left - 58} {top + plot_h / 2})" text-anchor="middle" class="ylabel">{html.escape(y_label)}</text>',
        ]
    )

    series: list[dict[str, object]] = []
    for index, (major, info) in enumerate(majors.items()):
        rows: list[dict[str, str]] = info["rows"]  # type: ignore[assignment]
        values = [float(row[score_field]) for row in rows]
        points = [(x_pos(i), y_pos(value)) for i, value in enumerate(values)]
        path_d = " ".join(("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}" for i, (x, y) in enumerate(points))
        color = color_for_index(index, major_count)
        series.append(
            {
                "major": major,
                "door": info["door"],
                "selection_id": info["selection_id"],
                "color": color,
                "values": values,
                "path": path_d,
                "end_x": points[-1][0],
                "end_y": points[-1][1],
                "end_value": values[-1],
            }
        )

    # Draw high-ranking final-value lines last so they remain visible.
    draw_order = sorted(series, key=lambda item: float(item["end_value"]))
    for item in draw_order:
        parts.append(
            f'<path d="{item["path"]}" fill="none" stroke="{item["color"]}" '
            f'stroke-width="{1.8 if major_count > 40 else 2.8}" '
            f'stroke-linecap="round" stroke-linejoin="round" opacity="{0.48 if major_count > 40 else 0.82}"/>'
        )
    for item in draw_order:
        x = float(item["end_x"])
        y = float(item["end_y"])
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{2.8 if major_count > 40 else 4.2}" '
            f'fill="#ffffff" stroke="{item["color"]}" stroke-width="1.7"/>'
        )

    if show_inline_legend:
        legend_x = width - right + 34
        legend_y = top
        parts.append(f'<text x="{legend_x}" y="{legend_y - 24}" class="ylabel">图例（按2030-S2分数排序）</text>')
        legend_items = sorted(series, key=lambda item: (-float(item["end_value"]), str(item["major"])))
        for idx, item in enumerate(legend_items, start=1):
            y = legend_y + (idx - 1) * 35
            label = f'{idx}. {item["major"]}'
            value = fmt(float(item["end_value"]))
            parts.append(f'<line x1="{legend_x}" y1="{y:.2f}" x2="{legend_x + 26}" y2="{y:.2f}" stroke="{item["color"]}" stroke-width="3.5" stroke-linecap="round"/>')
            parts.append(f'<text x="{legend_x + 36}" y="{y + 4:.2f}" class="legend">{html.escape(label)}：{value}</text>')
            parts.append(f'<text x="{legend_x + 36}" y="{y + 19:.2f}" class="legend-small">{html.escape(str(item["door"]))}</text>')

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def legend_rows(majors: dict[str, dict[str, object]], score_field: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = len(majors)
    for index, (major, info) in enumerate(majors.items()):
        period_rows: list[dict[str, str]] = info["rows"]  # type: ignore[assignment]
        values = [float(row[score_field]) for row in period_rows]
        rows.append(
            {
                "selection_id": info["selection_id"],
                "major": major,
                "door": info["door"],
                "color": color_for_index(index, total),
                "first": values[0],
                "last": values[-1],
                "delta": values[-1] - values[0],
            }
        )
    return sorted(rows, key=lambda item: (-float(item["last"]), str(item["major"])))


def legend_table(rows: list[dict[str, object]]) -> str:
    body = []
    for rank, row in enumerate(rows, start=1):
        color = str(row["color"])
        body.append(
            "<tr>"
            f"<td>{rank}</td>"
            f'<td><span class="swatch" style="background:{html.escape(color)}"></span>{html.escape(str(row["selection_id"]))}</td>'
            f"<td>{html.escape(str(row['major']))}</td>"
            f"<td>{html.escape(str(row['door']))}</td>"
            f"<td>{fmt(float(row['first']))}</td>"
            f"<td>{fmt(float(row['last']))}</td>"
            f"<td>{fmt(float(row['delta']))}</td>"
            "</tr>"
        )
    return "\n".join(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render combined major trend charts for employment and salary.")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("output/major_outlook_survey/full_zgc_20260521_2x100/analysis"),
    )
    parser.add_argument("--output-subdir", default="plots/combined")
    args = parser.parse_args()

    analysis_dir = args.analysis_dir
    input_path = analysis_dir / "major_period_mean_scores.csv"
    if not input_path.exists():
        raise SystemExit(f"missing score table: {input_path}")

    majors = read_scores(input_path)
    major_count = len(majors)
    show_inline_legend = major_count <= 30
    output_dir = analysis_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    employment_svg = make_svg(
        title=f"{major_count}个专业就业前景均分趋势",
        subtitle="横轴为半年期，纵轴固定为1-10分；每条折线代表一个专业大类。",
        majors=majors,
        score_field="employment_mean",
        y_label="就业前景均分",
        show_inline_legend=show_inline_legend,
    )
    salary_svg = make_svg(
        title=f"{major_count}个专业薪资前景均分趋势",
        subtitle="横轴为半年期，纵轴固定为1-10分；每条折线代表一个专业大类。",
        majors=majors,
        score_field="salary_mean",
        y_label="薪资前景均分",
        show_inline_legend=show_inline_legend,
    )
    (output_dir / "combined_employment_trends.svg").write_text(employment_svg, encoding="utf-8")
    (output_dir / "combined_salary_trends.svg").write_text(salary_svg, encoding="utf-8")

    index = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>Combined Major Outlook Trend Charts</title>
<style>
body{{font-family:'Noto Sans CJK SC','Microsoft YaHei','PingFang SC',Arial,sans-serif;margin:28px;background:#f5f7fb;color:#172033}}
h1{{font-size:28px;margin:0 0 8px}}
p{{color:#5f6b7a;margin:0 0 20px}}
section{{background:#fff;border:1px solid #d9e0ea;border-radius:8px;margin:0 0 22px;padding:18px}}
img{{width:100%;height:auto;border:1px solid #e5eaf2;border-radius:6px;background:#fff}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid #e5eaf2;padding:7px 8px;text-align:left;white-space:nowrap}}
th{{background:#f8fafc;color:#344054}}
.tablewrap{{max-height:520px;overflow:auto;border:1px solid #e5eaf2;border-radius:6px}}
.swatch{{display:inline-block;width:18px;height:3px;border-radius:2px;margin-right:8px;vertical-align:middle}}
</style>
</head>
<body>
<h1>{major_count}个专业组合趋势图</h1>
<p>两张图分别展示就业前景均分和薪资前景均分，每张图包含{major_count}条专业折线。专业较多时，完整图例放在下方表格中。</p>
<section>
<h2>就业前景</h2>
<a href="combined_employment_trends.svg"><img src="combined_employment_trends.svg" alt="{major_count}个专业就业前景均分趋势"/></a>
</section>
<section>
<h2>薪资前景</h2>
<a href="combined_salary_trends.svg"><img src="combined_salary_trends.svg" alt="{major_count}个专业薪资前景均分趋势"/></a>
</section>
<section>
<h2>就业图例与期末排序</h2>
<div class="tablewrap"><table>
<thead><tr><th>Rank</th><th>ID</th><th>专业</th><th>门类</th><th>2026-S1</th><th>2030-S2</th><th>变化</th></tr></thead>
<tbody>
{legend_table(legend_rows(majors, "employment_mean"))}
</tbody></table></div>
</section>
<section>
<h2>薪资图例与期末排序</h2>
<div class="tablewrap"><table>
<thead><tr><th>Rank</th><th>ID</th><th>专业</th><th>门类</th><th>2026-S1</th><th>2030-S2</th><th>变化</th></tr></thead>
<tbody>
{legend_table(legend_rows(majors, "salary_mean"))}
</tbody></table></div>
</section>
</body>
</html>
"""
    (output_dir / "index.html").write_text(index, encoding="utf-8")

    print(
        {
            "output_dir": str(output_dir),
            "charts": [
                str(output_dir / "combined_employment_trends.svg"),
                str(output_dir / "combined_salary_trends.svg"),
            ],
            "majors": len(majors),
        }
    )


if __name__ == "__main__":
    main()
