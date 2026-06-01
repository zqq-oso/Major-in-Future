from __future__ import annotations

import argparse
import colorsys
import csv
import html
import re
from collections import defaultdict
from pathlib import Path


def slugify(text: str) -> str:
    text = re.sub(r"[\\/:*?\"<>|\s]+", "_", text.strip())
    return text.strip("_") or "door"


def fmt(value: float) -> str:
    return f"{value:.2f}"


def color_for_index(index: int, total: int) -> str:
    hue = 0.58 if total <= 1 else (index * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.58, 0.72)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def read_rows(path: Path) -> dict[str, dict[str, dict[str, object]]]:
    by_door: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    rows_by_major: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_major[row["major_category"]].append(row)

    for major, rows in rows_by_major.items():
        rows.sort(key=lambda row: int(row["period_order"]))
        door = rows[0]["door"]
        by_door[door][major] = {
            "selection_id": rows[0]["selection_id"],
            "door": door,
            "rows": rows,
        }
    return {
        door: dict(sorted(majors.items(), key=lambda item: str(item[1]["selection_id"])))
        for door, majors in sorted(by_door.items(), key=lambda item: str(next(iter(item[1].values()))["selection_id"]))
    }


def legend_rows(majors: dict[str, dict[str, object]], score_field: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = len(majors)
    for index, (major, info) in enumerate(majors.items()):
        period_rows: list[dict[str, str]] = info["rows"]  # type: ignore[assignment]
        values = [float(row[score_field]) for row in period_rows]
        rows.append(
            {
                "rank_value": values[-1],
                "selection_id": info["selection_id"],
                "major": major,
                "color": color_for_index(index, total),
                "first": values[0],
                "last": values[-1],
                "delta": values[-1] - values[0],
            }
        )
    return sorted(rows, key=lambda item: (-float(item["last"]), str(item["major"])))


def legend_table(rows: list[dict[str, object]]) -> str:
    body: list[str] = []
    for rank, row in enumerate(rows, start=1):
        color = html.escape(str(row["color"]))
        body.append(
            "<tr>"
            f"<td>{rank}</td>"
            f'<td><span class="swatch" style="background:{color}"></span>{html.escape(str(row["selection_id"]))}</td>'
            f"<td>{html.escape(str(row['major']))}</td>"
            f"<td>{fmt(float(row['first']))}</td>"
            f"<td>{fmt(float(row['last']))}</td>"
            f"<td>{fmt(float(row['delta']))}</td>"
            "</tr>"
        )
    return "\n".join(body)


def make_svg(
    title: str,
    subtitle: str,
    majors: dict[str, dict[str, object]],
    score_field: str,
    y_label: str,
) -> str:
    major_count = len(majors)
    left = 82
    right = 430
    top = 92
    bottom = 92
    legend_step = 28 if major_count > 24 else 35
    width = 1440
    height = max(760, top + bottom + 32 + major_count * legend_step)
    plot_w = width - left - right
    plot_h = height - top - bottom

    first_major = next(iter(majors.values()))
    periods = [row["period"] for row in first_major["rows"]]  # type: ignore[index]

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

    series: list[dict[str, object]] = []
    for index, (major, info) in enumerate(majors.items()):
        period_rows: list[dict[str, str]] = info["rows"]  # type: ignore[assignment]
        values = [float(row[score_field]) for row in period_rows]
        points = [(x_pos(i), y_pos(value)) for i, value in enumerate(values)]
        path_d = " ".join(("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}" for i, (x, y) in enumerate(points))
        series.append(
            {
                "major": major,
                "selection_id": info["selection_id"],
                "color": color_for_index(index, major_count),
                "path": path_d,
                "end_x": points[-1][0],
                "end_y": points[-1][1],
                "end_value": values[-1],
            }
        )

    for item in sorted(series, key=lambda row: float(row["end_value"])):
        parts.append(
            f'<path d="{item["path"]}" fill="none" stroke="{item["color"]}" '
            f'stroke-width="{2.0 if major_count > 24 else 2.8}" stroke-linecap="round" '
            f'stroke-linejoin="round" opacity="{0.64 if major_count > 24 else 0.84}"/>'
        )
    for item in series:
        parts.append(
            f'<circle cx="{float(item["end_x"]):.2f}" cy="{float(item["end_y"]):.2f}" r="{3.2 if major_count > 24 else 4.2}" '
            f'fill="#ffffff" stroke="{item["color"]}" stroke-width="1.7"/>'
        )

    legend_x = width - right + 34
    legend_y = top
    legend_period = periods[-1]
    parts.append(f'<text x="{legend_x}" y="{legend_y - 24}" class="ylabel">图例（按{html.escape(legend_period)}排序）</text>')
    for rank, item in enumerate(sorted(series, key=lambda row: (-float(row["end_value"]), str(row["major"]))), start=1):
        y = legend_y + (rank - 1) * legend_step
        value = fmt(float(item["end_value"]))
        label = f'{rank}. {item["major"]}'
        parts.append(f'<line x1="{legend_x}" y1="{y:.2f}" x2="{legend_x + 26}" y2="{y:.2f}" stroke="{item["color"]}" stroke-width="3.5" stroke-linecap="round"/>')
        parts.append(f'<text x="{legend_x + 36}" y="{y + 4:.2f}" class="legend">{html.escape(label)}：{value}</text>')
        parts.append(f'<text x="{legend_x + 36}" y="{y + 19:.2f}" class="legend-small">{html.escape(str(item["selection_id"]))}</text>')

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render per-door major trend charts.")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("output/your_run/analysis"),
    )
    parser.add_argument("--output-subdir", default="plots/by_door")
    args = parser.parse_args()

    analysis_dir = args.analysis_dir
    input_path = analysis_dir / "major_period_mean_scores.csv"
    if not input_path.exists():
        raise SystemExit(f"missing score table: {input_path}")

    by_door = read_rows(input_path)
    output_dir = analysis_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    index_items: list[dict[str, object]] = []
    for door, majors in by_door.items():
        slug = slugify(door)
        major_count = len(majors)
        employment_file = f"{slug}_employment.svg"
        salary_file = f"{slug}_salary.svg"
        subtitle = f"门类：{door} | 专业数：{major_count} | 样本数：每专业每期 n=100 | 分数范围固定为 1-10"

        (output_dir / employment_file).write_text(
            make_svg(
                title=f"{door}：就业前景均分趋势",
                subtitle=subtitle,
                majors=majors,
                score_field="employment_mean",
                y_label="就业前景均分",
            ),
            encoding="utf-8",
        )
        (output_dir / salary_file).write_text(
            make_svg(
                title=f"{door}：薪酬前景均分趋势",
                subtitle=subtitle,
                majors=majors,
                score_field="salary_mean",
                y_label="薪酬前景均分",
            ),
            encoding="utf-8",
        )
        index_items.append(
            {
                "door": door,
                "major_count": major_count,
                "employment_file": employment_file,
                "salary_file": salary_file,
                "employment_legend": legend_rows(majors, "employment_mean"),
                "salary_legend": legend_rows(majors, "salary_mean"),
            }
        )

    html_parts = [
        "<!doctype html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8"/>',
        "<title>By-Door Major Outlook Trend Charts</title>",
        "<style>",
        "body{font-family:'Noto Sans CJK SC','Microsoft YaHei','PingFang SC',Arial,sans-serif;margin:28px;background:#f5f7fb;color:#172033}",
        "h1{font-size:28px;margin:0 0 8px}",
        "p{color:#5f6b7a;margin:0 0 20px}",
        "section{background:#fff;border:1px solid #d9e0ea;border-radius:8px;margin:0 0 22px;padding:18px}",
        "h2{font-size:20px;margin:0 0 14px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:16px}",
        "img{width:100%;height:auto;border:1px solid #e5eaf2;border-radius:6px;background:#fff}",
        "details{margin-top:14px}",
        "summary{cursor:pointer;color:#344054;font-weight:600;margin:6px 0}",
        "table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 14px}",
        "th,td{border-bottom:1px solid #e5eaf2;padding:7px 8px;text-align:left;white-space:nowrap}",
        "th{background:#f8fafc;color:#344054}",
        ".tablewrap{max-height:360px;overflow:auto;border:1px solid #e5eaf2;border-radius:6px}",
        ".swatch{display:inline-block;width:18px;height:3px;border-radius:2px;margin-right:8px;vertical-align:middle}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>按学科门类的专业前景趋势图</h1>",
        "<p>每个学科门类两张图：就业前景均分、薪酬前景均分。横轴为半年期，纵轴固定为 1-10 分；每张 SVG 右侧直接标注专业大类名称。</p>",
    ]

    for item in index_items:
        door = html.escape(str(item["door"]))
        html_parts.extend(
            [
                "<section>",
                f"<h2>{door}（{item['major_count']} 个专业）</h2>",
                '<div class="grid">',
                f'<a href="{html.escape(str(item["employment_file"]))}"><img src="{html.escape(str(item["employment_file"]))}" alt="{door} 就业前景"/></a>',
                f'<a href="{html.escape(str(item["salary_file"]))}"><img src="{html.escape(str(item["salary_file"]))}" alt="{door} 薪酬前景"/></a>',
                "</div>",
                "<details>",
                "<summary>就业图例与2030-S2排序</summary>",
                '<div class="tablewrap"><table><thead><tr><th>Rank</th><th>ID</th><th>专业</th><th>2026-S1</th><th>2030-S2</th><th>变化</th></tr></thead><tbody>',
                legend_table(item["employment_legend"]),  # type: ignore[arg-type]
                "</tbody></table></div>",
                "</details>",
                "<details>",
                "<summary>薪酬图例与2030-S2排序</summary>",
                '<div class="tablewrap"><table><thead><tr><th>Rank</th><th>ID</th><th>专业</th><th>2026-S1</th><th>2030-S2</th><th>变化</th></tr></thead><tbody>',
                legend_table(item["salary_legend"]),  # type: ignore[arg-type]
                "</tbody></table></div>",
                "</details>",
                "</section>",
            ]
        )

    html_parts.extend(["</body>", "</html>"])
    (output_dir / "index.html").write_text("\n".join(html_parts) + "\n", encoding="utf-8")

    print(
        {
            "output_dir": str(output_dir),
            "doors": len(index_items),
            "charts": len(index_items) * 2,
            "index": str(output_dir / "index.html"),
        }
    )


if __name__ == "__main__":
    main()
