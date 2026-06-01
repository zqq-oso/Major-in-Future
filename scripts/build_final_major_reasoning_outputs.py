from __future__ import annotations

import argparse
import csv
import html
import re
from pathlib import Path


def read_by_major(path: Path, key: str = "major_category") -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def exact_trend(row: dict[str, str], prefix: str) -> str:
    start = row[f"{prefix}_2026S1"]
    end = row[f"{prefix}_2030S2"]
    delta = row[f"{prefix}_delta_2030S2_minus_2026S1"]
    slope = row[f"{prefix}_slope_per_half_year"]
    return f"2026-S1={start}，2030-S2={end}，变化={delta}，半年期斜率={slope}"


def sanitize_llm_text(text: str) -> str:
    # Keep exact numeric scores in dedicated trend columns; avoid stale free-text score mentions.
    return re.sub(r"(?<!\d)[1-9]\.\d{2,4}(?![\d%])", "相应分值", text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final one-row-per-major reasoning outputs.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("output/major_outlook_survey/full_zgc_20260521_2x100"),
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    reasoning_dir = run_dir / "analysis" / "reasoning"
    agent = read_by_major(reasoning_dir / "agent_reasoning_aggregated_by_major.csv")
    llm = read_by_major(reasoning_dir / "llm_major_reasoning_merged.csv")
    trends = read_by_major(run_dir / "analysis" / "major_trend_summary.csv")

    rows: list[dict[str, str]] = []
    for major, trend in sorted(trends.items(), key=lambda item: item[1]["selection_id"]):
        agent_row = agent[major]
        llm_row = llm[major]
        llm_combined = sanitize_llm_text(llm_row["combined_major_reasoning"])
        rows.append(
            {
                "selection_id": trend["selection_id"],
                "door": trend["door"],
                "major_category": major,
                "employment_exact_trend": exact_trend(trend, "employment"),
                "salary_exact_trend": exact_trend(trend, "salary"),
                "combined_exact_trend": exact_trend(trend, "combined"),
                "agent_aggregated_reasoning": agent_row["agent_aggregated_reasoning"],
                "llm_news_trend_reasoning": llm_combined,
            }
        )

    fields = [
        "selection_id",
        "door",
        "major_category",
        "employment_exact_trend",
        "salary_exact_trend",
        "combined_exact_trend",
        "agent_aggregated_reasoning",
        "llm_news_trend_reasoning",
    ]
    write_csv(reasoning_dir / "major_reasoning_final.csv", rows, fields)

    lines = [
        "# Final Major Reasoning",
        "",
        "Each major has two reasoning sources:",
        "",
        "1. `agent_aggregated_reasoning`: deterministic aggregation of all profile-agent reasons.",
        "2. `llm_news_trend_reasoning`: 18 x 2 LLM calls using technology-news context plus score trends.",
        "",
        "Exact scores should be taken from the exact-trend lines below and the CSV score tables, not from free-form explanatory text.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['selection_id']} | {row['major_category']} | {row['door']}",
                "",
                f"- Employment: {row['employment_exact_trend']}",
                f"- Salary: {row['salary_exact_trend']}",
                f"- Combined: {row['combined_exact_trend']}",
                "",
                "### Agent Aggregated Reasoning",
                "",
                row["agent_aggregated_reasoning"],
                "",
                "### LLM News + Trend Reasoning",
                "",
                row["llm_news_trend_reasoning"],
                "",
            ]
        )
    (reasoning_dir / "major_reasoning_final.md").write_text("\n".join(lines), encoding="utf-8")

    html_lines = [
        "<!doctype html>",
        '<html lang="zh-CN"><head><meta charset="utf-8"/>',
        "<title>Final Major Reasoning</title>",
        "<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:28px;line-height:1.6;color:#172033}section{border-bottom:1px solid #ddd;padding:18px 0}h1{margin-top:0}h2{font-size:20px}.trend{background:#f5f7fb;padding:10px;border-radius:6px}</style>",
        "</head><body>",
        "<h1>Final Major Reasoning</h1>",
    ]
    for row in rows:
        html_lines.extend(
            [
                "<section>",
                f"<h2>{html.escape(row['selection_id'])} | {html.escape(row['major_category'])} | {html.escape(row['door'])}</h2>",
                '<div class="trend">',
                f"<p><b>Employment:</b> {html.escape(row['employment_exact_trend'])}</p>",
                f"<p><b>Salary:</b> {html.escape(row['salary_exact_trend'])}</p>",
                f"<p><b>Combined:</b> {html.escape(row['combined_exact_trend'])}</p>",
                "</div>",
                "<h3>Agent Aggregated Reasoning</h3>",
                f"<p>{html.escape(row['agent_aggregated_reasoning'])}</p>",
                "<h3>LLM News + Trend Reasoning</h3>",
                f"<p>{html.escape(row['llm_news_trend_reasoning']).replace(chr(10), '<br/>')}</p>",
                "</section>",
            ]
        )
    html_lines.extend(["</body></html>"])
    (reasoning_dir / "major_reasoning_final.html").write_text("\n".join(html_lines), encoding="utf-8")

    print({"output_dir": str(reasoning_dir), "rows": len(rows)})


if __name__ == "__main__":
    main()
