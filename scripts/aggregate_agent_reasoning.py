from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


THEMES: dict[str, list[str]] = {
    "AI与自动化替代/赋能": ["AI", "人工智能", "大模型", "自动化", "智能", "机器人", "算法", "编程", "数据"],
    "数字化与平台经济需求": ["数字", "互联网", "平台", "电商", "软件", "系统", "云", "线上", "信息化"],
    "产业升级与实体需求": ["产业", "制造", "工厂", "供应链", "工程", "材料", "能源", "芯片", "半导体"],
    "医疗健康与老龄化": ["医疗", "医院", "医生", "药", "健康", "养老", "老龄", "生物", "疫苗", "蛋白"],
    "政策/公共部门/安全驱动": ["政策", "国家", "政府", "安全", "军工", "国防", "监管", "事业编", "体制"],
    "就业面与岗位数量": ["就业", "岗位", "工作", "需求", "招人", "好找", "机会", "缺口"],
    "薪资上限与变现能力": ["薪资", "工资", "收入", "挣钱", "赚钱", "高薪", "待遇", "钱"],
    "学历门槛与专业壁垒": ["学历", "研究生", "博士", "门槛", "证书", "资格", "专业性", "技术门槛"],
    "竞争/饱和/下行压力": ["竞争", "饱和", "内卷", "替代", "裁员", "下滑", "难", "压力"],
    "地域/普通家庭可及性": ["普通人", "农村", "城市", "基层", "本地", "家里", "小地方"],
    "稳定性与传统职业认知": ["稳定", "老师", "公务员", "编制", "传统", "长期", "铁饭碗"],
    "创意/人文/传播转型": ["创意", "媒体", "新闻", "传播", "内容", "影视", "文化", "历史", "哲学"],
}


def clean_reason(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text


def score_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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


def fmt(value: float) -> str:
    return f"{value:.4f}"


def theme_counts(reasons: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for reason in reasons:
        for theme, keywords in THEMES.items():
            if any(keyword in reason for keyword in keywords):
                counts[theme] += 1
    return counts


def select_examples(counter: Counter[str], limit: int = 5) -> list[str]:
    examples: list[str] = []
    for reason, _count in counter.most_common():
        if len(reason) < 8:
            continue
        if any(reason in old or old in reason for old in examples):
            continue
        examples.append(reason)
        if len(examples) >= limit:
            break
    return examples


def trend_phrase(start: float, end: float, trend_slope: float) -> str:
    delta = end - start
    if delta >= 0.5:
        direction = "明显上升"
    elif delta >= 0.1:
        direction = "小幅上升"
    elif delta <= -0.5:
        direction = "明显下降"
    elif delta <= -0.1:
        direction = "小幅下降"
    else:
        direction = "基本稳定"
    slope_desc = "持续向上" if trend_slope > 0.03 else "持续向下" if trend_slope < -0.03 else "波动不大"
    return f"从 {start:.2f} 到 {end:.2f}，变化 {delta:+.2f}，整体呈{direction}，线性斜率显示{ slope_desc }。"


def make_agent_summary(
    major: str,
    door: str,
    reason_count: int,
    theme_counter: Counter[str],
    examples: list[str],
    trend: dict[str, str],
) -> str:
    top_themes = theme_counter.most_common(5)
    theme_text = "、".join(f"{theme}({count}次)" for theme, count in top_themes) if top_themes else "无明显高频主题"
    example_text = "；".join(examples[:3])
    combined = trend_phrase(
        float(trend["combined_2026S1"]),
        float(trend["combined_2030S2"]),
        float(trend["combined_slope_per_half_year"]),
    )
    employment = trend_phrase(
        float(trend["employment_2026S1"]),
        float(trend["employment_2030S2"]),
        float(trend["employment_slope_per_half_year"]),
    )
    salary = trend_phrase(
        float(trend["salary_2026S1"]),
        float(trend["salary_2030S2"]),
        float(trend["salary_slope_per_half_year"]),
    )
    return (
        f"{major}（{door}）的群体解释共汇总 {reason_count} 条有效 reasoning。智能体最常提到的归因主题是：{theme_text}。"
        f"从分数看，综合前景{combined}就业维度{employment}薪资维度{salary}"
        f"综合这些理由，群体判断主要围绕“真实岗位需求、技术或产业变化是否扩大专业应用场景、薪资是否有高端岗位支撑、以及普通个体进入该领域的门槛”展开。"
        f"代表性原句包括：{example_text}。"
    )


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate all profile-agent reasoning by major.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("output/major_outlook_survey/full_zgc_20260521_2x100"),
    )
    parser.add_argument("--output-subdir", default="analysis/reasoning")
    args = parser.parse_args()

    run_dir = args.run_dir
    parsed_path = run_dir / "parsed_ratings.csv"
    trend_path = run_dir / "analysis" / "major_trend_summary.csv"
    if not parsed_path.exists():
        raise SystemExit(f"missing parsed ratings: {parsed_path}")
    if not trend_path.exists():
        raise SystemExit(f"missing trend summary: {trend_path}")

    trends: dict[str, dict[str, str]] = {}
    with trend_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            trends[row["major_category"]] = row

    reasons_by_major: dict[str, list[str]] = defaultdict(list)
    reason_counter_by_major: dict[str, Counter[str]] = defaultdict(Counter)
    meta_by_major: dict[str, dict[str, str]] = {}
    with parsed_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            major = row["major_category"]
            reason = clean_reason(row.get("reason", ""))
            if not reason:
                continue
            reasons_by_major[major].append(reason)
            reason_counter_by_major[major][reason] += 1
            meta_by_major.setdefault(
                major,
                {
                    "selection_id": row["selection_id"],
                    "door": row["door"],
                    "major_category": major,
                },
            )

    output_rows: list[dict[str, object]] = []
    detail: dict[str, object] = {}
    for major, reasons in sorted(reasons_by_major.items(), key=lambda item: meta_by_major[item[0]]["selection_id"]):
        meta = meta_by_major[major]
        themes = theme_counts(reasons)
        examples = select_examples(reason_counter_by_major[major], limit=8)
        trend = trends[major]
        top_themes = themes.most_common(8)
        top_exact = reason_counter_by_major[major].most_common(8)
        summary = make_agent_summary(major, meta["door"], len(reasons), themes, examples, trend)
        row = {
            "selection_id": meta["selection_id"],
            "door": meta["door"],
            "major_category": major,
            "reason_count": len(reasons),
            "top_themes": "；".join(f"{name}:{count}" for name, count in top_themes),
            "top_exact_reasons": "；".join(f"{reason}({count})" for reason, count in top_exact[:5]),
            "agent_aggregated_reasoning": summary,
            "combined_2026S1": trend["combined_2026S1"],
            "combined_2030S2": trend["combined_2030S2"],
            "combined_delta": trend["combined_delta_2030S2_minus_2026S1"],
        }
        output_rows.append(row)
        detail[major] = {
            "meta": meta,
            "reason_count": len(reasons),
            "top_themes": top_themes,
            "top_exact_reasons": top_exact,
            "examples": examples,
            "trend": trend,
            "agent_aggregated_reasoning": summary,
        }

    out_dir = run_dir / args.output_subdir
    fields = [
        "selection_id",
        "door",
        "major_category",
        "reason_count",
        "top_themes",
        "top_exact_reasons",
        "agent_aggregated_reasoning",
        "combined_2026S1",
        "combined_2030S2",
        "combined_delta",
    ]
    write_csv(out_dir / "agent_reasoning_aggregated_by_major.csv", output_rows, fields)
    (out_dir / "agent_reasoning_aggregated_by_major.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Agent Reasoning Aggregated By Major",
        "",
        f"- Source: `{parsed_path}`",
        f"- Majors: {len(output_rows)}",
        "",
    ]
    for row in output_rows:
        lines.extend(
            [
                f"## {row['selection_id']} | {row['major_category']} | {row['door']}",
                "",
                str(row["agent_aggregated_reasoning"]),
                "",
                f"- Top themes: {row['top_themes']}",
                "",
            ]
        )
    (out_dir / "agent_reasoning_aggregated_by_major.md").write_text("\n".join(lines), encoding="utf-8")

    index_lines = [
        "<!doctype html>",
        '<html lang="zh-CN"><head><meta charset="utf-8"/>',
        "<title>Agent Reasoning Aggregation</title>",
        "<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:28px;line-height:1.6;color:#172033}section{border-bottom:1px solid #ddd;padding:16px 0}h1{margin-top:0}h2{font-size:20px}</style>",
        "</head><body>",
        "<h1>Agent Reasoning Aggregated By Major</h1>",
    ]
    for row in output_rows:
        index_lines.extend(
            [
                "<section>",
                f"<h2>{html.escape(str(row['selection_id']))} | {html.escape(str(row['major_category']))} | {html.escape(str(row['door']))}</h2>",
                f"<p>{html.escape(str(row['agent_aggregated_reasoning']))}</p>",
                f"<p><b>Top themes:</b> {html.escape(str(row['top_themes']))}</p>",
                "</section>",
            ]
        )
    index_lines.extend(["</body></html>"])
    (out_dir / "agent_reasoning_aggregated_by_major.html").write_text("\n".join(index_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "majors": len(output_rows),
                "total_reasons": sum(len(v) for v in reasons_by_major.values()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
