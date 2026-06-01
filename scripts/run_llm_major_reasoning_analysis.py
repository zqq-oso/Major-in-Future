from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import litellm
from litellm import Router


litellm.drop_params = True
litellm.suppress_debug_info = True


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

DIMENSIONS = {
    "employment": "就业前景",
    "salary": "薪资前景",
}

MAJOR_KEYWORDS: dict[str, list[str]] = {
    "动物医学类": ["生物", "医药", "动物", "农业", "疫苗", "基因", "检测", "机器人", "AI"],
    "基础医学类": ["生物", "医药", "医疗", "蛋白", "药物", "疫苗", "mRNA", "基因", "AI", "AlphaFold"],
    "历史学类": ["AI", "内容", "数字记忆", "文化", "教育", "伦理", "生成", "版权", "社会"],
    "哲学类": ["AI", "伦理", "治理", "监管", "社会", "内容", "智能体", "人机", "风险"],
    "兵器类": ["国防", "安全", "军", "无人机", "机器人", "网络安全", "量子", "卫星", "芯片", "供应链"],
    "建筑类": ["建筑", "城市", "材料", "能源", "机器人", "自动化", "工程", "基础设施", "3D"],
    "计算机类": ["AI", "大模型", "智能体", "Agent", "Codex", "软件", "数据", "云", "网络安全", "芯片", "量子"],
    "食品科学与工程类": ["食品", "农业", "生物", "检测", "供应链", "材料", "能源", "健康", "自动化"],
    "教育学类": ["教育", "AI", "智能体", "再培训", "学习", "内容", "平台", "人机"],
    "心理学类": ["心理", "健康", "医疗", "教育", "AI", "神经", "脑机", "社会", "情绪"],
    "新闻传播学类": ["新闻", "传播", "媒体", "内容", "生成", "深度伪造", "版权", "社交", "核验"],
    "社会学类": ["社会", "劳动力", "就业", "监管", "平台", "AI", "再培训", "伦理", "治理"],
    "化学类": ["化学", "材料", "能源", "电池", "半导体", "芯片", "医药", "生物", "制造"],
    "统计学类": ["统计", "数据", "AI", "模型", "算法", "风险", "量子", "金融", "计算"],
    "图书情报与档案管理类": ["信息", "知识", "档案", "数据", "内容", "数字记忆", "AI", "治理", "检索"],
    "金融学类": ["金融", "经济", "风险", "监管", "供应链", "支付", "AI代理", "量子", "数据"],
    "综合试验班类": ["跨学科", "AI", "数据", "工程", "产业", "机器人", "生物", "金融", "治理"],
    "戏剧与影视学类": ["影视", "视频", "内容", "生成", "媒体", "版权", "深度伪造", "文化", "AI"],
}

DIMENSION_KEYWORDS = {
    "employment": ["就业", "岗位", "需求", "替代", "自动化", "再培训", "产业", "应用"],
    "salary": ["薪资", "收入", "高薪", "资本", "企业", "利润", "稀缺", "门槛", "产业化"],
}


PROMPT_FIELDS = [
    "task_id",
    "selection_id",
    "door",
    "major_category",
    "dimension",
    "dimension_zh",
    "prompt",
]

PARSED_FIELDS = [
    "task_id",
    "selection_id",
    "door",
    "major_category",
    "dimension",
    "dimension_zh",
    "llm_reasoning",
    "trend_diagnosis",
    "news_drivers",
    "risk_factors",
    "confidence",
]

FAILURE_FIELDS = ["task_id", "selection_id", "major_category", "dimension", "error", "raw_excerpt"]


def clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def make_router(config_path: Path) -> Router:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    return Router(
        model_list=cfg["chat_models"],
        num_retries=3,
        timeout=180,
        retry_after=2,
        routing_strategy="least-busy",
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def read_completed(parsed_path: Path) -> set[str]:
    if not parsed_path.exists():
        return set()
    with parsed_path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["task_id"] for row in csv.DictReader(handle) if row.get("task_id")}


def read_trends(path: Path) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    rows: list[dict[str, str]] = []
    by_major: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            by_major.setdefault(row["major_category"], []).append(row)
    for major_rows in by_major.values():
        major_rows.sort(key=lambda row: int(row["period_order"]))
    return rows, by_major


def score_news_item(item: dict[str, str], keywords: list[str]) -> int:
    text = (item.get("title", "") + " " + item.get("summary", "") + " " + item.get("source_period", "")).lower()
    score = 0
    for keyword in keywords:
        key = keyword.lower()
        if key in text:
            score += 3
    for global_key in ["AI", "智能体", "机器人", "芯片", "量子", "监管", "医疗", "材料", "能源", "数据"]:
        if global_key.lower() in text:
            score += 1
    return score


def select_news_digest(
    news_contexts: dict[str, list[dict[str, str]]],
    major: str,
    dimension: str,
    items_per_period: int,
) -> str:
    keywords = MAJOR_KEYWORDS.get(major, ["AI", "产业", "技术", "就业"])
    keywords = keywords + DIMENSION_KEYWORDS.get(dimension, [])
    lines: list[str] = []
    for period in PERIODS:
        items = news_contexts.get(period, [])
        scored = sorted(
            enumerate(items),
            key=lambda pair: (-score_news_item(pair[1], keywords), pair[0]),
        )
        selected = [item for _idx, item in scored[:items_per_period]]
        lines.append(f"{period}:")
        for item in selected:
            source_period = item.get("source_period", "")
            title = clip(item.get("title", ""), 70)
            summary = clip(item.get("summary", ""), 120)
            lines.append(f"- [{source_period}] {title}：{summary}")
    return "\n".join(lines)


def trend_table(rows: list[dict[str, str]]) -> str:
    lines = ["period | employment_mean | salary_mean | combined_mean | n"]
    for row in rows:
        lines.append(
            f"{row['period']} | {row['employment_mean']} | {row['salary_mean']} | "
            f"{row['combined_mean']} | {row['n_profiles']}"
        )
    return "\n".join(lines)


def target_dimension_summary(rows: list[dict[str, str]], dimension: str) -> str:
    field = "employment_mean" if dimension == "employment" else "salary_mean"
    values = [(row["period"], float(row[field])) for row in rows]
    start_period, start = values[0]
    end_period, end = values[-1]
    min_period, min_value = min(values, key=lambda item: item[1])
    max_period, max_value = max(values, key=lambda item: item[1])
    return (
        f"目标维度序列字段：{field}\n"
        f"起点：{start_period} = {start:.4f}\n"
        f"终点：{end_period} = {end:.4f}\n"
        f"终点-起点：{end - start:+.4f}\n"
        f"最低期：{min_period} = {min_value:.4f}\n"
        f"最高期：{max_period} = {max_value:.4f}"
    )


def build_prompt(
    selection_id: str,
    door: str,
    major: str,
    dimension: str,
    trend_rows: list[dict[str, str]],
    news_digest: str,
) -> str:
    dimension_zh = DIMENSIONS[dimension]
    return f"""你是一个严谨的教育、产业与就业趋势分析师。请基于给定材料，对指定专业的“{dimension_zh}”给出归因分析。

重要边界：
- 新闻材料包含历史事实新闻和本次模拟生成的未来科技新闻上下文。未来新闻只能作为模拟情境信号，不要把它写成已经发生的事实。
- 趋势分数来自 1000 个画像智能体的模拟调查均分，分数范围为 1-10。
- 你的任务不是重新打分，而是解释为什么趋势可能呈现当前形态。
- 如需引用具体分数，只能使用下面“目标维度关键趋势摘要”和“趋势表”里的数字；不要编造、混淆或四舍五入到错误方向。

专业信息：
- selection_id: {selection_id}
- 门类: {door}
- 专业大类: {major}
- 分析维度: {dimension_zh}

趋势表：
{trend_table(trend_rows)}

目标维度关键趋势摘要：
{target_dimension_summary(trend_rows, dimension)}

相关科技新闻/模拟新闻摘要：
{news_digest}

请只输出一个 JSON 对象，字段如下：
{{
  "major_category": "{major}",
  "dimension": "{dimension}",
  "dimension_zh": "{dimension_zh}",
  "trend_diagnosis": "用2-3句话概括该维度从2026-S1到2030-S2的趋势形态，包括高低、升降、波动。",
  "news_drivers": ["列出3-5个由新闻和技术变化支持的主要驱动因素"],
  "risk_factors": ["列出2-4个可能压低该维度分数或造成不确定性的因素"],
  "llm_reasoning": "适当详细的一段中文归因，约250-450字。请同时解释趋势、新闻信号、产业机制和普通学习者/从业者视角。",
  "confidence": "low/medium/high"
}}
"""


def extract_json(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


async def call_llm(router: Router, prompt: str) -> str:
    response = await router.acompletion(
        model="chat",
        temperature=0.25,
        max_tokens=1400,
        messages=[
            {"role": "system", "content": "你是严谨的中文趋势归因分析师。只输出可解析 JSON。"},
            {"role": "user", "content": prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def merge_major_reasoning(parsed_path: Path, out_dir: Path) -> None:
    rows: list[dict[str, str]] = []
    with parsed_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_major: dict[str, dict[str, dict[str, str]]] = {}
    meta: dict[str, dict[str, str]] = {}
    for row in rows:
        major = row["major_category"]
        by_major.setdefault(major, {})[row["dimension"]] = row
        meta.setdefault(major, {"selection_id": row["selection_id"], "door": row["door"], "major_category": major})

    merged_rows: list[dict[str, str]] = []
    for major, dimensions in sorted(by_major.items(), key=lambda item: meta[item[0]]["selection_id"]):
        employment = dimensions.get("employment", {})
        salary = dimensions.get("salary", {})
        combined_reason = (
            f"就业前景归因：{employment.get('llm_reasoning', '')}\n\n"
            f"薪资前景归因：{salary.get('llm_reasoning', '')}"
        ).strip()
        merged_rows.append(
            {
                "selection_id": meta[major]["selection_id"],
                "door": meta[major]["door"],
                "major_category": major,
                "employment_reasoning": employment.get("llm_reasoning", ""),
                "salary_reasoning": salary.get("llm_reasoning", ""),
                "combined_major_reasoning": combined_reason,
            }
        )

    fields = [
        "selection_id",
        "door",
        "major_category",
        "employment_reasoning",
        "salary_reasoning",
        "combined_major_reasoning",
    ]
    append_csv(out_dir / "llm_major_reasoning_merged.csv", merged_rows, fields)

    lines = ["# LLM Major Reasoning Merged", ""]
    for row in merged_rows:
        lines.extend(
            [
                f"## {row['selection_id']} | {row['major_category']} | {row['door']}",
                "",
                "### 就业前景归因",
                "",
                row["employment_reasoning"],
                "",
                "### 薪资前景归因",
                "",
                row["salary_reasoning"],
                "",
            ]
        )
    (out_dir / "llm_major_reasoning_merged.md").write_text("\n".join(lines), encoding="utf-8")


async def run(args: argparse.Namespace) -> None:
    run_dir: Path = args.run_dir
    analysis_dir = run_dir / "analysis"
    out_dir = run_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    _rows, trends_by_major = read_trends(analysis_dir / "major_period_mean_scores.csv")
    news_contexts = json.loads((run_dir / "news_contexts.json").read_text(encoding="utf-8"))
    router = make_router(args.llm_config)

    raw_path = out_dir / "llm_dimension_reasoning_raw.jsonl"
    prompts_path = out_dir / "llm_dimension_reasoning_prompts.jsonl"
    parsed_path = out_dir / "llm_dimension_reasoning.csv"
    failures_path = out_dir / "llm_dimension_reasoning_failures.csv"

    completed = read_completed(parsed_path) if args.resume else set()
    tasks: list[dict[str, str]] = []
    for major, trend_rows in sorted(trends_by_major.items(), key=lambda item: item[1][0]["selection_id"]):
        selection_id = trend_rows[0]["selection_id"]
        door = trend_rows[0]["door"]
        for dimension, dimension_zh in DIMENSIONS.items():
            task_id = f"{selection_id}_{dimension}"
            if task_id in completed:
                continue
            news_digest = select_news_digest(news_contexts, major, dimension, args.news_items_per_period)
            prompt = build_prompt(selection_id, door, major, dimension, trend_rows, news_digest)
            tasks.append(
                {
                    "task_id": task_id,
                    "selection_id": selection_id,
                    "door": door,
                    "major_category": major,
                    "dimension": dimension,
                    "dimension_zh": dimension_zh,
                    "prompt": prompt,
                }
            )

    if not prompts_path.exists():
        for task in tasks:
            append_jsonl(prompts_path, task)

    print(json.dumps({"output_dir": str(out_dir), "remaining_calls": len(tasks)}, ensure_ascii=False, indent=2), flush=True)

    semaphore = asyncio.Semaphore(args.concurrency)

    async def worker(task: dict[str, str]) -> None:
        async with semaphore:
            try:
                raw = await call_llm(router, task["prompt"])
                append_jsonl(
                    raw_path,
                    {
                        "task_id": task["task_id"],
                        "selection_id": task["selection_id"],
                        "door": task["door"],
                        "major_category": task["major_category"],
                        "dimension": task["dimension"],
                        "raw": raw,
                    },
                )
                parsed = extract_json(raw)
                append_csv(
                    parsed_path,
                    [
                        {
                            "task_id": task["task_id"],
                            "selection_id": task["selection_id"],
                            "door": task["door"],
                            "major_category": task["major_category"],
                            "dimension": task["dimension"],
                            "dimension_zh": task["dimension_zh"],
                            "llm_reasoning": parsed.get("llm_reasoning", ""),
                            "trend_diagnosis": parsed.get("trend_diagnosis", ""),
                            "news_drivers": json.dumps(parsed.get("news_drivers", []), ensure_ascii=False),
                            "risk_factors": json.dumps(parsed.get("risk_factors", []), ensure_ascii=False),
                            "confidence": parsed.get("confidence", ""),
                        }
                    ],
                    PARSED_FIELDS,
                )
            except Exception as exc:
                append_csv(
                    failures_path,
                    [
                        {
                            "task_id": task["task_id"],
                            "selection_id": task["selection_id"],
                            "major_category": task["major_category"],
                            "dimension": task["dimension"],
                            "error": f"{type(exc).__name__}: {exc}",
                            "raw_excerpt": "",
                        }
                    ],
                    FAILURE_FIELDS,
                )
            await asyncio.sleep(random.random() * 0.2)

    await asyncio.gather(*(worker(task) for task in tasks))

    if parsed_path.exists():
        merged_csv = out_dir / "llm_major_reasoning_merged.csv"
        if merged_csv.exists():
            merged_csv.unlink()
        merge_major_reasoning(parsed_path, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 18 x 2 LLM reasoning analysis calls for major trends.")
    parser.add_argument("--run-dir", type=Path, default=Path("output/major_outlook_survey/full_zgc_20260521_2x100"))
    parser.add_argument("--output-subdir", default="analysis/reasoning")
    parser.add_argument("--llm-config", type=Path, default=Path("configs/llm/llm_config.zgc.deepseek_v4_flash_2x100.json"))
    parser.add_argument("--concurrency", type=int, default=18)
    parser.add_argument("--news-items-per-period", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
