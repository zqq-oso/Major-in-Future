from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import dotenv
import litellm
from litellm import Router


dotenv.load_dotenv()
litellm.drop_params = True
litellm.suppress_debug_info = True

MAJOR_FIELDS = ["selection_id", "door", "major_category", "child_count", "source_row_index"]
PARSED_FIELDS = [
    "period",
    "profile_id",
    "source_id",
    "province",
    "age",
    "gender",
    "education",
    "consumption",
    "selection_id",
    "door",
    "major_category",
    "employment_score",
    "salary_score",
    "reason",
]
FAILURE_FIELDS = ["period", "profile_id", "attempt", "error", "raw_excerpt"]


def build_half_year_sequence(start: str, end: str) -> list[str]:
    year, half = start.split("-S", 1)
    current_year = int(year)
    current_half = int(half)
    periods: list[str] = []
    while True:
        period = f"{current_year}-S{current_half}"
        periods.append(period)
        if period == end:
            return periods
        if current_half == 1:
            current_half = 2
        else:
            current_half = 1
            current_year += 1


def parse_major_categories(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|") or "---" in line or "序号" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 4:
            continue
        try:
            source_row_index = int(parts[0])
            child_count = int(parts[3])
        except ValueError:
            continue
        rows.append(
            {
                "source_row_index": source_row_index,
                "door": parts[1],
                "major_category": parts[2],
                "child_count": child_count,
            }
        )
    return rows


def select_majors_by_door(rows: list[dict[str, Any]], ratio: float, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_door: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_door.setdefault(row["door"], []).append(row)

    selected: list[dict[str, Any]] = []
    for door in sorted(by_door):
        candidates = sorted(by_door[door], key=lambda item: item["source_row_index"])
        k = max(1, math.ceil(len(candidates) * ratio))
        sampled = rng.sample(candidates, k=min(k, len(candidates)))
        for row in sorted(sampled, key=lambda item: item["source_row_index"]):
            selected.append(
                {
                    "selection_id": f"major_{len(selected) + 1:03d}",
                    "door": row["door"],
                    "major_category": row["major_category"],
                    "child_count": row["child_count"],
                    "source_row_index": row["source_row_index"],
                }
            )
    return selected


def load_profiles(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    profiles = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        profiles = profiles[:limit]
    return profiles


def load_news_by_period(news_dir: Path) -> dict[str, list[dict[str, str]]]:
    news: dict[str, list[dict[str, str]]] = {}
    for path in sorted(news_dir.glob("*.jsonl")):
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        news[path.stem] = rows
    return news


def half_year_for_date(date_text: str) -> str:
    parts = date_text.split("-")
    year = int(parts[0])
    month = int(parts[1])
    return f"{year}-S{1 if month <= 6 else 2}"


def load_seed_news_csv(path: Path) -> dict[str, list[dict[str, str]]]:
    news: dict[str, list[dict[str, str]]] = {}
    if not path.exists():
        return news
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            period = half_year_for_date(row["时间"])
            news.setdefault(period, []).append(
                {
                    "date": row["时间"],
                    "domain": row.get("领域", ""),
                    "category": row.get("事件类别", ""),
                    "title": row["新闻标题"],
                    "summary": row["新闻摘要"],
                }
            )
    return news


def merge_historical_seed_news(
    run_news: dict[str, list[dict[str, str]]],
    seed_news: dict[str, list[dict[str, str]]],
    history_end: str,
) -> dict[str, list[dict[str, str]]]:
    merged = dict(run_news)
    for period, rows in seed_news.items():
        if period <= history_end:
            merged[period] = rows
    return merged


def news_context_periods(all_periods: list[str], period: str, previous_periods: int) -> list[str]:
    idx = all_periods.index(period)
    return all_periods[max(0, idx - previous_periods): idx + 1]


def build_news_contexts(
    news_by_period: dict[str, list[dict[str, str]]],
    all_periods: list[str],
    survey_periods: list[str],
    previous_periods: int,
    items_per_period: int,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    contexts: dict[str, list[dict[str, str]]] = {}
    for period in survey_periods:
        rows: list[dict[str, str]] = []
        for source_period in news_context_periods(all_periods, period, previous_periods):
            pool = news_by_period.get(source_period, [])
            rng = random.Random(f"{seed}:{period}:{source_period}")
            sampled = rng.sample(pool, min(items_per_period, len(pool))) if pool else []
            for item in sampled:
                rows.append(
                    {
                        "source_period": source_period,
                        "date": str(item.get("date", source_period)),
                        "title": str(item.get("title", "")),
                        "summary": str(item.get("summary", "")),
                    }
                )
        contexts[period] = rows
    return contexts


def compact_news_context(rows: list[dict[str, str]], summary_chars: int) -> str:
    lines = []
    for idx, row in enumerate(rows, start=1):
        summary = re.sub(r"\s+", " ", row["summary"]).strip()
        if summary_chars > 0 and len(summary) > summary_chars:
            summary = summary[:summary_chars].rstrip() + "..."
        lines.append(f"{idx}. [{row['source_period']}] {row['title']}：{summary}")
    return "\n".join(lines)


def score_context_periods(survey_periods: list[str], period: str, history_periods: int) -> list[str]:
    if history_periods <= 0:
        return []
    idx = survey_periods.index(period)
    return survey_periods[max(0, idx - history_periods):idx]


def compute_period_score_summary(
    parsed_path: Path,
    period: str,
    selected_majors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    accum: dict[str, dict[str, Any]] = {
        row["selection_id"]: {
            "selection_id": row["selection_id"],
            "door": row["door"],
            "major_category": row["major_category"],
            "n_profiles": 0,
            "employment_sum": 0.0,
            "salary_sum": 0.0,
        }
        for row in selected_majors
    }
    if parsed_path.exists():
        with parsed_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("period") != period:
                    continue
                selection_id = row.get("selection_id", "")
                if selection_id not in accum:
                    continue
                try:
                    employment = float(row["employment_score"])
                    salary = float(row["salary_score"])
                except Exception:
                    continue
                accum[selection_id]["n_profiles"] += 1
                accum[selection_id]["employment_sum"] += employment
                accum[selection_id]["salary_sum"] += salary

    summary: list[dict[str, Any]] = []
    for row in selected_majors:
        item = accum[row["selection_id"]]
        n = item["n_profiles"]
        summary.append(
            {
                "selection_id": item["selection_id"],
                "door": item["door"],
                "major_category": item["major_category"],
                "n_profiles": n,
                "employment_mean": round(item["employment_sum"] / n, 4) if n else None,
                "salary_mean": round(item["salary_sum"] / n, 4) if n else None,
            }
        )
    return summary


def build_score_contexts_for_period(
    parsed_path: Path,
    period: str,
    survey_periods: list[str],
    selected_majors: list[dict[str, Any]],
    history_periods: int,
) -> dict[str, list[dict[str, Any]]]:
    contexts: dict[str, list[dict[str, Any]]] = {}
    for source_period in score_context_periods(survey_periods, period, history_periods):
        contexts[source_period] = compute_period_score_summary(parsed_path, source_period, selected_majors)
    return contexts


def compact_period_aligned_context(
    news_rows: list[dict[str, str]],
    score_context: dict[str, list[dict[str, Any]]],
    summary_chars: int,
) -> str:
    news_by_period: dict[str, list[dict[str, str]]] = {}
    period_order: list[str] = []
    for row in news_rows:
        source_period = row["source_period"]
        if source_period not in news_by_period:
            period_order.append(source_period)
            news_by_period[source_period] = []
        news_by_period[source_period].append(row)

    for source_period in score_context:
        if source_period not in news_by_period:
            period_order.append(source_period)
            news_by_period[source_period] = []

    lines: list[str] = []
    for source_period in period_order:
        lines.append(f"### {source_period}")
        score_rows = score_context.get(source_period)
        if score_rows:
            lines.append("该期已完成调查的专业均分（供当前期判断参考，不包含当前期结果）：")
            for item in score_rows:
                employment = item["employment_mean"]
                salary = item["salary_mean"]
                if employment is None or salary is None:
                    score_text = "样本不足，暂无均分"
                else:
                    score_text = f"就业均分 {employment:.4f}，薪资均分 {salary:.4f}，n={item['n_profiles']}"
                lines.append(f"- {item['selection_id']} | {item['major_category']}：{score_text}")
        else:
            lines.append("该期没有可用的历史专业均分。")

        rows = news_by_period.get(source_period, [])
        if rows:
            lines.append("该期可见科技新闻：")
            for idx, row in enumerate(rows, start=1):
                summary = re.sub(r"\s+", " ", row["summary"]).strip()
                if summary_chars > 0 and len(summary) > summary_chars:
                    summary = summary[:summary_chars].rstrip() + "..."
                lines.append(f"{idx}. {row['title']}：{summary}")
        else:
            lines.append("该期没有抽取到新闻。")
        lines.append("")
    return "\n".join(lines).strip()


def profile_text(profile: dict[str, Any]) -> str:
    if profile.get("profile_text"):
        return str(profile["profile_text"])
    base = (
        f"省份：{profile.get('province')}，年龄：{profile.get('age')}，"
        f"性别：{profile.get('gender')}，学历：{profile.get('education')}，"
        f"消费水平：{profile.get('consumption')}"
    )
    if profile.get("post_snippet"):
        base += f"\n该用户近期发言摘录：「{profile['post_snippet']}」"
    return base


def make_prompt(
    period: str,
    profile: dict[str, Any],
    majors: list[dict[str, Any]],
    evidence_context: str,
    score_history_periods: int = 0,
) -> str:
    major_lines = [
        f"- {row['selection_id']} | 门类：{row['door']} | 专业大类：{row['major_category']}"
        for row in majors
    ]
    if score_history_periods > 0:
        context_intro = (
            f"可见信息按半年期分组。每组包含该期科技新闻；如果该期已经完成过调查，还包含该期各专业的就业/薪资统计均分。\n"
            f"专业均分最多来自当前期之前{score_history_periods}个半年期，不包含当前期结果；请把同一时期的均分和新闻对应起来理解。\n"
        )
        context_label = "可见科技新闻与历史专业均分："
    else:
        context_intro = "可见科技新闻概况（当前期及此前6个半年期，每期随机抽取若干条）：\n"
        context_label = ""
    return (
        f"调查时期：{period}\n\n"
        f"受访智能体画像：\n{profile_text(profile)}\n\n"
        f"{context_intro}{context_label}\n{evidence_context}\n\n"
        "待评价专业大类：\n"
        + "\n".join(major_lines)
        + "\n\n"
        "请你站在该受访者画像的视角，结合可见科技新闻，对每个专业大类的未来就业前景和薪资前景打分。\n"
        "打分要求：1-10的整数，10分最好，1分最差。\n"
        "解释要求：每个专业大类给1-2句原因，说明为什么这样评分。\n"
        "严格输出JSON对象，不要输出其他文字。格式如下：\n"
        "{\n"
        '  "ratings": [\n'
        '    {"selection_id": "major_001", "employment_score": 8, "salary_score": 9, "reason": "..." }\n'
        "  ]\n"
        "}\n"
        "必须覆盖全部 selection_id，不能新增列表外的专业。"
    )


def extract_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start:end + 1])
            return value if isinstance(value, dict) else None
        except Exception:
            return None
    return None


def clamp_score(value: Any) -> int | None:
    try:
        score = int(value)
    except Exception:
        return None
    if 1 <= score <= 10:
        return score
    return None


def parse_response(
    raw: str,
    period: str,
    profile: dict[str, Any],
    majors_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    obj = extract_json_object(raw)
    if obj is None:
        return [], "json_parse_failed"
    ratings = obj.get("ratings")
    if not isinstance(ratings, list):
        return [], "ratings_not_list"

    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ratings:
        if not isinstance(item, dict):
            continue
        selection_id = str(item.get("selection_id", "")).strip()
        major = majors_by_id.get(selection_id)
        if not major or selection_id in seen:
            continue
        employment_score = clamp_score(item.get("employment_score"))
        salary_score = clamp_score(item.get("salary_score"))
        reason = str(item.get("reason", "")).strip()
        if employment_score is None or salary_score is None or not reason:
            continue
        seen.add(selection_id)
        parsed.append(
            {
                "period": period,
                "profile_id": profile["id"],
                "source_id": profile.get("source_id", ""),
                "province": profile.get("province", ""),
                "age": profile.get("age", ""),
                "gender": profile.get("gender", ""),
                "education": profile.get("education", ""),
                "consumption": profile.get("consumption", ""),
                "selection_id": selection_id,
                "door": major["door"],
                "major_category": major["major_category"],
                "employment_score": employment_score,
                "salary_score": salary_score,
                "reason": reason,
            }
        )
    missing = sorted(set(majors_by_id) - seen)
    if missing:
        return parsed, f"missing_or_invalid_ratings:{','.join(missing[:5])}"
    return parsed, None


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


def load_completed(parsed_path: Path) -> set[tuple[str, int]]:
    if not parsed_path.exists():
        return set()
    completed: set[tuple[str, int]] = set()
    with parsed_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                completed.add((row["period"], int(row["profile_id"])))
            except Exception:
                continue
    return completed


def make_router(config_path: Path) -> Router:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    return Router(
        model_list=cfg["chat_models"],
        num_retries=3,
        timeout=180,
        retry_after=2,
        routing_strategy="least-busy",
    )


async def call_llm(router: Router, prompt: str, temperature: float, max_retries: int) -> str:
    attempt = 0
    while True:
        try:
            response = await router.acompletion(
                model="chat",
                temperature=temperature,
                messages=[
                    {"role": "system", "content": "你是一个严谨的模拟调查受访者。只输出可解析JSON。"},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content
            return (content or "").strip()
        except Exception:
            attempt += 1
            if attempt > max_retries:
                raise
            await asyncio.sleep(min(2 ** attempt, 30) + random.random())


async def run_survey(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_majors = parse_major_categories(Path(args.major_categories))
    selected_majors = select_majors_by_door(all_majors, args.major_ratio, args.major_seed)
    majors_by_id = {row["selection_id"]: row for row in selected_majors}
    major_selection_path = out_dir / "selected_major_categories.csv"
    with major_selection_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAJOR_FIELDS)
        writer.writeheader()
        writer.writerows(selected_majors)

    profiles = load_profiles(Path(args.profiles), args.limit_profiles)
    survey_periods = build_half_year_sequence(args.start_period, args.end_period)
    all_periods = build_half_year_sequence(args.news_start_period, args.end_period)
    run_news_by_period = load_news_by_period(Path(args.news_dir))
    seed_news_by_period = load_seed_news_csv(Path(args.seed_news_csv)) if args.seed_news_csv else {}
    news_by_period = merge_historical_seed_news(run_news_by_period, seed_news_by_period, args.history_end)
    news_contexts = build_news_contexts(
        news_by_period,
        all_periods,
        survey_periods,
        args.previous_periods,
        args.news_items_per_period,
        args.news_seed,
    )
    (out_dir / "news_contexts.json").write_text(json.dumps(news_contexts, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "profiles": args.profiles,
        "profile_count": len(profiles),
        "major_categories": args.major_categories,
        "major_ratio": args.major_ratio,
        "major_seed": args.major_seed,
        "selected_major_count": len(selected_majors),
        "news_dir": args.news_dir,
        "seed_news_csv": args.seed_news_csv,
        "history_end": args.history_end,
        "news_seed": args.news_seed,
        "news_items_per_period": args.news_items_per_period,
        "previous_periods": args.previous_periods,
        "score_history_periods": args.score_history_periods,
        "score_history_enabled": args.score_history_periods > 0,
        "score_history_note": (
            "When enabled, each period is run after earlier periods complete, and each prompt groups news with "
            "available previous-period major mean scores for matching half-years."
        ),
        "survey_periods": survey_periods,
        "calls_expected": len(profiles) * len(survey_periods),
        "rating_rows_expected": len(profiles) * len(survey_periods) * len(selected_majors),
        "llm_config": args.llm_config,
        "raw_output": "raw_outputs.jsonl",
        "parsed_output": "parsed_ratings.csv",
        "failures": "failures.csv",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.prepare_only:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return

    router = make_router(Path(args.llm_config))
    raw_path = out_dir / "raw_outputs.jsonl"
    parsed_path = out_dir / "parsed_ratings.csv"
    failures_path = out_dir / "failures.csv"
    score_contexts_path = out_dir / "score_contexts.json"
    prompt_samples_path = out_dir / "prompt_samples.jsonl"
    completed = load_completed(parsed_path) if args.resume else set()
    sem = asyncio.Semaphore(args.concurrency)

    async def worker(
        period: str,
        profile: dict[str, Any],
        task_index: int,
        task_total: int,
        evidence_context: str,
    ) -> None:
        prompt = make_prompt(period, profile, selected_majors, evidence_context, args.score_history_periods)
        async with sem:
            t0 = time.time()
            try:
                raw = await call_llm(router, prompt, args.temperature, args.max_retries)
                elapsed = round(time.time() - t0, 3)
                append_jsonl(raw_path, {
                    "period": period,
                    "profile_id": profile["id"],
                    "source_id": profile.get("source_id", ""),
                    "raw": raw,
                    "elapsed_seconds": elapsed,
                })
                parsed_rows, error = parse_response(raw, period, profile, majors_by_id)
                if parsed_rows:
                    append_csv(parsed_path, parsed_rows, PARSED_FIELDS)
                if error:
                    append_csv(failures_path, [{
                        "period": period,
                        "profile_id": profile["id"],
                        "attempt": "final",
                        "error": error,
                        "raw_excerpt": raw[:1000],
                    }], FAILURE_FIELDS)
            except Exception as exc:
                append_csv(failures_path, [{
                    "period": period,
                    "profile_id": profile["id"],
                    "attempt": "exception",
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_excerpt": "",
                    }], FAILURE_FIELDS)
        if task_index % args.progress_every == 0:
            print(f"progress {period} {task_index}/{task_total}", flush=True)

    async def run_period(period: str, period_tasks: list[tuple[str, dict[str, Any]]], evidence_context: str) -> None:
        if not period_tasks:
            print(f"progress {period} 0/0", flush=True)
            return
        sample_profile = period_tasks[0][1]
        append_jsonl(prompt_samples_path, {
            "period": period,
            "profile_id": sample_profile["id"],
            "source_id": sample_profile.get("source_id", ""),
            "score_history_periods": args.score_history_periods,
            "prompt": make_prompt(period, sample_profile, selected_majors, evidence_context, args.score_history_periods),
        })
        await asyncio.gather(
            *(
                worker(period, profile, idx, len(period_tasks), evidence_context)
                for idx, (_period, profile) in enumerate(period_tasks, start=1)
            )
        )

    if args.score_history_periods > 0:
        score_contexts: dict[str, dict[str, list[dict[str, Any]]]] = {}
        remaining_total = 0
        for period in survey_periods:
            for profile in profiles:
                key = (period, int(profile["id"]))
                if key not in completed:
                    remaining_total += 1
        print(json.dumps({
            "output_dir": str(out_dir),
            "profiles": len(profiles),
            "periods": len(survey_periods),
            "selected_majors": len(selected_majors),
            "remaining_calls": remaining_total,
            "expected_rating_rows_remaining": remaining_total * len(selected_majors),
            "score_history_periods": args.score_history_periods,
            "execution_mode": "period_sequential_within_period_concurrent",
        }, ensure_ascii=False, indent=2), flush=True)

        for period in survey_periods:
            completed = load_completed(parsed_path) if args.resume else completed
            period_tasks: list[tuple[str, dict[str, Any]]] = []
            for profile in profiles:
                key = (period, int(profile["id"]))
                if key not in completed:
                    period_tasks.append((period, profile))
            score_context = build_score_contexts_for_period(
                parsed_path,
                period,
                survey_periods,
                selected_majors,
                args.score_history_periods,
            )
            score_contexts[period] = score_context
            score_contexts_path.write_text(json.dumps(score_contexts, ensure_ascii=False, indent=2), encoding="utf-8")
            evidence_context = compact_period_aligned_context(
                news_contexts[period],
                score_context,
                args.news_summary_chars,
            )
            await run_period(period, period_tasks, evidence_context)
    else:
        tasks: list[tuple[str, dict[str, Any]]] = []
        for period in survey_periods:
            for profile in profiles:
                key = (period, int(profile["id"]))
                if key not in completed:
                    tasks.append((period, profile))

        print(json.dumps({
            "output_dir": str(out_dir),
            "profiles": len(profiles),
            "periods": len(survey_periods),
            "selected_majors": len(selected_majors),
            "remaining_calls": len(tasks),
            "expected_rating_rows_remaining": len(tasks) * len(selected_majors),
            "score_history_periods": 0,
            "execution_mode": "all_periods_concurrent",
        }, ensure_ascii=False, indent=2), flush=True)

        evidence_contexts = {
            period: compact_news_context(news_contexts[period], args.news_summary_chars)
            for period in survey_periods
        }
        await asyncio.gather(
            *(
                worker(period, profile, idx, len(tasks), evidence_contexts[period])
                for idx, (period, profile) in enumerate(tasks, start=1)
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run profile-agent major outlook scoring survey.")
    parser.add_argument("--profiles", default="data/profiles.jsonl")
    parser.add_argument("--major-categories", default="major/major_categories.md")
    parser.add_argument("--news-dir", default="data/news_by_period")
    parser.add_argument("--seed-news-csv", default="data/seed_news.csv")
    parser.add_argument("--history-end", default="2026-S1")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--llm-config", default=os.getenv("LLM_CONFIG_PATH", "configs/llm/llm_config.json"))
    parser.add_argument("--start-period", default="2026-S1")
    parser.add_argument("--end-period", default="2030-S2")
    parser.add_argument("--news-start-period", default="2020-S1")
    parser.add_argument("--previous-periods", type=int, default=6)
    parser.add_argument(
        "--score-history-periods",
        type=int,
        default=0,
        help="If >0, include previous N survey-period major mean scores aligned with same-period news; periods run sequentially.",
    )
    parser.add_argument("--news-items-per-period", type=int, default=10)
    parser.add_argument("--news-summary-chars", type=int, default=120)
    parser.add_argument("--major-ratio", type=float, default=0.10)
    parser.add_argument("--major-seed", type=int, default=20260521)
    parser.add_argument("--news-seed", type=int, default=20260521)
    parser.add_argument("--limit-profiles", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output/major_outlook_survey/run_{stamp}"

    asyncio.run(run_survey(args))


if __name__ == "__main__":
    main()
