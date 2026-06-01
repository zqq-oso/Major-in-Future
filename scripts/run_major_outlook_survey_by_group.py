from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from run_major_outlook_survey import (
    FAILURE_FIELDS,
    MAJOR_FIELDS,
    PARSED_FIELDS,
    append_csv,
    append_jsonl,
    build_half_year_sequence,
    build_news_contexts,
    build_score_contexts_for_period,
    call_llm,
    compact_period_aligned_context,
    compact_news_context,
    load_news_by_period,
    load_profiles,
    load_seed_news_csv,
    make_router,
    merge_historical_seed_news,
    parse_major_categories,
    parse_response,
    profile_text,
    select_majors_by_door,
)


GROUP_FIELDS = ["major_group", "group_label", "door", "selection_id", "major_category", "source_row_index"]


def split_evenly(rows: list[dict[str, Any]], parts: int) -> list[list[dict[str, Any]]]:
    if parts <= 1 or len(rows) <= 1:
        return [rows]
    chunk_size = math.ceil(len(rows) / parts)
    return [rows[index:index + chunk_size] for index in range(0, len(rows), chunk_size)]


def build_major_groups(
    selected_majors: list[dict[str, Any]],
    split_door: str,
    split_count: int,
) -> list[dict[str, Any]]:
    by_door: dict[str, list[dict[str, Any]]] = {}
    door_order: list[str] = []
    for row in selected_majors:
        door = row["door"]
        if door not in by_door:
            door_order.append(door)
            by_door[door] = []
        by_door[door].append(row)

    groups: list[dict[str, Any]] = []
    for door in door_order:
        rows = by_door[door]
        chunks = split_evenly(rows, split_count) if door == split_door else [rows]
        for chunk_index, chunk in enumerate(chunks, start=1):
            if not chunk:
                continue
            group_id = f"group_{len(groups) + 1:03d}"
            label = f"{door}_{chunk_index}" if len(chunks) > 1 else door
            groups.append(
                {
                    "major_group": group_id,
                    "group_label": label,
                    "door": door,
                    "majors": chunk,
                }
            )
    return groups


def write_major_groups(path: Path, groups: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUP_FIELDS)
        writer.writeheader()
        for group in groups:
            for major in group["majors"]:
                writer.writerow(
                    {
                        "major_group": group["major_group"],
                        "group_label": group["group_label"],
                        "door": group["door"],
                        "selection_id": major["selection_id"],
                        "major_category": major["major_category"],
                        "source_row_index": major["source_row_index"],
                    }
                )


def load_completed_group_keys(
    parsed_path: Path,
    groups: list[dict[str, Any]],
) -> set[tuple[str, int, str]]:
    if not parsed_path.exists():
        return set()

    group_by_selection_id: dict[str, str] = {}
    expected_by_group: dict[str, int] = {}
    for group in groups:
        group_id = group["major_group"]
        expected_by_group[group_id] = len(group["majors"])
        for major in group["majors"]:
            group_by_selection_id[major["selection_id"]] = group_id

    seen: dict[tuple[str, int, str], set[str]] = {}
    with parsed_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            group_id = group_by_selection_id.get(row.get("selection_id", ""))
            if not group_id:
                continue
            try:
                key = (row["period"], int(row["profile_id"]), group_id)
            except Exception:
                continue
            seen.setdefault(key, set()).add(row["selection_id"])

    completed: set[tuple[str, int, str]] = set()
    for key, selection_ids in seen.items():
        if len(selection_ids) >= expected_by_group[key[2]]:
            completed.add(key)
    return completed


def make_group_prompt(
    period: str,
    profile: dict[str, Any],
    group: dict[str, Any],
    evidence_context: str,
    score_history_periods: int = 0,
) -> str:
    majors = group["majors"]
    major_lines = [
        f"- {row['selection_id']} | 门类：{row['door']} | 专业大类：{row['major_category']}"
        for row in majors
    ]
    if score_history_periods > 0:
        context_intro = (
            f"可见信息按半年期分组。每组包含该期科技新闻；如果该期已经完成过调查，还包含当前专业组内各专业的就业/薪资统计均分。\n"
            f"专业均分最多来自当前期之前{score_history_periods}个半年期，不包含当前期结果；请把同一时期的均分和新闻对应起来理解。\n"
        )
        context_label = "可见科技新闻与历史专业均分："
    else:
        context_intro = "可见科技新闻概况（当前期及此前若干半年期，每期随机抽取若干条）：\n"
        context_label = ""
    return (
        f"调查时期：{period}\n"
        f"当前专业组：{group['group_label']}（{len(majors)}个专业）\n\n"
        f"受访智能体画像：\n{profile_text(profile)}\n\n"
        f"{context_intro}{context_label}\n{evidence_context}\n\n"
        "待评价专业大类：\n"
        + "\n".join(major_lines)
        + "\n\n"
        "请你站在该受访者画像的视角，结合可见科技新闻，对每个专业大类的未来就业前景和薪资前景打分。\n"
        "评分必须使用跨全部专业统一的绝对标尺，不要只在当前专业组内部相对排名。\n"
        "评分锚点：9-10代表全国性高需求且薪资强；7-8代表需求明确、薪资较好；5-6代表需求稳定但增长有限；"
        "3-4代表就业面窄或受产业/自动化压力影响；1-2代表长期低需求、低薪资或严重受替代冲击。\n"
        "打分要求：1-10的整数，10分最好，1分最差。\n"
        "解释要求：每个专业大类给1-2句原因，说明为什么这样评分。\n"
        "严格输出JSON对象，不要输出其他文字。格式如下：\n"
        "{\n"
        '  "ratings": [\n'
        '    {"selection_id": "major_001", "employment_score": 8, "salary_score": 9, "reason": "..." }\n'
        "  ]\n"
        "}\n"
        "必须覆盖本专业组内全部 selection_id，不能新增列表外的专业。"
    )


async def run_grouped_survey(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_majors = parse_major_categories(Path(args.major_categories))
    selected_majors = select_majors_by_door(all_majors, args.major_ratio, args.major_seed)
    groups = build_major_groups(selected_majors, args.split_door, args.split_count)

    with (out_dir / "selected_major_categories.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAJOR_FIELDS)
        writer.writeheader()
        writer.writerows(selected_majors)
    write_major_groups(out_dir / "major_groups.csv", groups)

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

    group_summaries = [
        {
            "major_group": group["major_group"],
            "group_label": group["group_label"],
            "door": group["door"],
            "major_count": len(group["majors"]),
            "selection_ids": [major["selection_id"] for major in group["majors"]],
        }
        for group in groups
    ]
    metadata = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "profiles": args.profiles,
        "profile_count": len(profiles),
        "major_categories": args.major_categories,
        "major_ratio": args.major_ratio,
        "major_seed": args.major_seed,
        "selected_major_count": len(selected_majors),
        "major_grouping": "by_door",
        "split_door": args.split_door,
        "split_count": args.split_count,
        "major_group_count": len(groups),
        "major_groups": group_summaries,
        "news_dir": args.news_dir,
        "seed_news_csv": args.seed_news_csv,
        "history_end": args.history_end,
        "news_seed": args.news_seed,
        "news_items_per_period": args.news_items_per_period,
        "previous_periods": args.previous_periods,
        "score_history_periods": args.score_history_periods,
        "score_history_enabled": args.score_history_periods > 0,
        "score_history_note": (
            "Grouped mode runs each period sequentially. Within a period, profile-group tasks are concurrent. "
            "Each prompt only includes the current major group's previous-period score summaries."
        ),
        "survey_periods": survey_periods,
        "calls_expected": len(profiles) * len(survey_periods) * len(groups),
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
    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async def worker(
        period: str,
        profile: dict[str, Any],
        group: dict[str, Any],
        task_index: int,
        task_total: int,
        evidence_context: str,
    ) -> None:
        group_id = group["major_group"]
        majors = group["majors"]
        majors_by_id = {major["selection_id"]: major for major in majors}
        prompt = make_group_prompt(period, profile, group, evidence_context, args.score_history_periods)
        async with sem:
            t0 = time.time()
            try:
                final_raw = ""
                final_elapsed = 0.0
                final_rows: list[dict[str, Any]] = []
                final_error: str | None = None
                success = False
                for parse_attempt in range(args.parse_retries + 1):
                    raw = await call_llm(router, prompt, args.temperature, args.max_retries)
                    elapsed = round(time.time() - t0, 3)
                    parsed_rows, error = parse_response(raw, period, profile, majors_by_id)
                    final_raw = raw
                    final_elapsed = elapsed
                    final_rows = parsed_rows
                    final_error = error
                    success = error is None and len(parsed_rows) == len(majors)
                    async with write_lock:
                        append_jsonl(
                            raw_path,
                            {
                                "period": period,
                                "major_group": group_id,
                                "group_label": group["group_label"],
                                "profile_id": profile["id"],
                                "source_id": profile.get("source_id", ""),
                                "retry_attempt": parse_attempt,
                                "parse_error": None if success else error or "incomplete_group",
                                "parsed_rows": len(parsed_rows),
                                "expected_rows": len(majors),
                                "raw": raw,
                                "elapsed_seconds": elapsed,
                            },
                        )
                    if success:
                        break

                async with write_lock:
                    if success:
                        append_csv(parsed_path, parsed_rows, PARSED_FIELDS)
                    else:
                        append_csv(
                            failures_path,
                            [
                                {
                                    "period": period,
                                    "profile_id": profile["id"],
                                    "attempt": "final",
                                    "error": (
                                        f"{group_id}:{final_error or 'incomplete_group'}:"
                                        f"{len(final_rows)}/{len(majors)}"
                                    ),
                                    "raw_excerpt": final_raw[:1000],
                                }
                            ],
                            FAILURE_FIELDS,
                        )
            except Exception as exc:
                async with write_lock:
                    append_csv(
                        failures_path,
                        [
                            {
                                "period": period,
                                "profile_id": profile["id"],
                                "attempt": "exception",
                                "error": f"{group_id}:{type(exc).__name__}: {exc}",
                                "raw_excerpt": "",
                            }
                        ],
                        FAILURE_FIELDS,
                    )
        if task_index % args.progress_every == 0:
            print(f"progress {period} {task_index}/{task_total}", flush=True)

    completed = load_completed_group_keys(parsed_path, groups) if args.resume else set()
    remaining_total = 0
    remaining_rating_rows = 0
    for period in survey_periods:
        for group in groups:
            for profile in profiles:
                key = (period, int(profile["id"]), group["major_group"])
                if key not in completed:
                    remaining_total += 1
                    remaining_rating_rows += len(group["majors"])
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "profiles": len(profiles),
                "periods": len(survey_periods),
                "selected_majors": len(selected_majors),
                "major_groups": len(groups),
                "remaining_calls": remaining_total,
                "expected_rating_rows_remaining": remaining_rating_rows,
                "rating_rows_expected_total": len(profiles) * len(survey_periods) * len(selected_majors),
                "score_history_periods": args.score_history_periods,
                "execution_mode": "period_sequential_grouped_by_door_within_period_concurrent",
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    score_contexts: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for period in survey_periods:
        completed = load_completed_group_keys(parsed_path, groups) if args.resume else completed
        score_contexts[period] = {}
        evidence_by_group: dict[str, str] = {}
        for group in groups:
            score_context = build_score_contexts_for_period(
                parsed_path,
                period,
                survey_periods,
                group["majors"],
                args.score_history_periods,
            )
            score_contexts[period][group["major_group"]] = score_context
            if args.score_history_periods > 0:
                evidence_by_group[group["major_group"]] = compact_period_aligned_context(
                    news_contexts[period],
                    score_context,
                    args.news_summary_chars,
                )
            else:
                evidence_by_group[group["major_group"]] = compact_news_context(
                    news_contexts[period],
                    args.news_summary_chars,
                )
        score_contexts_path.write_text(json.dumps(score_contexts, ensure_ascii=False, indent=2), encoding="utf-8")

        def incomplete_period_tasks() -> list[tuple[dict[str, Any], dict[str, Any]]]:
            current_completed = load_completed_group_keys(parsed_path, groups) if args.resume else completed
            missing: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for group_item in groups:
                for profile_item in profiles:
                    key = (period, int(profile_item["id"]), group_item["major_group"])
                    if key not in current_completed:
                        missing.append((group_item, profile_item))
            return missing

        period_tasks = incomplete_period_tasks()

        if not period_tasks:
            print(f"progress {period} 0/0", flush=True)
            continue

        sampled_groups: set[str] = set()
        for group, profile in period_tasks:
            if group["major_group"] in sampled_groups:
                continue
            sampled_groups.add(group["major_group"])
            append_jsonl(
                prompt_samples_path,
                {
                    "period": period,
                    "major_group": group["major_group"],
                    "group_label": group["group_label"],
                    "profile_id": profile["id"],
                    "source_id": profile.get("source_id", ""),
                    "score_history_periods": args.score_history_periods,
                    "prompt": make_group_prompt(
                        period,
                        profile,
                        group,
                        evidence_by_group[group["major_group"]],
                        args.score_history_periods,
                    ),
                },
            )

        async def run_task_batch(batch: list[tuple[dict[str, Any], dict[str, Any]]], label: str) -> None:
            print(f"progress {period} {label} 0/{len(batch)}", flush=True)
            await asyncio.gather(
                *(
                    worker(period, profile, group, idx, len(batch), evidence_by_group[group["major_group"]])
                    for idx, (group, profile) in enumerate(batch, start=1)
                )
            )

        await run_task_batch(period_tasks, "main")

        for repair_round in range(1, args.period_repair_rounds + 1):
            remaining = incomplete_period_tasks()
            if not remaining:
                print(f"period_complete {period}", flush=True)
                break
            print(f"period_repair {period} round={repair_round} missing_calls={len(remaining)}", flush=True)
            await run_task_batch(remaining, f"repair{repair_round}")
        else:
            remaining = incomplete_period_tasks()
            if remaining:
                raise RuntimeError(
                    f"period {period} still has {len(remaining)} incomplete profile-group calls after "
                    f"{args.period_repair_rounds} repair rounds"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grouped profile-agent major outlook scoring survey.")
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
    parser.add_argument("--score-history-periods", type=int, default=6)
    parser.add_argument("--news-items-per-period", type=int, default=10)
    parser.add_argument("--news-summary-chars", type=int, default=120)
    parser.add_argument("--major-ratio", type=float, default=1.0)
    parser.add_argument("--major-seed", type=int, default=20260521)
    parser.add_argument("--news-seed", type=int, default=20260521)
    parser.add_argument("--split-door", default="工学")
    parser.add_argument("--split-count", type=int, default=2)
    parser.add_argument("--limit-profiles", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--parse-retries", type=int, default=2)
    parser.add_argument("--period-repair-rounds", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output/major_outlook_survey/grouped_by_door_{stamp}"

    asyncio.run(run_grouped_survey(args))


if __name__ == "__main__":
    main()
