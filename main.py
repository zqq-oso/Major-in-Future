from __future__ import annotations
import asyncio
import dotenv
import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import random
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TARGET_RECORDS = 1_000_000
DEFAULT_SAMPLES_PER_REQUEST = 1
DEFAULT_REFERENCE_SAMPLE_COUNT = 3
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_DB_PATH = Path(os.getenv("DB_PATH", "data/source_profiles.db"))
DEFAULT_PLAN_PATH = Path("data/province_plan.json")
DEFAULT_MODEL_FILE = Path("模型.md")
DEFAULT_ECO_ROOT = Path(
    "/home/batchcom/.cache/huggingface/hub/datasets--Lishi0905--SocioVerse/"
    "snapshots/4d437a4ce612266bc969c6e4ae7d8f5a080da44d/sample_pool_eco"
)
DEFAULT_REFERENCE_CHAR_LIMIT = 600
DEFAULT_MAX_INFLIGHT_REQUESTS = 3200
DEFAULT_RETRY_LOG_INTERVAL = 5
DEFAULT_MODEL_FAILURE_THRESHOLD = 3
DEFAULT_MODEL_COOLDOWN_SECONDS = 60.0

SYSTEM_PROMPT = (
    "你是一名中文社交媒体写作者。"
    "请根据用户画像生成自然、真实、像真人发帖的小红书正文。"
    "不要解释，不要加标题，不要加编号，不要输出任何额外说明。"
)

USER_PROMPT_TEMPLATE = """你是来自{province}的用户。
你的画像是：{demographic_text}。
下面是同省份其他用户的 3 条真实发言，请参考他们的表达习惯、生活语境和细节密度，但不要照抄：
示例1：{sample_1}
示例2：{sample_2}
示例3：{sample_3}
请参考这些发言，写你自己的小红书稿件，主题不限。
要求：
1. 使用中文。
2. 只输出正文。
3. 长度控制在 80 到 400 字。
4. 口吻自然，允许口语化和生活细节。
5. 不要逐句改写示例，也不要直接复用示例里的句子。
"""

PROVINCE_POPULATIONS: list[tuple[str, int]] = [
    ("广东省", 12859),
    ("山东省", 10043),
    ("河南省", 9785),
    ("江苏省", 8526),
    ("四川省", 8364),
    ("河北省", 7378),
    ("浙江省", 6701),
    ("湖南省", 6539),
    ("安徽省", 6123),
    ("湖北省", 5834),
    ("广西壮族自治区", 4989),
    ("云南省", 4655),
    ("江西省", 4502),
    ("福建省", 4193),
    ("辽宁省", 4155),
    ("陕西省", 3953),
    ("贵州省", 3857),
    ("山西省", 3446),
    ("重庆市", 3190),
    ("黑龙江省", 3029),
    ("新疆维吾尔自治区", 2623),
    ("上海市", 2480),
    ("甘肃省", 2443),
    ("内蒙古自治区", 2388),
    ("北京市", 2183),
    ("吉林省", 2317),
    ("天津市", 1364),
    ("海南省", 1055),
    ("宁夏回族自治区", 732),
    ("青海省", 592),
    ("西藏自治区", 370),
]

ECO_FILE_TO_PROVINCE = {
    "anhui": "安徽省",
    "beijing": "北京市",
    "chongqing": "重庆市",
    "fujian": "福建省",
    "gansu": "甘肃省",
    "guangdong": "广东省",
    "guangxi": "广西壮族自治区",
    "guizhou": "贵州省",
    "hainan": "海南省",
    "hebei": "河北省",
    "heilongjiang": "黑龙江省",
    "henan": "河南省",
    "hubei": "湖北省",
    "hunan": "湖南省",
    "jiangsu": "江苏省",
    "jiangxi": "江西省",
    "jilin": "吉林省",
    "liaoning": "辽宁省",
    "neimenggu": "内蒙古自治区",
    "ningxia": "宁夏回族自治区",
    "qinghai": "青海省",
    "shaanxi": "陕西省",
    "shandong": "山东省",
    "shanghai": "上海市",
    "shanxi": "山西省",
    "sichuan": "四川省",
    "tianjin": "天津市",
    "xinjiang": "新疆维吾尔自治区",
    "xizang": "西藏自治区",
    "yunnan": "云南省",
    "zhejiang": "浙江省",
}
PROVINCE_TO_ECO_FILE = {province: file_stem for file_stem, province in ECO_FILE_TO_PROVINCE.items()}

AGE_OPTIONS = [("a", "0-18 岁"), ("b", "19-35 岁"), ("c", "36 岁及以上")]
EDUCATION_OPTIONS = [("a", "高中及以下"), ("b", "本科及以上")]
GENDER_OPTIONS = [("a", "男"), ("b", "女")]
CONSUMPTION_OPTIONS = [("a", "低"), ("b", "中等"), ("c", "高")]


@dataclass(frozen=True)
class DemographicCombo:
    age_code: str
    age_text: str
    education_code: str
    education_text: str
    gender_code: str
    gender_text: str
    consumption_code: str
    consumption_text: str

    def label(self) -> dict[str, str]:
        return {
            "AGE": self.age_code,
            "Education": self.education_code,
            "GENDER": self.gender_code,
            "Level of Consumption": self.consumption_code,
        }

    def prompt_text(self) -> str:
        return (
            f"年龄：{self.age_text} "
            f"学历：{self.education_text} "
            f"性别：{self.gender_text} "
            f"消费水平：{self.consumption_text}"
        )


@dataclass(frozen=True)
class ProvincePlan:
    order: int
    province: str
    population: int
    raw_target: float
    floor_target: int
    assigned_records: int
    fractional_remainder: float
    generation_calls: int
    discarded_samples: int


class EmptyCompletionError(RuntimeError):
    pass


def build_demographic_pool() -> list[DemographicCombo]:
    combos: list[DemographicCombo] = []
    for age_code, age_text in AGE_OPTIONS:
        for education_code, education_text in EDUCATION_OPTIONS:
            for gender_code, gender_text in GENDER_OPTIONS:
                for consumption_code, consumption_text in CONSUMPTION_OPTIONS:
                    combos.append(
                        DemographicCombo(
                            age_code=age_code,
                            age_text=age_text,
                            education_code=education_code,
                            education_text=education_text,
                            gender_code=gender_code,
                            gender_text=gender_text,
                            consumption_code=consumption_code,
                            consumption_text=consumption_text,
                        )
                    )
    return combos


def compute_province_plan(
    target_records: int,
    samples_per_request: int,
) -> tuple[list[ProvincePlan], dict[str, Any]]:
    total_population = sum(population for _, population in PROVINCE_POPULATIONS)
    raw_targets = [
        target_records * population / total_population for _, population in PROVINCE_POPULATIONS
    ]
    floor_targets = [math.floor(value) for value in raw_targets]
    floor_total = sum(floor_targets)
    gap_after_floor = target_records - floor_total

    remainders = sorted(
        [
            (raw_targets[index] - floor_targets[index], index)
            for index in range(len(PROVINCE_POPULATIONS))
        ],
        key=lambda item: (-item[0], item[1]),
    )
    assigned = floor_targets[:]
    for _, index in remainders[:gap_after_floor]:
        assigned[index] += 1

    province_plans: list[ProvincePlan] = []
    total_generation_calls = 0
    total_discarded_samples = 0
    for order, ((province, population), raw_target, floor_target, assigned_records) in enumerate(
        zip(PROVINCE_POPULATIONS, raw_targets, floor_targets, assigned),
        start=1,
    ):
        generation_calls = math.ceil(assigned_records / samples_per_request)
        discarded_samples = generation_calls * samples_per_request - assigned_records
        total_generation_calls += generation_calls
        total_discarded_samples += discarded_samples
        province_plans.append(
            ProvincePlan(
                order=order,
                province=province,
                population=population,
                raw_target=raw_target,
                floor_target=floor_target,
                assigned_records=assigned_records,
                fractional_remainder=raw_target - floor_target,
                generation_calls=generation_calls,
                discarded_samples=discarded_samples,
            )
        )

    summary = {
        "target_records": target_records,
        "samples_per_request": samples_per_request,
        "gap_after_floor": gap_after_floor,
        "floor_total": floor_total,
        "final_total": sum(plan.assigned_records for plan in province_plans),
        "total_generation_calls": total_generation_calls,
        "discarded_samples_after_trim": total_discarded_samples,
        "demographic_combo_count": len(build_demographic_pool()),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": USER_PROMPT_TEMPLATE,
    }
    return province_plans, summary


def save_plan(plan_path: Path, province_plans: list[ProvincePlan], summary: dict[str, Any]) -> None:
    payload = {
        "summary": summary,
        "provinces": [
            {
                "order": plan.order,
                "province": plan.province,
                "population_10k": plan.population,
                "raw_target": round(plan.raw_target, 6),
                "floor_target": plan.floor_target,
                "assigned_records": plan.assigned_records,
                "fractional_remainder": round(plan.fractional_remainder, 6),
                "generation_calls": plan.generation_calls,
                "discarded_samples": plan.discarded_samples,
            }
            for plan in province_plans
        ],
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_setting(
    cli_value: str | None,
    env_values: dict[str, str],
    *keys: str,
    default: str | None = None,
) -> str | None:
    if cli_value:
        return cli_value
    for key in keys:
        value = env_values.get(key)
        if value:
            return value
    return default


def mask_secret(value: str | None) -> str:
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def load_model_pool_file(model_file: Path) -> list[str]:
    if not model_file.exists():
        raise SystemExit(f"model file not found: {model_file}")
    models = [
        line.strip()
        for line in model_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not models:
        raise SystemExit(f"model file is empty: {model_file}")
    return models


def resolve_model_pool(
    cli_value: str | None,
    env_values: dict[str, str],
    model_file: Path | None = None,
) -> list[str]:
    raw_value = resolve_setting(
        cli_value,
        env_values,
        "models",
        "model_pool",
        "MODEL_POOL",
        "OPENAI_MODELS",
        "model",
        "OPENAI_MODEL",
    )
    if not raw_value:
        if model_file is None:
            raise SystemExit("missing model pool: use --models model_a,model_b")
        return load_model_pool_file(model_file)
    models = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not models:
        raise SystemExit("model pool is empty after parsing --models")
    return models


def connect_sqlite(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA cache_size=-200000;")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            province TEXT NOT NULL,
            content TEXT NOT NULL,
            demographic_label TEXT NOT NULL,
            source TEXT NOT NULL,
            model_name TEXT,
            province_record_index INTEGER,
            prompt_index INTEGER,
            created_at TEXT NOT NULL
        );
        """
    )
    existing_columns = {
        row[1]: row for row in conn.execute("PRAGMA table_info(posts)")
    }
    column_migrations = {
        "source": "TEXT NOT NULL DEFAULT 'generated'",
        "model_name": "TEXT",
        "province_record_index": "INTEGER",
        "prompt_index": "INTEGER",
        "created_at": "TEXT NOT NULL DEFAULT ''",
    }
    for column_name, column_sql in column_migrations.items():
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE posts ADD COLUMN {column_name} {column_sql};")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_source ON posts(source);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_province ON posts(province);")
    conn.commit()


def insert_records(conn: sqlite3.Connection, records: list[tuple[Any, ...]]) -> int:
    if not records:
        return 0
    cursor = conn.executemany(
        """
        INSERT OR IGNORE INTO posts (
            id,
            province,
            content,
            demographic_label,
            source,
            model_name,
            province_record_index,
            prompt_index,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        records,
    )
    conn.commit()
    return cursor.rowcount if cursor.rowcount != -1 else 0


def clean_generated_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip().strip('"').strip("'")


def extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content).strip()


def normalize_reference_text(text: str, char_limit: int = DEFAULT_REFERENCE_CHAR_LIMIT) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= char_limit:
        return normalized
    return normalized[:char_limit].rstrip() + "..."


def build_user_prompt(province: str, combo: DemographicCombo, reference_samples: list[str]) -> str:
    if len(reference_samples) != DEFAULT_REFERENCE_SAMPLE_COUNT:
        raise ValueError(
            f"expected {DEFAULT_REFERENCE_SAMPLE_COUNT} reference samples, got {len(reference_samples)}"
        )
    return USER_PROMPT_TEMPLATE.format(
        province=province,
        demographic_text=combo.prompt_text(),
        sample_1=normalize_reference_text(reference_samples[0]),
        sample_2=normalize_reference_text(reference_samples[1]),
        sample_3=normalize_reference_text(reference_samples[2]),
    ).strip()


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "未知"
    secs = int(seconds)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def eco_record_to_row(province: str, province_key: str, payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        f"eco:{province_key}:{payload['id']}",
        province,
        str(payload.get("clean_extracted_texts", "")).strip(),
        json.dumps(payload.get("demographic_label", {}), ensure_ascii=False, sort_keys=True),
        "eco",
        None,
        None,
        None,
        current_timestamp(),
    )


def load_province_sample_texts(eco_root: Path, province: str) -> list[str]:
    province_key = PROVINCE_TO_ECO_FILE.get(province)
    if not province_key:
        raise SystemExit(f"missing eco file mapping for province: {province}")
    eco_path = eco_root / f"{province_key}.json"
    if not eco_path.exists():
        raise SystemExit(f"province eco file not found: {eco_path}")

    payloads = json.loads(eco_path.read_text(encoding="utf-8"))
    sample_texts: list[str] = []
    for payload in payloads:
        text = str(payload.get("clean_extracted_texts", "")).strip()
        if text:
            sample_texts.append(text)
    if len(sample_texts) < DEFAULT_REFERENCE_SAMPLE_COUNT:
        raise SystemExit(
            f"province sample pool too small for {province}: got {len(sample_texts)} texts"
        )
    return sample_texts


def ingest_eco_dataset(db_path: Path, eco_root: Path, batch_size: int) -> None:
    conn = connect_sqlite(db_path)
    ensure_schema(conn)

    inserted_total = 0
    batch: list[tuple[Any, ...]] = []
    files = sorted(eco_root.glob("*.json"))
    if not files:
        raise SystemExit(f"eco dataset not found under {eco_root}")

    for path in files:
        province = ECO_FILE_TO_PROVINCE.get(path.stem)
        if not province:
            raise SystemExit(f"missing province mapping for {path.stem}")
        payloads = json.loads(path.read_text(encoding="utf-8"))
        for payload in payloads:
            batch.append(eco_record_to_row(province, path.stem, payload))
            if len(batch) >= batch_size:
                inserted_total += insert_records(conn, batch)
                batch.clear()

    if batch:
        inserted_total += insert_records(conn, batch)

    conn.close()
    print(f"eco import finished: inserted_or_ignored={inserted_total}", flush=True)


def fetch_existing_generated_counts(db_path: Path) -> dict[str, int]:
    if not db_path.exists():
        return {}
    conn = connect_sqlite(db_path)
    ensure_schema(conn)
    rows = conn.execute(
        """
        SELECT province, COUNT(*)
        FROM posts
        WHERE source = 'generated'
        GROUP BY province;
        """
    ).fetchall()
    conn.close()
    return {province: count for province, count in rows}


def writer_process_main(db_path: Path, record_queue: mp.Queue, flush_batch_size: int, target_total: int = 0) -> None:
    conn = connect_sqlite(db_path)
    ensure_schema(conn)

    pending: list[tuple[Any, ...]] = []
    committed = 0
    start_time = time.time()
    last_progress_time = start_time
    PROGRESS_INTERVAL = 60.0

    def maybe_print_progress() -> None:
        nonlocal last_progress_time
        now = time.time()
        if target_total <= 0 or now - last_progress_time < PROGRESS_INTERVAL:
            return
        elapsed = now - start_time
        rate_per_min = committed / elapsed * 60 if elapsed > 0 else 0
        eta = (target_total - committed) / (committed / elapsed) if committed > 0 else float("inf")
        pct = 100.0 * committed / target_total
        print(
            f"[进度] {committed}/{target_total} ({pct:.1f}%) | "
            f"速度 {rate_per_min:.0f}条/min | 预计剩余 {format_eta(eta)}",
            flush=True,
        )
        last_progress_time = now

    while True:
        try:
            item = record_queue.get(timeout=1.0)
        except queue.Empty:
            if pending:
                committed += insert_records(conn, pending)
                pending.clear()
                print(f"writer committed={committed}", flush=True)
            maybe_print_progress()
            continue

        if item is None:
            break

        pending.extend(item)
        if len(pending) >= flush_batch_size:
            committed += insert_records(conn, pending)
            pending.clear()
            print(f"writer committed={committed}", flush=True)
            maybe_print_progress()

    if pending:
        committed += insert_records(conn, pending)

    conn.close()
    if target_total > 0:
        elapsed = time.time() - start_time
        rate_per_min = committed / elapsed * 60 if elapsed > 0 else 0
        print(
            f"[完成] {committed}/{target_total} ({100.0 * committed / target_total:.1f}%) | "
            f"平均速度 {rate_per_min:.0f}条/min",
            flush=True,
        )
    print(f"writer finished committed={committed}", flush=True)


async def async_request_one_sample(
    client: Any,
    province: str,
    combo: DemographicCombo,
    reference_samples: list[str],
    model_name: str,
    temperature: float,
    timeout_seconds: float,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str, str] | None:
    """Fire-and-forget: try once with the given model, return None on any error."""
    user_prompt = build_user_prompt(province, combo, reference_samples)
    try:
        async with semaphore:
            response = await client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                timeout=timeout_seconds,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        choices = getattr(response, "choices", None) or []
        if not choices:
            return None
        text = clean_generated_text(extract_message_text(choices[0].message.content))
        if not text:
            return None
        return text, user_prompt, model_name
    except Exception as exc:
        print(f"[request error] {province} {model_name}: {type(exc).__name__}: {exc}", flush=True)
        return None


def generated_record_to_row(
    province: str,
    province_record_index: int,
    content: str,
    demographic_label: dict[str, str],
    model_name: str,
    prompt_index: int,
) -> tuple[Any, ...]:
    import uuid
    return (
        f"generated:{province}:{province_record_index}:{uuid.uuid4().hex[:8]}",
        province,
        content,
        json.dumps(demographic_label, ensure_ascii=False, sort_keys=True),
        "generated",
        model_name,
        province_record_index,
        prompt_index,
        current_timestamp(),
    )


async def async_province_worker(
    province: str,
    remaining_records: int,
    start_index: int,
    province_sample_texts: list[str],
    client: Any,
    model_pool: list[str],
    temperature: float,
    timeout_seconds: float,
    seed: int,
    record_queue: mp.Queue,
    semaphore: asyncio.Semaphore,
) -> None:
    """Concurrent fire-and-forget with bounded inflight tasks."""
    rng = random.Random(seed)
    combos = build_demographic_pool()
    model_idx = 0

    produced = 0
    fired = 0
    batch: list[tuple[Any, ...]] = []
    FLUSH = 100

    def _prep() -> tuple[DemographicCombo, list[str], str]:
        """Prepare request params (called from event loop, single-threaded, rng-safe)."""
        nonlocal model_idx
        combo = rng.choice(combos)
        ref = rng.sample(province_sample_texts, k=DEFAULT_REFERENCE_SAMPLE_COUNT)
        mname = model_pool[model_idx % len(model_pool)]
        model_idx += 1
        return combo, ref, mname

    async def _do_one(combo: DemographicCombo, ref: list[str], mname: str) -> tuple[str, DemographicCombo, str] | None:
        r = await async_request_one_sample(
            client=client, province=province, combo=combo,
            reference_samples=ref, model_name=mname,
            temperature=temperature, timeout_seconds=timeout_seconds,
            semaphore=semaphore,
        )
        if r is None:
            return None
        text, _, model_name = r
        return text, combo, model_name

    pending: set[asyncio.Task] = set()
    MAX_INFLIGHT = 200
    failures_without_success = 0
    FAILURE_BACKOFF_THRESHOLD = 500
    FAILURE_BACKOFF_SECONDS = 30.0

    while produced < remaining_records:
        # Fill up to MAX_INFLIGHT
        while len(pending) < MAX_INFLIGHT:
            combo, ref, mname = _prep()
            fired += 1
            t = asyncio.create_task(_do_one(combo, ref, mname))
            pending.add(t)
            t.add_done_callback(pending.discard)

        # Wait for at least one to complete
        done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

        for t in done:
            try:
                r = t.result()
            except Exception:
                failures_without_success += 1
                continue
            if r is None:
                failures_without_success += 1
                continue
            failures_without_success = 0
            text, combo, mname = r
            row = generated_record_to_row(
                province=province,
                province_record_index=start_index + produced,
                content=text,
                demographic_label=combo.label(),
                model_name=mname,
                prompt_index=fired,
            )
            batch.append(row)
            produced += 1
            if len(batch) >= FLUSH or produced == remaining_records:
                record_queue.put(batch)
                batch = []
            if produced % 100 == 0:
                print(f"{province} produced={produced}/{remaining_records} fired={fired}", flush=True)
            if produced >= remaining_records:
                break

        if failures_without_success >= FAILURE_BACKOFF_THRESHOLD:
            print(
                f"{province} WARNING: {failures_without_success} requests failed without success "
                f"(fired={fired}, produced={produced}), backing off {FAILURE_BACKOFF_SECONDS}s ...",
                flush=True,
            )
            await asyncio.sleep(FAILURE_BACKOFF_SECONDS)
            failures_without_success = 0

    # Cancel stragglers
    for t in pending:
        t.cancel()
    if batch:
        record_queue.put(batch)
    print(f"{province} done produced={produced}/{remaining_records} fired={fired}", flush=True)


def print_plan_summary(province_plans: list[ProvincePlan], summary: dict[str, Any]) -> None:
    print(
        "plan ready:",
        json.dumps(
            {
                "target_records": summary["target_records"],
                "gap_after_floor": summary["gap_after_floor"],
                "total_generation_calls": summary["total_generation_calls"],
                "discarded_samples_after_trim": summary["discarded_samples_after_trim"],
                "demographic_combo_count": summary["demographic_combo_count"],
            },
            ensure_ascii=False,
        ),
    )
    preview = [
        {
            "province": plan.province,
            "assigned_records": plan.assigned_records,
            "generation_calls": plan.generation_calls,
        }
        for plan in province_plans[:5]
    ]
    print("top5 provinces:", json.dumps(preview, ensure_ascii=False))


def _province_process_main(
    province: str,
    remaining_records: int,
    start_index: int,
    eco_root: Path,
    api_key: str,
    base_url: str,
    model_pool: list[str],
    temperature: float,
    timeout_seconds: float,
    seed: int,
    workers_per_province: int,
    max_inflight_requests: int,
    record_queue: mp.Queue,
) -> None:
    """Entry point for each province subprocess. Runs its own asyncio loop."""

    async def _run() -> None:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=0)
        semaphore = asyncio.Semaphore(max_inflight_requests)

        province_sample_texts = load_province_sample_texts(eco_root, province)

        coroutines = []
        chunk_size = math.ceil(remaining_records / workers_per_province)
        for worker_idx in range(workers_per_province):
            offset = worker_idx * chunk_size
            if offset >= remaining_records:
                break
            worker_remaining = min(chunk_size, remaining_records - offset)
            coroutines.append(
                async_province_worker(
                    province=province,
                    remaining_records=worker_remaining,
                    start_index=start_index + offset,
                    province_sample_texts=province_sample_texts,
                    client=client,
                    model_pool=model_pool,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                    seed=seed + worker_idx,
                    record_queue=record_queue,
                    semaphore=semaphore,
                )
            )

        print(f"{province} launching {len(coroutines)} async workers (pid={os.getpid()})", flush=True)
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        failed_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_count += 1
                print(f"{province} worker {i} failed: {result}", flush=True)
        if failed_count:
            print(f"{province} {failed_count} workers failed", flush=True)
        else:
            print(f"{province} all workers done", flush=True)

    asyncio.run(_run())


def run_generate(args: argparse.Namespace) -> None:
    if args.samples_per_request != DEFAULT_SAMPLES_PER_REQUEST:
        raise SystemExit("generate currently requires --samples-per-request=3")

    env_values = load_env_file(args.env_path)
    api_key = resolve_setting(args.api_key, env_values, "api_key", "OPENAI_API_KEY")
    base_url = resolve_setting(args.base_url, env_values, "base_url", "OPENAI_BASE_URL")
    model_pool = resolve_model_pool(args.models, env_values, args.model_file)
    print(
        "env check:",
        json.dumps(
            {
                "env_path": str(args.env_path.resolve()),
                "env_exists": args.env_path.exists(),
                "env_keys": sorted(env_values.keys()),
                "api_key_loaded": bool(api_key),
                "api_key_masked": mask_secret(api_key),
                "base_url_loaded": bool(base_url),
                "base_url": base_url or "<missing>",
                "model_pool_size": len(model_pool),
                "model_pool_preview": model_pool[:5],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not api_key:
        raise SystemExit("missing api key: pass --api-key or put api_key=... in .env")
    if not base_url:
        raise SystemExit("missing base url: pass --base-url or put base_url=... in .env")

    province_plans, summary = compute_province_plan(args.target_records, args.samples_per_request)
    save_plan(args.plan_output, province_plans, summary)

    conn = connect_sqlite(args.db_path)
    ensure_schema(conn)
    conn.close()

    if args.import_eco_first:
        ingest_eco_dataset(args.db_path, args.eco_root, args.db_batch_size)

    existing_counts = fetch_existing_generated_counts(args.db_path)
    tasks: list[tuple[ProvincePlan, int, int]] = []
    for plan in province_plans:
        existing = existing_counts.get(plan.province, 0)
        remaining = max(plan.assigned_records - existing, 0)
        if remaining > 0:
            tasks.append((plan, remaining, existing))

    if not tasks:
        print("all province quotas already exist in database, nothing to do", flush=True)
        return

    tasks_total = sum(remaining for _, remaining, _ in tasks)
    ctx = mp.get_context("spawn")
    record_queue: mp.Queue = ctx.Queue(maxsize=args.queue_maxsize)
    writer = ctx.Process(
        target=writer_process_main,
        args=(args.db_path, record_queue, args.db_batch_size, tasks_total),
        name="sqlite-writer",
    )
    writer.start()

    # Launch one subprocess per province, each with its own event loop
    province_processes: list[mp.Process] = []
    for idx, (plan, remaining, start_index) in enumerate(tasks):
        p = ctx.Process(
            target=_province_process_main,
            args=(
                plan.province,
                remaining,
                start_index,
                args.eco_root,
                api_key,
                base_url,
                model_pool,
                args.temperature,
                args.timeout_seconds,
                args.seed + idx * args.workers_per_province,
                args.workers_per_province,
                args.max_inflight_requests,
                record_queue,
            ),
            name=f"province-{plan.province}",
        )
        province_processes.append(p)

    print(f"launching {len(province_processes)} province processes, "
          f"each with up to {args.workers_per_province} async workers", flush=True)

    for p in province_processes:
        p.start()

    try:
        for p in province_processes:
            p.join()
        failed = [p for p in province_processes if p.exitcode != 0]
        if failed:
            print(f"{len(failed)} province processes failed: "
                  f"{[p.name for p in failed]}", flush=True)
    finally:
        record_queue.put(None)
        writer.join()
        if writer.exitcode != 0:
            print(f"writer failed: exitcode={writer.exitcode}", flush=True)

    print(
        f"generation finished db={args.db_path} model_pool={model_pool}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Province-based LLM data generator with SQLite storage.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Only compute province allocation.")
    plan_parser.add_argument("--target-records", type=int, default=DEFAULT_TARGET_RECORDS)
    plan_parser.add_argument("--samples-per-request", type=int, default=DEFAULT_SAMPLES_PER_REQUEST)
    plan_parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN_PATH)

    ingest_parser = subparsers.add_parser("ingest-eco", help="Import legacy sample_pool_eco data into SQLite.")
    ingest_parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    ingest_parser.add_argument("--eco-root", type=Path, default=DEFAULT_ECO_ROOT)
    ingest_parser.add_argument("--db-batch-size", type=int, default=1000)

    generate_parser = subparsers.add_parser("generate", help="Run 31 province processes and write to SQLite.")
    generate_parser.add_argument("--target-records", type=int, default=DEFAULT_TARGET_RECORDS)
    generate_parser.add_argument("--samples-per-request", type=int, default=DEFAULT_SAMPLES_PER_REQUEST)
    generate_parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    generate_parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    generate_parser.add_argument("--env-path", type=Path, default=Path(".env"))
    generate_parser.add_argument("--api-key", type=str, default=None)
    generate_parser.add_argument("--base-url", type=str, default=None)
    generate_parser.add_argument("--models", type=str, default=None)
    generate_parser.add_argument("--model-file", type=Path, default=DEFAULT_MODEL_FILE)
    generate_parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    generate_parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN_PATH)
    generate_parser.add_argument("--eco-root", type=Path, default=DEFAULT_ECO_ROOT)
    generate_parser.add_argument("--db-batch-size", type=int, default=1000)
    generate_parser.add_argument("--queue-maxsize", type=int, default=4096)
    generate_parser.add_argument("--max-inflight-requests", type=int, default=DEFAULT_MAX_INFLIGHT_REQUESTS)
    generate_parser.add_argument("--seed", type=int, default=20260319)
    generate_parser.add_argument("--workers-per-province", type=int, default=1)
    generate_parser.add_argument(
        "--import-eco-first",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "plan":
        province_plans, summary = compute_province_plan(args.target_records, args.samples_per_request)
        save_plan(args.plan_output, province_plans, summary)
        print_plan_summary(province_plans, summary)
        print(f"plan saved to {args.plan_output}", flush=True)
        return

    if args.command == "ingest-eco":
        ingest_eco_dataset(args.db_path, args.eco_root, args.db_batch_size)
        return

    if args.command == "generate":
        run_generate(args)
        return

    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
