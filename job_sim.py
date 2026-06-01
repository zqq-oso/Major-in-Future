"""
蒙特卡洛未来热门学科专业预测系统
按半年期推进：真实/生成新闻 → 抽样智能体 → 并行预测热门专业方向 → Embedding+UMAP+HDBSCAN聚类 → LLM命名
"""
from __future__ import annotations
import asyncio
import csv
import json
import os
import random
import re
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import datetime

import dotenv
dotenv.load_dotenv()

import numpy as np
import umap
import hdbscan as hdbscan_lib
import litellm
from litellm import Router

litellm.drop_params = True
litellm.suppress_debug_info = True

# ── 多端点配置（从 configs/llm/llm_config.json 读取）─────────────────
_LLM_CONFIG_PATH = Path(os.getenv("LLM_CONFIG_PATH", "configs/llm/llm_config.json"))
_llm_cfg = json.loads(_LLM_CONFIG_PATH.read_text(encoding="utf-8"))
chat_router = Router(
    model_list=_llm_cfg["chat_models"],
    num_retries=9999,
    timeout=180,
    retry_after=2,
    routing_strategy="least-busy",
)
embed_router = Router(
    model_list=_llm_cfg["embed_models"],
    num_retries=9999,
    timeout=180,
    retry_after=2,
    routing_strategy="least-busy",
)

NEWS_CSV = Path(os.getenv("NEWS_CSV", "data/seed_news.csv"))
DB_PATH = Path(os.getenv("DB_PATH", "data/source_profiles.db"))
RUNS_DIR = Path("output/runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def _write_time(path: Path, t0: float, extra: dict | None = None) -> None:
    """将计时结果写入 .time 文件（JSON），供最终汇总用。"""
    elapsed = round(time.time() - t0, 2)
    record = {"start": datetime.datetime.utcfromtimestamp(t0).isoformat() + "Z",
              "end": _now_iso(), "elapsed_s": elapsed}
    if extra:
        record.update(extra)
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

def _collect_timing(run_dir: Path, quarter_seq: list[str]) -> dict:
    """汇总所有 .time 文件，返回 timing 字典。"""
    timing: dict = {"periods": {}}
    for q in quarter_seq:
        entry: dict = {}
        for stage in ("news", "predictions", "cluster"):
            tf = run_dir / stage / f"{q}.time"
            if not tf.exists():
                tf = run_dir / "clusters" / f"{q}.time"  # cluster 在 clusters/ 下
            if tf.exists():
                entry[stage] = json.loads(tf.read_text(encoding="utf-8"))
        if entry:
            timing["periods"][q] = entry
    global_tf = run_dir / "clusters" / "global_cluster.time"
    if global_tf.exists():
        timing["global_cluster"] = json.loads(global_tf.read_text(encoding="utf-8"))
    return timing


def make_run_dir() -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

# 模拟参数（均可通过同名环境变量覆盖，方便小规模测试）
TOTAL_PERIODS = int(os.getenv("TOTAL_PERIODS", "22"))            # 模拟半年期数（2020-S1 ~ 2030-S2）
HISTORY_END = os.getenv("HISTORY_END", "2026-S1")                # 历史/未来分界：<= 此值用真实新闻
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "10000"))             # 每季度预测 Agent 调用次数（百万总体抽1%=万人）
NEWS_CALLS_PER_QUARTER = int(os.getenv("NEWS_CALLS_PER_QUARTER", "100"))  # 每季度生成新闻的并发调用次数
NEWS_ITEMS_PER_CALL = 2         # 每次调用生成 2 条 → 每季度 200 条
NEWS_SAMPLE_PER_QUARTER = 5      # 每个季度采样多少条生成新闻（供预测 Agent）
NEWS_SAMPLE_FOR_EVOLVE = int(os.getenv("NEWS_SAMPLE_FOR_EVOLVE", "30"))  # 每次 evolve_news 调用可见的已推演新闻采样条数
NEWS_REAL_CONTEXT_LIMIT = int(os.getenv("NEWS_REAL_CONTEXT_LIMIT", "1000"))  # 未来新闻推演可见的历史/已知真实新闻上限；默认足够覆盖当前139条
NEWS_GENERATED_CONTEXT_PER_PERIOD = int(os.getenv("NEWS_GENERATED_CONTEXT_PER_PERIOD", "20"))  # 每个过去未来期进入候选池的推演新闻数
EVOLVE_HISTORY_WINDOW = int(os.getenv("EVOLVE_HISTORY_WINDOW", "4"))  # 从过去几个半年期采样已推演新闻
AGENT_NEWS_WINDOW = 1            # 预测 Agent 可见最近几个季度的新闻
PREDICTIONS_PER_AGENT = (1, 2)  # 每个预测 Agent 生成 1~2 条学科专业预测
CONCURRENCY = int(os.getenv("CONCURRENCY", "400"))               # 预测Agent并发（太高会触发限速反而更慢）
NEWS_CONCURRENCY = int(os.getenv("NEWS_CONCURRENCY", "100"))     # evolve_news 独立并发，不被预测任务挤占
MAX_INFLIGHT_FACTOR = int(os.getenv("MAX_INFLIGHT_FACTOR", "2")) # 滑动窗口大小 = CONCURRENCY × 此值
EMBED_BATCH_SIZE = 500      # 每批 embedding 请求的文本数
EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "20"))   # embedding 并发批次数
HDBSCAN_MIN_CLUSTER_SIZE = int(os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", "50"))  # 有效专业簇最小人数
HDBSCAN_MIN_SAMPLES = int(os.getenv("HDBSCAN_MIN_SAMPLES", "10"))
CENTROID_SAMPLE_COUNT = 10  # 每簇取距中心最近的N个样本命名
PROFILE_SNIPPET_CHAR_LIMIT = int(os.getenv("PROFILE_SNIPPET_CHAR_LIMIT", "600"))

NEWS_EVOLVE_FOCUS_AREAS = [
    "基础大模型与智能体",
    "机器人与具身智能",
    "芯片、算力与终端",
    "量子信息与下一代计算",
    "生物医药、脑科学与健康科技",
    "能源、材料与先进制造",
    "空天、低空经济与遥感",
    "网络安全、可信AI与治理",
    "教育科技与知识生产",
    "消费电子、内容生成与数字媒体",
]


# ── 人物 Profile（从数据库读取）────────────────────────
AGE_MAP = {"a": "0-18岁", "b": "19-35岁", "c": "36岁及以上"}
EDU_MAP = {"a": "高中及以下", "b": "本科及以上"}
GENDER_MAP = {"a": "男", "b": "女"}
CONSUMPTION_MAP = {"a": "低", "b": "中等", "c": "高"}


@dataclass
class Profile:
    id: int
    province: str
    age: str
    gender: str
    education: str
    consumption: str
    post_snippet: str

    def text(self) -> str:
        base = (
            f"省份：{self.province}，年龄：{self.age}，"
            f"性别：{self.gender}，学历：{self.education}，消费水平：{self.consumption}"
        )
        if self.post_snippet:
            base += f"\n该用户近期发言摘录：「{self.post_snippet}」"
        return base


def load_profiles_from_db(db_path: Path, limit: int = 10000) -> list[Profile]:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT rowid, province, demographic_label, content FROM posts ORDER BY RANDOM() LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    profiles = []
    for rowid, province, demo_json, content in rows:
        try:
            d = json.loads(demo_json)
        except (json.JSONDecodeError, TypeError):
            continue
        snippet = (content or "").replace("\n", " ").strip()
        if PROFILE_SNIPPET_CHAR_LIMIT > 0 and len(snippet) > PROFILE_SNIPPET_CHAR_LIMIT:
            snippet = snippet[:PROFILE_SNIPPET_CHAR_LIMIT].rstrip() + "..."
        profiles.append(Profile(
            id=rowid,
            province=province,
            age=AGE_MAP.get(d.get("AGE", "b"), "19-35岁"),
            gender=GENDER_MAP.get(d.get("GENDER", "b"), "女"),
            education=EDU_MAP.get(d.get("Education", "b"), "本科及以上"),
            consumption=CONSUMPTION_MAP.get(d.get("Level of Consumption", "b"), "中等"),
            post_snippet=snippet,
        ))
    return profiles


# ── 新闻加载 ──────────────────────────────────────────
@dataclass
class NewsEvent:
    date: str
    domain: str
    category: str
    title: str
    summary: str

    def text(self) -> str:
        return f"[{self.date}] {self.title}：{self.summary}"


def load_seed_news(csv_path: Path) -> list[NewsEvent]:
    events = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(NewsEvent(
                date=row["时间"],
                domain=row.get("领域", ""),
                category=row.get("事件类别", ""),
                title=row["新闻标题"],
                summary=row["新闻摘要"],
            ))
    return events


def news_to_half_year_map(events: list[NewsEvent]) -> dict[str, list[NewsEvent]]:
    """将 CSV 日期映射到半年期：1-6月→S1，7-12月→S2"""
    hmap: dict[str, list[NewsEvent]] = {}
    for e in events:
        parts = e.date.split("-")
        if len(parts) >= 2:
            year, month = int(parts[0]), int(parts[1])
            s = 1 if month <= 6 else 2
            key = f"{year}-S{s}"
            hmap.setdefault(key, []).append(e)
    return hmap


def build_half_year_sequence(n: int = 22) -> list[str]:
    """生成固定半年期序列：2020-S1, 2020-S2, ..., 2030-S2（共22个）"""
    seq: list[str] = []
    year, s = 2020, 1
    for _ in range(n):
        seq.append(f"{year}-S{s}")
        if s == 1:
            s = 2
        else:
            s = 1
            year += 1
    return seq


DATE_MENTION_PATTERNS = (
    re.compile(r"(20\d{2})年\s*(\d{1,2})月(?:\s*\d{1,2}日)?"),
    re.compile(r"\b(20\d{2})[-/](\d{1,2})(?:[-/]\d{1,2})?\b"),
)
BARE_MONTH_MENTION_PATTERN = re.compile(r"(?<![\d年-])([1-9]|1[0-2])月(?:\s*\d{1,2}日)?(?![\d个份])")
QUARTER_MENTION_PATTERNS = (
    re.compile(r"(20\d{2})年\s*第?([一二三四1234])季度"),
    re.compile(r"\b(20\d{2})\s*Q([1-4])\b", re.IGNORECASE),
)
HALF_YEAR_MENTION_PATTERN = re.compile(r"(20\d{2})年\s*(上半年|下半年)")
_QUARTER_TO_INT = {"一": 1, "二": 2, "三": 3, "四": 4, "1": 1, "2": 2, "3": 3, "4": 4}


def half_year_month_range(quarter_label: str) -> tuple[int, int, int]:
    """Return (year, start_month, end_month) for labels like 2026-S2."""
    year_text, half_text = quarter_label.split("-S", 1)
    year = int(year_text)
    half = int(half_text)
    if half == 1:
        return year, 1, 6
    return year, 7, 12


def date_rule_text(quarter_label: str) -> str:
    year, start_month, end_month = half_year_month_range(quarter_label)
    return f"{year}年{start_month}月1日到{year}年{end_month}月31日之间"


def explicit_dates_match_half_year(text: str, quarter_label: str) -> bool:
    """Reject generated future-news text that mentions dates outside the target half-year."""
    year, start_month, end_month = half_year_month_range(quarter_label)
    target_half = 1 if end_month == 6 else 2
    for pattern in DATE_MENTION_PATTERNS:
        for match in pattern.finditer(text):
            mentioned_year = int(match.group(1))
            mentioned_month = int(match.group(2))
            if mentioned_year != year or not (start_month <= mentioned_month <= end_month):
                return False
    for match in BARE_MONTH_MENTION_PATTERN.finditer(text):
        mentioned_month = int(match.group(1))
        if not (start_month <= mentioned_month <= end_month):
            return False
    for pattern in QUARTER_MENTION_PATTERNS:
        for match in pattern.finditer(text):
            mentioned_year = int(match.group(1))
            mentioned_quarter = _QUARTER_TO_INT[match.group(2)]
            mentioned_half = 1 if mentioned_quarter <= 2 else 2
            if mentioned_year != year or mentioned_half != target_half:
                return False
    for match in HALF_YEAR_MENTION_PATTERN.finditer(text):
        mentioned_year = int(match.group(1))
        mentioned_half = 1 if match.group(2) == "上半年" else 2
        if mentioned_year != year or mentioned_half != target_half:
            return False
    return True


def collect_real_news_context(
    qmap: dict[str, list[NewsEvent]],
    period_seq: list[str],
    current_index: int,
) -> list[NewsEvent]:
    """收集当前期之前/当前期可用的全部历史种子新闻。"""
    events: list[NewsEvent] = []
    for q in period_seq[:current_index + 1]:
        events.extend(qmap.get(q, []))
    if NEWS_REAL_CONTEXT_LIMIT > 0 and len(events) > NEWS_REAL_CONTEXT_LIMIT:
        return events[-NEWS_REAL_CONTEXT_LIMIT:]
    return events


def collect_generated_news_context(
    generated_by_quarter: dict[str, list[NewsEvent]],
    period_seq: list[str],
    current_index: int,
) -> list[NewsEvent]:
    """从最近几个未来期分层抽取已推演新闻，作为每次 LLM 调用的候选池。"""
    events: list[NewsEvent] = []
    start = max(0, current_index - EVOLVE_HISTORY_WINDOW)
    for q in period_seq[start:current_index]:
        if q <= HISTORY_END:
            continue
        pool = generated_by_quarter.get(q, [])
        if pool:
            events.extend(
                random.sample(pool, min(NEWS_GENERATED_CONTEXT_PER_PERIOD, len(pool)))
            )
    return events


# 向后兼容别名
def news_to_quarter_map(events: list[NewsEvent]) -> dict[str, list[NewsEvent]]:
    return news_to_half_year_map(events)


def build_quarter_sequence(qmap: dict[str, list[NewsEvent]], n: int) -> list[str]:
    return build_half_year_sequence(n)


# ── JSON 修复工具 ─────────────────────────────────────
def _extract_json_array(raw: str) -> list | None:
    """从 LLM 输出中提取 JSON 数组，容忍常见格式错误。"""
    # 去掉 markdown 代码块
    text = re.sub(r'```(?:json)?\s*', '', raw).replace('```', '').strip()
    # 找第一个 [ 到最后一个 ]
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1 or end <= start:
        return None
    fragment = text[start:end + 1]
    # 先直接尝试
    try:
        result = json.loads(fragment)
        return result if isinstance(result, list) else None
    except json.JSONDecodeError:
        pass
    # 用 ast.literal_eval 容忍单引号
    try:
        import ast
        result = ast.literal_eval(fragment)
        return result if isinstance(result, list) else None
    except Exception:
        pass
    return None


# ── LLM 调用 ─────────────────────────────────────────
async def llm_chat(system: str, user: str, temperature: float = 0.9) -> str:
    attempt = 0
    while True:
        try:
            resp = await chat_router.acompletion(
                model="chat",
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                timeout=180,
            )
            content = resp.choices[0].message.content
            if content is None:
                return ""
            return content.strip()
        except Exception as e:
            attempt += 1
            msg = str(e)
            is_rate_limit = "429" in msg or "rate_limit" in msg.lower() or "too many requests" in msg.lower()
            is_timeout    = "timeout" in msg.lower() or "timed out" in msg.lower()
            is_transient  = is_rate_limit or is_timeout or "503" in msg or "502" in msg or "connection" in msg.lower()
            if is_transient:
                wait = min(2 ** min(attempt, 8), 60) + random.uniform(0, 2)
                tag = "限速" if is_rate_limit else "超时/连接"
                print(f"  [{tag}] {wait:.1f}s 后重试（第{attempt}次）", flush=True)
                await asyncio.sleep(wait)
            else:
                raise


# ── 阶段1：环境演进 ──────────────────────────────────────
# 100 次并发，每次生成 2 条 → 每半年期约 200 条科技事件
# 每次调用可见全部历史种子新闻，并从近期已推演新闻候选池中独立采样。
async def evolve_news(real_news: list[NewsEvent],
                      prev_generated: list[NewsEvent],
                      quarter_label: str) -> list[NewsEvent]:

    async def one_call(call_index: int, sem: asyncio.Semaphore) -> list[NewsEvent]:
        # 每次调用独立采样一组近期推演新闻，历史种子新闻则完整保留。
        sampled = random.sample(
            prev_generated, min(NEWS_SAMPLE_FOR_EVOLVE, len(prev_generated))
        ) if prev_generated else []
        focus = NEWS_EVOLVE_FOCUS_AREAS[call_index % len(NEWS_EVOLVE_FOCUS_AREAS)]

        ctx_parts = []
        if real_news:
            ctx_parts.append("历史/已知科技事件（截至当前期，可全部参考）：\n" + "\n".join(e.text() for e in real_news))
        if sampled:
            ctx_parts.append("近期已推演科技事件（随机采样）：\n" + "\n".join(e.text() for e in sampled))
        context = "\n\n".join(ctx_parts)
        valid_date_range = date_rule_text(quarter_label)

        prompt = (
            f"当前时间：{quarter_label}。\n"
            f"本期合法日期范围：{valid_date_range}。\n"
            f"{context}\n\n"
            f"本次推演重点：{focus}。请围绕该重点生成与其他方向有区分度的事件，但保持与整体科技趋势一致。\n"
            f"基于上述事件与趋势，推演并生成本半年期（{quarter_label}）的 {NEWS_ITEMS_PER_CALL} 条核心科技新闻。\n"
            f"硬性日期规则：标题和摘要中凡是出现 YYYY年M月、YYYY年M月D日、YYYY-MM 或 YYYY-MM-DD，"
            f"或者单独出现 M月/M月D日，都必须落在{valid_date_range}；"
            f"凡是出现第几季度、Q1-Q4、上半年、下半年，也必须与{quarter_label}一致；"
            f"不要写当前期以外月份，也不要用旧事件的具体日期作为新闻发生日期。"
            "避免只改写已有新闻标题，避免重复同一产品线或同一机构的近似发布。\n"
            f"严格输出 JSON 数组，每个元素包含 title 和 summary 字段，不要输出其他内容。"
        )
        async with sem:
            raw = await llm_chat("你是一个科技趋势分析师。", prompt, temperature=0.8)
            items = _extract_json_array(raw)
            if items is None:
                return []
            events = []
            for item in items:
                if isinstance(item, dict):
                    event = NewsEvent(
                        date=quarter_label, domain="AI", category="LLM推演",
                        title=item.get("title") or item.get("Title") or str(item.get("title", "未知")),
                        summary=item.get("summary") or item.get("Summary") or str(item.get("summary", "")),
                    )
                    if explicit_dates_match_half_year(f"{event.title}\n{event.summary}", quarter_label):
                        events.append(event)
                elif isinstance(item, str) and item.strip():
                    event = NewsEvent(
                        date=quarter_label, domain="AI", category="LLM推演",
                        title=item.strip(), summary="",
                    )
                    if explicit_dates_match_half_year(event.title, quarter_label):
                        events.append(event)
            return events

    sem = asyncio.Semaphore(NEWS_CONCURRENCY)
    results = await asyncio.gather(*[one_call(i, sem) for i in range(NEWS_CALLS_PER_QUARTER)])
    events = [e for batch in results for e in batch]
    return events


# ── 阶段2：并行预测未来热门学科专业 ─────────────────────
@dataclass
class Prediction:
    quarter: str
    profile_id: int
    profile_text: str
    major_name: str
    rationale: str

    @property
    def text(self) -> str:
        return f"{self.major_name} {self.rationale}".strip()


def prediction_from_record(d: dict) -> Prediction:
    """Load current major records, with old job-field fallback for interrupted legacy runs."""
    major_name = d.get("major_name") or d.get("job_title") or ""
    rationale = d.get("rationale") or d.get("one_line_rationale") or d.get("description") or ""
    return Prediction(
        quarter=d["quarter"],
        profile_id=d["profile_id"],
        profile_text=d["profile_text"],
        major_name=major_name,
        rationale=rationale,
    )


def prediction_embedding_text(prediction: Prediction) -> str:
    return f"{prediction.major_name}: {prediction.rationale}".strip()


async def predict_majors(
    profile: Profile,
    quarter: str,
    real_news: list[NewsEvent],
    sampled_prev: list[NewsEvent],
    sem: asyncio.Semaphore,
) -> list[Prediction]:
    """
    给定 Agent 的个人背景 + 真实新闻全量 + 上一季度生成新闻采样（10条），
    让 Agent 生成 1~2 条未来热门学科专业/专业方向预测。
    """
    ctx_parts = []
    if real_news:
        ctx_parts.append("近期真实科技动态：\n" + "\n".join(e.text() for e in real_news))
    if sampled_prev:
        ctx_parts.append("近期推演的科技事件（采样）：\n" + "\n".join(e.text() for e in sampled_prev))
    context = "\n\n".join(ctx_parts)

    prompt = (
        f"当前时间：{quarter}\n"
        f"你的个人背景：{profile.text()}\n\n"
        f"近期科技动态：\n{context}\n\n"
        "请结合你的个人背景、近期科技发展和社会需求变化，预测未来1-3年内在高校招生、"
        "继续教育或跨学科培养中最可能变热门的1个学科专业或专业方向。"
        "要求符合教育培养逻辑、产业需求和技术现状，避免把具体岗位、公司产品、培训课程或科幻概念当成专业。\n"
        '严格输出JSON数组，每个元素包含 major_name 和 one_line_rationale 字段，输出1到2个元素，不要输出其他内容。'
    )
    async with sem:
        try:
            raw = await llm_chat("你是一个关注升学选择、产业趋势和高等教育变化的普通人。", prompt, temperature=1.0)
            items = _extract_json_array(raw)
            if items is None:
                return []
            preds = []
            for item in items:
                if isinstance(item, dict) and (item.get("major_name") or item.get("job_title")):
                    major_name = item.get("major_name") or item.get("job_title", "")
                    preds.append(Prediction(
                        quarter=quarter,
                        profile_id=profile.id,
                        profile_text=profile.text(),
                        major_name=major_name,
                        rationale=(
                            item.get("one_line_rationale", "")
                            or item.get("rationale", "")
                            or item.get("one_line_description", "")
                            or item.get("description", "")
                        ),
                    ))
                elif isinstance(item, str) and item.strip():
                    preds.append(Prediction(
                        quarter=quarter,
                        profile_id=profile.id,
                        profile_text=profile.text(),
                        major_name=item.strip(),
                        rationale="",
                    ))
            return preds
        except Exception as e:
            print(f"  [ERR] profile {profile.id}: {e}")
            return []


# ── 阶段3：Embedding ──────────────────────────────────
async def embed_texts(texts: list[str]) -> np.ndarray:
    """批量 embed，所有批次并发发送，返回 (N, D) float32 数组。"""
    batches = [texts[i:i + EMBED_BATCH_SIZE] for i in range(0, len(texts), EMBED_BATCH_SIZE)]
    sem = asyncio.Semaphore(EMBED_CONCURRENCY)

    async def _one_batch(batch: list[str]) -> list[list[float]]:
        async with sem:
            resp = await embed_router.aembedding(model="embed", input=batch)
            return [item["embedding"] for item in sorted(resp.data, key=lambda x: x["index"])]

    print(f"    embedding {len(texts)} 条，{len(batches)} 批并发（cap={EMBED_CONCURRENCY}）...", flush=True)
    results = await asyncio.gather(*[_one_batch(b) for b in batches])
    all_embeddings: list[list[float]] = [vec for batch_result in results for vec in batch_result]
    return np.array(all_embeddings, dtype=np.float32)


# ── 阶段4：UMAP + HDBSCAN + LLM命名 ──────────────────
async def cluster_predictions(predictions: list[Prediction], quarter: str) -> list[dict]:
    if len(predictions) < HDBSCAN_MIN_CLUSTER_SIZE * 2:
        print(f"  [WARN] 预测数量({len(predictions)})不足以聚类")
        return []

    print(f"  embedding {len(predictions)} 条专业预测...", flush=True)
    texts = [prediction_embedding_text(p) for p in predictions]
    embeddings = await embed_texts(texts)

    print("  UMAP 降维...", flush=True)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        low_memory=True,
    )
    reduced = reducer.fit_transform(embeddings)

    print("  HDBSCAN 聚类...", flush=True)
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
    )
    labels = clusterer.fit_predict(reduced)

    unique_labels = sorted(set(labels) - {-1})
    noise_count = int((labels == -1).sum())
    print(f"  聚类结果：{len(unique_labels)} 个簇，{noise_count} 个噪声点", flush=True)

    # 并发对每个簇命名
    sem = asyncio.Semaphore(10)

    async def name_cluster(label: int) -> dict:
        mask = labels == label
        indices = np.where(mask)[0]
        cluster_reduced = reduced[mask]
        centroid = cluster_reduced.mean(axis=0)
        dists = np.linalg.norm(cluster_reduced - centroid, axis=1)
        nearest_idx = indices[np.argsort(dists)[:CENTROID_SAMPLE_COUNT]]
        samples = [predictions[i] for i in nearest_idx]

        sample_text = "\n".join(f"- {prediction_embedding_text(s)}" for s in samples)
        prompt = (
            f"以下是一组相似的未来热门学科专业或专业方向预测：\n{sample_text}\n\n"
            "提取这些预测的共性，生成一个高度概括、标准、适合高校专业目录或跨学科方向表达的名称（不超过12个字）。"
            "只输出专业方向名称，不要其他内容。"
        )
        try:
            async with sem:
                name = await llm_chat("你是一个学科专业分类专家。", prompt, temperature=0.3)
            name = name.strip().strip('"').strip("'")
        except asyncio.CancelledError:
            name = f"专业方向簇{label}"
            print(f"  [WARN] 簇 {label} 命名请求被取消，使用兜底名称", flush=True)
        except Exception as e:
            name = f"专业方向簇{label}"
            print(f"  [WARN] 簇 {label} 命名失败，使用兜底名称: {e}", flush=True)

        return {
            "cluster_name": name,
            "label": int(label),
            "count": int(mask.sum()),
            "centroid": centroid.tolist(),
            "member_indices": indices.tolist(),
        }

    cluster_results = await asyncio.gather(*[name_cluster(lbl) for lbl in unique_labels])
    clusters = list(cluster_results)

    if noise_count > 0:
        clusters.append({
            "cluster_name": "噪声（孤立专业预测）",
            "label": -1,
            "count": noise_count,
            "centroid": None,
            "member_indices": np.where(labels == -1)[0].tolist(),
        })

    return clusters


# ── 单季度聚类（供聚类进程调用）───────────────────────────
async def _cluster_one_quarter(run_dir: Path, q_label: str):
    """读取 predictions/{q}.jsonl，embedding + 聚类，写入 clusters/{q}.json 和 embeddings/{q}.npy。"""
    t0 = time.time()
    pred_path = run_dir / "predictions" / f"{q_label}.jsonl"
    if not pred_path.exists():
        return

    preds: list[Prediction] = []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            preds.append(prediction_from_record(d))

    if len(preds) < HDBSCAN_MIN_CLUSTER_SIZE * 2:
        print(f"[聚类] {q_label} 预测不足（{len(preds)}），跳过")
        return

    emb_path = run_dir / "embeddings" / f"{q_label}.npy"
    if emb_path.exists():
        embeddings = np.load(str(emb_path))
        print(f"[聚类] {q_label}: 加载已有 embedding，shape={embeddings.shape}")
    else:
        texts = [prediction_embedding_text(p) for p in preds]
        embeddings = await embed_texts(texts)
        (run_dir / "embeddings").mkdir(exist_ok=True)
        np.save(str(emb_path), embeddings)
        print(f"[聚类] {q_label}: {len(preds)} 条预测，embedding shape={embeddings.shape}，已存盘")

    # UMAP + HDBSCAN + 命名
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, low_memory=True)
    reduced = reducer.fit_transform(embeddings)
    clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES)
    labels = clusterer.fit_predict(reduced)
    unique_labels = sorted(set(labels) - {-1})
    noise_count = int((labels == -1).sum())

    sem = asyncio.Semaphore(10)

    async def name_one(label: int) -> dict:
        mask = labels == label
        indices = np.where(mask)[0]
        sub_reduced = reduced[mask]
        centroid = sub_reduced.mean(axis=0)
        dists = np.linalg.norm(sub_reduced - centroid, axis=1)
        nearest = indices[np.argsort(dists)[:CENTROID_SAMPLE_COUNT]]
        samples = [preds[i] for i in nearest]
        sample_text = "\n".join(f"- {prediction_embedding_text(s)}" for s in samples)
        prompt = (
            f"以下是一组相似的未来热门学科专业或专业方向预测：\n{sample_text}\n\n"
            "提取这些预测的共性，生成一个高度概括、标准、适合高校专业目录或跨学科方向表达的名称（不超过12个字）。"
            "只输出专业方向名称，不要其他内容。"
        )
        try:
            async with sem:
                name = await llm_chat("你是一个学科专业分类专家。", prompt, temperature=0.3)
            name = name.strip().strip('"').strip("'")
        except asyncio.CancelledError:
            name = f"专业方向簇{label}"
            print(f"[聚类] {q_label} 簇 {label} 命名请求被取消，使用兜底名称", flush=True)
        except Exception as e:
            name = f"专业方向簇{label}"
            print(f"[聚类] {q_label} 簇 {label} 命名失败，使用兜底名称: {e}", flush=True)
        return {
            "cluster_name": name,
            "label": int(label),
            "count": int(mask.sum()),
            "centroid": centroid.tolist(),
            "member_indices": indices.tolist(),
        }

    clusters = await asyncio.gather(*[name_one(lbl) for lbl in unique_labels])
    clusters = list(clusters)
    if noise_count > 0:
        clusters.append({
            "cluster_name": "噪声（孤立专业预测）",
            "label": -1,
            "count": noise_count,
            "centroid": None,
            "member_indices": np.where(labels == -1)[0].tolist(),
        })

    with open(run_dir / "clusters" / f"{q_label}.json", "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    elapsed = time.time() - t0
    _write_time(run_dir / "clusters" / f"{q_label}.time", t0, {"clusters": len(clusters), "predictions": len(preds)})
    print(f"[聚类] {q_label} 完成：{len(clusters)} 个簇（不含噪声），耗时 {elapsed:.1f}s")


# ── 主循环（3 进程流水线）────────────────────────────────
#  进程 1（evolve_news）：主进程顺序执行，每个半年期生成 200 条新闻，写文件后发信号
#  进程 2（预测 Agent）：后台 asyncio 轮询，半年期新闻就绪后立即启动并发 Agent
#  进程 3（聚类）：后台 asyncio 轮询，所有半年期预测就绪后一次性聚类
#  信号机制：每个半年期完成后写 .done 文件，后台进程轮询检测

POLL_INTERVAL = 10  # 秒


async def run_prediction_for_period(
    q_label: str,
    run_dir: Path,
    profiles: list[Profile],
    real_by_quarter: dict[str, list[NewsEvent]],
    generated_by_quarter: dict[str, list[NewsEvent]],
    period_seq: list[str] | None = None,
) -> list[Prediction]:
    """为一个半年期运行预测 Agent：读取新闻文件 → 并发 → 写预测文件。"""
    # 加载本期新闻（JSONL）
    news_path = run_dir / "news" / f"{q_label}.jsonl"
    generated_this = []
    if news_path.exists():
        with open(news_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    generated_this.append(NewsEvent(**json.loads(line)))

    # 加载历史新闻
    qmap = news_to_half_year_map(load_seed_news(NEWS_CSV))
    if period_seq is None:
        period_seq = build_half_year_sequence(TOTAL_PERIODS)

    i = period_seq.index(q_label) if q_label in period_seq else 0

    window_periods = period_seq[max(0, i - AGENT_NEWS_WINDOW + 1):i + 1]
    agent_real_news: list[NewsEvent] = []
    agent_sampled_gen: list[NewsEvent] = []
    is_history = q_label <= HISTORY_END
    for wq in window_periods:
        # 真实新闻从 CSV 加载的历史中取
        agent_real_news.extend(qmap.get(wq, []))
        # 历史期只用真实新闻，不从 generated_by_quarter 采样
        if not is_history:
            gen_pool = generated_by_quarter.get(wq, [])
            if gen_pool:
                agent_sampled_gen.extend(
                    random.sample(gen_pool, min(NEWS_SAMPLE_PER_QUARTER, len(gen_pool)))
                )

    print(f"[预测Agent] {q_label}: 真实 {len(agent_real_news)} 条 + 生成采样 {len(agent_sampled_gen)} 条")

    t0_pred = time.time()
    sem = asyncio.Semaphore(CONCURRENCY)
    MAX_INFLIGHT = CONCURRENCY * MAX_INFLIGHT_FACTOR

    # 流式写入预测文件，边跑边写，不攒到最后
    pred_path = run_dir / "predictions" / f"{q_label}.jsonl"
    progress_path = run_dir / "predictions" / f"{q_label}.progress.json"

    preds: list[Prediction] = []
    existing_records = 0
    if pred_path.exists():
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    preds.append(prediction_from_record(json.loads(line)))
                    existing_records += 1
                except Exception:
                    continue

    pending: set[asyncio.Task] = set()
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            progress = {}
        processed = min(int(progress.get("processed_calls", 0)), SAMPLE_SIZE)
        fired = min(int(progress.get("fired_calls", processed)), SAMPLE_SIZE)
    else:
        # Older interrupted runs did not track completed calls. Since each successful call
        # emits 1-2 prediction rows, use a conservative estimate and keep existing rows.
        processed = min(int(round(existing_records / 1.5)), SAMPLE_SIZE) if existing_records else 0
        fired = processed

    if existing_records or processed:
        print(
            f"[预测Agent] {q_label}: 断点续跑 existing_records={existing_records}, "
            f"processed_calls={processed}, fired_calls={fired}",
            flush=True,
        )

    pred_file = open(pred_path, "a" if existing_records else "w", encoding="utf-8")

    def write_progress(done: bool = False) -> None:
        progress_path.write_text(
            json.dumps(
                {
                    "quarter": q_label,
                    "processed_calls": processed,
                    "fired_calls": fired,
                    "records": len(preds),
                    "target_calls": SAMPLE_SIZE,
                    "done": done,
                    "updated_at": _now_iso(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    try:
        while processed < SAMPLE_SIZE:
            # 补充 inflight 到 MAX_INFLIGHT
            while len(pending) < MAX_INFLIGHT and fired < SAMPLE_SIZE:
                profile = random.choice(profiles)
                t = asyncio.create_task(
                    predict_majors(profile, q_label, agent_real_news, agent_sampled_gen, sem)
                )
                pending.add(t)
                fired += 1

            if not pending:
                break

            # 等待至少一个完成，立即处理，无批次空档
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for t in done:
                processed += 1
                try:
                    result = t.result()
                    for p in result:
                        preds.append(p)
                        pred_file.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
                except Exception:
                    pass  # predict_majors 内部已打印错误

            if processed % 500 == 0 or processed == SAMPLE_SIZE:
                pred_file.flush()
                write_progress(done=False)
                elapsed = time.time() - t0_pred
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (SAMPLE_SIZE - processed) / rate if rate > 0 else 0
                print(f"    [{q_label}] {processed}/{SAMPLE_SIZE}  fired={fired}  "
                      f"{rate:.1f}req/s  ETA {eta:.0f}s", flush=True)
    finally:
        # 取消残余 inflight（正常情况 pending 为空）
        for t in pending:
            t.cancel()
        pred_file.close()

    # 发信号：预测完成 + 计时
    (run_dir / "predictions" / f"{q_label}.done").write_text("")
    write_progress(done=True)
    elapsed_pred = time.time() - t0_pred
    _write_time(
        run_dir / "predictions" / f"{q_label}.time",
        t0_pred,
        {"count": len(preds), "processed_calls": processed},
    )
    print(f"[预测Agent] {q_label} 完成，{len(preds)} 条预测，耗时 {elapsed_pred:.1f}s")

    return preds


async def _poll_and_run_predictions(
    run_dir: Path,
    quarter_seq: list[str],
    profiles: list[Profile],
    done_event: asyncio.Event,
):
    """后台轮询：等待 evolve_news 写完每季度新闻，立即启动预测 Agent。"""
    predicted: set[str] = set()
    real_by_quarter: dict[str, list[NewsEvent]] = {}
    generated_by_quarter: dict[str, list[NewsEvent]] = {}
    pending_tasks: dict[str, asyncio.Task] = {}

    print("[预测Agent] 启动，轮询中...")
    # 续跑幂等：已有 predictions.done 的季度直接标记为完成
    for q_label in quarter_seq:
        if (run_dir / "predictions" / f"{q_label}.done").exists():
            predicted.add(q_label)
            print(f"[预测Agent] {q_label} 预测已完成，跳过")
    while len(predicted) < len(quarter_seq):
        await asyncio.sleep(POLL_INTERVAL)
        for q_label in quarter_seq:
            if q_label in predicted:
                continue
            news_file = run_dir / "news" / f"{q_label}.done"
            if news_file.exists() and q_label not in pending_tasks:
                # 加载新闻到内存
                news_path = run_dir / "news" / f"{q_label}.jsonl"
                if news_path.exists():
                    evts = []
                    with open(news_path, encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                evts.append(NewsEvent(**json.loads(line)))
                    generated_by_quarter[q_label] = evts
                print(f"[预测Agent] 检测到 {q_label} 新闻就绪，启动...")
                task = asyncio.create_task(
                    run_prediction_for_period(q_label, run_dir, profiles,
                                               real_by_quarter, generated_by_quarter)
                )
                pending_tasks[q_label] = task

        # 收集已完成任务
        for q_label in list(pending_tasks):
            t = pending_tasks[q_label]
            if t.done():
                try:
                    t.result()
                    predicted.add(q_label)
                    del pending_tasks[q_label]
                except Exception as e:
                    print(f"[预测Agent] {q_label} 出错: {e}")

    done_event.set()
    print(f"[预测Agent] 全部 {len(quarter_seq)} 半年期预测完成！")


async def _poll_and_cluster(
    run_dir: Path,
    quarter_seq: list[str],
    done_event: asyncio.Event,
):
    """
    聚类进程：
      - 轮询 predictions/*.done，检测到哪个季度完成就立即启动聚类（季度间并行）
      - 所有季度聚类跑完后，对全部 Embedding 重做一次全局 UMAP + 聚类
    """
    print("[聚类进程] 启动，轮询预测完成信号...")

    pending_quarters = set(quarter_seq)
    running_tasks: dict[str, asyncio.Task] = {}
    clustered: list[str] = []

    # 续跑幂等：已有 clusters/{q}.json 的直接标记为完成
    for q_label in quarter_seq:
        if (run_dir / "clusters" / f"{q_label}.json").exists():
            pending_quarters.discard(q_label)
            clustered.append(q_label)
            print(f"[聚类进程] {q_label} 聚类已完成，跳过")

    while pending_quarters or running_tasks:
        await asyncio.sleep(POLL_INTERVAL)

        # 启动新完成的季度聚类
        for q_label in list(pending_quarters):
            if (run_dir / "predictions" / f"{q_label}.done").exists():
                pending_quarters.remove(q_label)
                task = asyncio.create_task(_cluster_one_quarter(run_dir, q_label))
                running_tasks[q_label] = task
                print(f"[聚类进程] 检测到 {q_label} 预测完成，启动聚类（{len(clustered) + len(running_tasks)}/{len(quarter_seq)}）")

        # 收集已完成任务
        for q_label in list(running_tasks):
            t = running_tasks[q_label]
            if t.done():
                running_tasks.pop(q_label)
                try:
                    t.result()
                    clustered.append(q_label)
                except asyncio.CancelledError as e:
                    pending_quarters.add(q_label)
                    print(f"[聚类进程] {q_label} 聚类任务被取消，稍后重试: {e}")
                except Exception as e:
                    pending_quarters.add(q_label)
                    print(f"[聚类进程] {q_label} 聚类出错: {e}")
                else:
                    print(f"[聚类进程] {q_label} 聚类完成（{len(clustered)}/{len(quarter_seq)}）")

    print(f"[聚类进程] 全部 {len(quarter_seq)} 半年期聚类完毕。开始全局重聚...")
    t0_global = time.time()

    # ── 全局重聚：全部 Embedding 合并，重新 UMAP + 聚类 ──
    all_embeddings_list: list[np.ndarray] = []
    all_predictions_list: list[Prediction] = []
    for q_label in quarter_seq:
        emb_path = run_dir / "embeddings" / f"{q_label}.npy"
        pred_path = run_dir / "predictions" / f"{q_label}.jsonl"
        if not emb_path.exists():
            continue
        emb = np.load(str(emb_path))
        all_embeddings_list.append(emb)
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                all_predictions_list.append(prediction_from_record(d))

    if not all_embeddings_list:
        print("[聚类进程] 无 Embedding，全局聚类跳过")
        (run_dir / "clusters" / "all.done").write_text("")
        return

    all_emb = np.vstack(all_embeddings_list)
    print(f"[聚类进程] 全局重聚：{all_emb.shape[0]} 条 embedding，shape={all_emb.shape}")

    print("  全局 UMAP 降维...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        low_memory=True,
    )
    reduced = reducer.fit_transform(all_emb)

    print("  全局 HDBSCAN 聚类...")
    clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES)
    labels = clusterer.fit_predict(reduced)
    unique_labels = sorted(set(labels) - {-1})
    noise_count = int((labels == -1).sum())
    print(f"  全局聚类结果：{len(unique_labels)} 个簇，{noise_count} 个噪声点")

    # 按 quarter 拆分 member_indices
    global_member_indices: dict[str, list[int]] = {q: [] for q in quarter_seq}
    offset = 0
    for q_label in quarter_seq:
        n = len(all_embeddings_list[quarter_seq.index(q_label)])
        for idx in range(offset, offset + n):
            if labels[idx] != -1:
                global_member_indices[q_label].append(idx)
        offset += n

    # 全局 LLM 命名
    sem = asyncio.Semaphore(10)
    preds_flat = all_predictions_list

    async def name_one(label: int) -> dict:
        mask = labels == label
        indices = np.where(mask)[0]
        sub_reduced = reduced[mask]
        centroid = sub_reduced.mean(axis=0)
        dists = np.linalg.norm(sub_reduced - centroid, axis=1)
        nearest = indices[np.argsort(dists)[:CENTROID_SAMPLE_COUNT]]
        samples = [preds_flat[i] for i in nearest]
        sample_text = "\n".join(f"- {prediction_embedding_text(s)}" for s in samples)
        prompt = (
            f"以下是一组相似的未来热门学科专业或专业方向预测：\n{sample_text}\n\n"
            "提取这些预测的共性，生成一个高度概括、标准、适合高校专业目录或跨学科方向表达的名称（不超过12个字）。"
            "只输出专业方向名称，不要其他内容。"
        )
        async with sem:
            name = await llm_chat("你是一个学科专业分类专家。", prompt, temperature=0.3)
        name = name.strip().strip('"').strip("'")
        return {
            "cluster_name": name,
            "label": int(label),
            "count": int(mask.sum()),
            "centroid": centroid.tolist(),
            "member_indices": indices.tolist(),
        }

    clusters = await asyncio.gather(*[name_one(lbl) for lbl in unique_labels])
    clusters = list(clusters)
    if noise_count > 0:
        clusters.append({
            "cluster_name": "噪声（孤立专业预测）",
            "label": -1,
            "count": noise_count,
            "centroid": None,
            "member_indices": np.where(labels == -1)[0].tolist(),
        })

    # 写全局结果
    with open(run_dir / "clusters_all.json", "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    np.save(str(run_dir / "embeddings_all.npy"), all_emb)

    # 更新每季度的 clusters 文件（加入全局 cluster_label）
    for q_label in quarter_seq:
        cp = run_dir / "clusters" / f"{q_label}.json"
        if cp.exists():
            local = json.loads(cp.read_text(encoding="utf-8"))
            # 给每条 member_indices 附上全局 label
            for c in local:
                c["global_label"] = int(c.get("label", -1))
            with open(cp, "w", encoding="utf-8") as f:
                json.dump(local, f, ensure_ascii=False, indent=2)

    (run_dir / "clusters" / "all.done").write_text("")
    elapsed_global = time.time() - t0_global
    _write_time(run_dir / "clusters" / "global_cluster.time", t0_global,
                {"clusters": len(clusters), "embeddings": all_emb.shape[0]})
    print(f"[聚类进程] 全局聚类完成，{len(clusters)} 个簇，{all_emb.shape[0]} 条 embedding，耗时 {elapsed_global:.1f}s")


async def run():
    run_dir = make_run_dir()
    (run_dir / "news").mkdir()
    (run_dir / "predictions").mkdir()
    (run_dir / "clusters").mkdir()
    (run_dir / "embeddings").mkdir()

    print(f"=== 未来热门学科专业预测蒙特卡洛模拟 ===")
    print(f"运行目录: {run_dir}")
    chat_models = [m["litellm_params"]["model"] for m in _llm_cfg["chat_models"]]
    print(f"参数: 半年期数={TOTAL_PERIODS}, 历史分界={HISTORY_END}, 每期Agent={SAMPLE_SIZE}次, "
          f"evolve={NEWS_CALLS_PER_QUARTER}×{NEWS_ITEMS_PER_CALL}={NEWS_CALLS_PER_QUARTER*NEWS_ITEMS_PER_CALL}条, "
          f"W={AGENT_NEWS_WINDOW}, 模型={chat_models}")
    print(f"真实新闻输入: {NEWS_CSV}")

    # 加载数据
    seed_news = load_seed_news(NEWS_CSV)
    print(f"加载真实新闻 {len(seed_news)} 条")
    qmap = news_to_half_year_map(seed_news)
    quarter_seq = build_half_year_sequence(TOTAL_PERIODS)
    print(f"半年期序列: {quarter_seq}")

    print("从数据库加载人物档案...")
    profiles = load_profiles_from_db(DB_PATH, limit=1_000_000)
    print(f"加载 {len(profiles)} 个人物档案")

    # 写 meta.json
    meta = {
        "run_id": run_dir.name,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "params": {
            "total_periods": TOTAL_PERIODS,
            "history_end": HISTORY_END,
            "news_csv": str(NEWS_CSV),
            "sample_size": SAMPLE_SIZE,
            "news_calls_per_quarter": NEWS_CALLS_PER_QUARTER,
            "news_items_per_call": NEWS_ITEMS_PER_CALL,
            "news_sample_per_quarter": NEWS_SAMPLE_PER_QUARTER,
            "news_sample_for_evolve": NEWS_SAMPLE_FOR_EVOLVE,
            "news_real_context_limit": NEWS_REAL_CONTEXT_LIMIT,
            "news_generated_context_per_period": NEWS_GENERATED_CONTEXT_PER_PERIOD,
            "evolve_history_window": EVOLVE_HISTORY_WINDOW,
            "future_news_date_validation": "reject_explicit_dates_outside_target_half_year",
            "agent_news_window": AGENT_NEWS_WINDOW,
            "predictions_per_agent": PREDICTIONS_PER_AGENT,
            "prediction_target": "future_hot_disciplines_and_majors",
            "concurrency": CONCURRENCY,
            "chat_models": [m["litellm_params"]["model"] for m in _llm_cfg["chat_models"]],
            "embed_models": [m["litellm_params"]["model"] for m in _llm_cfg["embed_models"]],
            "clustering": "major_prediction_embedding+umap+hdbscan_all_periods",
            "hdbscan_min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
            "flow": "三进程流水线: evolve_news顺序→专业预测Agent轮询→聚类轮询",
        },
        "quarters": quarter_seq,
        "status": "running",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 启动后台进程
    preds_done_event = asyncio.Event()

    agent_task = asyncio.create_task(
        _poll_and_run_predictions(run_dir, quarter_seq, profiles, preds_done_event)
    )
    cluster_task = asyncio.create_task(
        _poll_and_cluster(run_dir, quarter_seq, preds_done_event)
    )

    # ── 进程 1：evolve_news（主进程顺序执行）──────────────
    generated_by_quarter: dict[str, list[NewsEvent]] = {}
    real_by_quarter: dict[str, list[NewsEvent]] = {}
    news_timeline: dict[str, list[NewsEvent]] = {}

    t0_news = time.time()
    news_times: list[float] = []

    for i, q_label in enumerate(quarter_seq):
        t0_q = time.time()
        print(f"\n{'='*50}")
        print(f"[evolve_news] 半年期 {q_label} ({i+1}/{len(quarter_seq)})")
        print(f"{'='*50}")

        quarter_real = qmap.get(q_label, [])
        real_by_quarter[q_label] = quarter_real
        print(f"  真实新闻 {len(quarter_real)} 条")

        # 历史阶段（<= HISTORY_END）：跳过 evolve_news，直接用真实新闻
        is_history = q_label <= HISTORY_END
        if is_history:
            print(f"  [历史期] 跳过 evolve_news，直接使用真实新闻")
            generated_this = quarter_real  # 真实新闻作为本期"生成"新闻
            generated_by_quarter[q_label] = generated_this
            news_timeline[q_label] = generated_this

            # 写新闻文件（JSONL）—— 写真实新闻
            news_path = run_dir / "news" / f"{q_label}.jsonl"
            with open(news_path, "w", encoding="utf-8") as f:
                for e in generated_this:
                    f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

            (run_dir / "news" / f"{q_label}.done").write_text("")
            _write_time(run_dir / "news" / f"{q_label}.time", t0_q, {"generated": len(generated_this), "type": "history"})
            elapsed_q = time.time() - t0_q
            print(f"  [历史期] {q_label} 完成，{len(generated_this)} 条真实新闻，耗时 {elapsed_q:.1f}s")
            continue

        # 未来阶段：正常调用 evolve_news。真实/种子新闻数量少，默认全部放入上下文；
        # 已推演新闻数量较大，先按最近期分层抽样成候选池，再由每次调用独立采样。
        real_for_evolve = collect_real_news_context(qmap, quarter_seq, i)
        sampled_for_evolve = collect_generated_news_context(generated_by_quarter, quarter_seq, i)
        print(
            f"  真实/种子上下文 {len(real_for_evolve)} 条；"
            f"已推演新闻候选 {len(sampled_for_evolve)} 条，"
            f"每次调用最多采样 {NEWS_SAMPLE_FOR_EVOLVE} 条"
        )

        print(f"  生成科技事件（{NEWS_CALLS_PER_QUARTER}×{NEWS_ITEMS_PER_CALL}={NEWS_CALLS_PER_QUARTER*NEWS_ITEMS_PER_CALL}条）...")
        generated_this = await evolve_news(real_for_evolve, sampled_for_evolve, q_label)
        elapsed_q = time.time() - t0_q
        news_times.append(elapsed_q)
        print(f"  推演新闻 {len(generated_this)} 条")
        generated_by_quarter[q_label] = generated_this
        news_timeline[q_label] = generated_this

        # 写新闻文件（JSONL）
        news_path = run_dir / "news" / f"{q_label}.jsonl"
        with open(news_path, "w", encoding="utf-8") as f:
            for e in generated_this:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

        # 发信号：新闻完成
        (run_dir / "news" / f"{q_label}.done").write_text("")
        _write_time(run_dir / "news" / f"{q_label}.time", t0_q, {"generated": len(generated_this)})
        # ETA（聚类进程和预测进程也受并发影响，这里给出 evolve_news 的理论下限 ETA）
        avg_t = sum(news_times) / len(news_times)
        remaining = len(quarter_seq) - len(news_times)
        eta_total = avg_t * remaining
        print(f"  [evolve_news] {q_label} 完成，已写 {news_path}，耗时 {elapsed_q:.1f}s  |  预计还需 {eta_total:.0f}s")

    print("\n[evolve_news] 全部半年期新闻生成完毕！")

    # ── 等待聚类完成 ────────────────────────────────────
    await cluster_task
    agent_task.cancel()

    # ── 汇总 ────────────────────────────────────────────
    print("\n汇总结果...")
    timeline_path = run_dir / "news_timeline.json"
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump({q: [asdict(e) for e in evts] for q, evts in news_timeline.items()},
                  f, ensure_ascii=False, indent=2)
    print(f"新闻时间线 → {timeline_path}")

    # ── 计时汇总 ────────────────────────────────────────
    timing = _collect_timing(run_dir, quarter_seq)
    news_total  = sum(t.get("news",        {}).get("elapsed_s", 0) for t in timing.get("periods", {}).values())
    pred_total  = sum(t.get("predictions", {}).get("elapsed_s", 0) for t in timing.get("periods", {}).values())
    clust_total = sum(t.get("cluster",     {}).get("elapsed_s", 0) for t in timing.get("periods", {}).values())
    g_total = timing.get("global_cluster", {}).get("elapsed_s", 0)
    run_elapsed = time.time() - t0_news if t0_news else 0
    print(f"\n计时报告:")
    print(f"  evolve_news   累计 {news_total:.1f}s")
    print(f"  预测Agent     累计 {pred_total:.1f}s")
    print(f"  单期聚类      累计 {clust_total:.1f}s")
    print(f"  全局聚类              {g_total:.1f}s")
    print(f"  总耗时（约）  {run_elapsed:.1f}s")

    # 更新 meta
    total_preds = sum(
        1 for q in quarter_seq
        for _ in open(run_dir / "predictions" / f"{q}.jsonl", encoding="utf-8")
        if _.strip()
    ) if (run_dir / "predictions").exists() else 0
    meta["status"] = "completed"
    meta["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta["total_predictions"] = total_preds
    meta["timing"] = timing
    meta["timing_summary"] = {
        "news_total_s": round(news_total, 1),
        "predictions_total_s": round(pred_total, 1),
        "cluster_total_s": round(clust_total, 1),
        "global_cluster_s": round(g_total, 1),
        "run_elapsed_s": round(run_elapsed, 1),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 模拟完成 ===")
    print(f"运行目录: {run_dir}")


async def resume_run(run_dir: Path) -> None:
    """
    续跑一个中断的 run。
    根据目录里已有的 .done 文件判断进度，只补跑缺失的部分：
      · news.done 存在  → 跳过 evolve_news
      · predictions.done 存在 → 跳过预测 Agent（_poll_and_run_predictions 内部幂等）
      · clusters/{q}.json 存在 → 跳过单季聚类（_poll_and_cluster 内部幂等）
    最后无论如何都会重做全局重聚（因为可能有新季度加入）。
    """
    if not run_dir.exists():
        raise SystemExit(f"run 目录不存在: {run_dir}")

    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"找不到 meta.json: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    quarter_seq: list[str] = meta["quarters"]

    news_done   = {q for q in quarter_seq if (run_dir / "news"        / f"{q}.done").exists()}
    preds_done  = {q for q in quarter_seq if (run_dir / "predictions" / f"{q}.done").exists()}
    clust_done  = {q for q in quarter_seq if (run_dir / "clusters"    / f"{q}.json").exists()}

    remaining_news  = [q for q in quarter_seq if q not in news_done]
    remaining_preds = [q for q in quarter_seq if q not in preds_done]

    # ── 恢复原跑参数（env var 优先，否则读 meta.json）────────────────────────
    global SAMPLE_SIZE, NEWS_CALLS_PER_QUARTER, NEWS_ITEMS_PER_CALL
    global NEWS_SAMPLE_PER_QUARTER, NEWS_SAMPLE_FOR_EVOLVE, NEWS_REAL_CONTEXT_LIMIT
    global NEWS_GENERATED_CONTEXT_PER_PERIOD, AGENT_NEWS_WINDOW
    global EVOLVE_HISTORY_WINDOW
    global PREDICTIONS_PER_AGENT, CONCURRENCY, HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES
    p = meta.get("params", {})
    def _param(env_key: str, meta_key: str, default, cast=int):
        if os.getenv(env_key) is not None:
            return cast(os.getenv(env_key))
        return cast(p.get(meta_key, default))

    SAMPLE_SIZE             = _param("SAMPLE_SIZE",             "sample_size",             SAMPLE_SIZE)
    NEWS_CALLS_PER_QUARTER  = _param("NEWS_CALLS_PER_QUARTER",  "news_calls_per_quarter",  NEWS_CALLS_PER_QUARTER)
    NEWS_ITEMS_PER_CALL     = _param("NEWS_ITEMS_PER_CALL",     "news_items_per_call",     NEWS_ITEMS_PER_CALL)
    NEWS_SAMPLE_PER_QUARTER = _param("NEWS_SAMPLE_PER_QUARTER", "news_sample_per_quarter", NEWS_SAMPLE_PER_QUARTER)
    NEWS_SAMPLE_FOR_EVOLVE  = _param("NEWS_SAMPLE_FOR_EVOLVE",  "news_sample_for_evolve",  NEWS_SAMPLE_FOR_EVOLVE)
    NEWS_REAL_CONTEXT_LIMIT = _param("NEWS_REAL_CONTEXT_LIMIT", "news_real_context_limit", NEWS_REAL_CONTEXT_LIMIT)
    NEWS_GENERATED_CONTEXT_PER_PERIOD = _param(
        "NEWS_GENERATED_CONTEXT_PER_PERIOD",
        "news_generated_context_per_period",
        NEWS_GENERATED_CONTEXT_PER_PERIOD,
    )
    EVOLVE_HISTORY_WINDOW    = _param("EVOLVE_HISTORY_WINDOW",   "evolve_history_window",   EVOLVE_HISTORY_WINDOW)
    AGENT_NEWS_WINDOW       = _param("AGENT_NEWS_WINDOW",       "agent_news_window",       AGENT_NEWS_WINDOW)
    CONCURRENCY             = _param("CONCURRENCY",             "concurrency",             CONCURRENCY)
    HDBSCAN_MIN_CLUSTER_SIZE= _param("HDBSCAN_MIN_CLUSTER_SIZE","hdbscan_min_cluster_size",HDBSCAN_MIN_CLUSTER_SIZE)
    HDBSCAN_MIN_SAMPLES     = _param("HDBSCAN_MIN_SAMPLES",     "hdbscan_min_samples",     HDBSCAN_MIN_SAMPLES)
    if isinstance(p.get("predictions_per_agent"), list):
        PREDICTIONS_PER_AGENT = tuple(p["predictions_per_agent"])

    print(f"=== 续跑 {run_dir.name} ===")
    print(f"  参数: SAMPLE_SIZE={SAMPLE_SIZE}, NEWS_CALLS={NEWS_CALLS_PER_QUARTER}, "
          f"NEWS_REAL_CONTEXT_LIMIT={NEWS_REAL_CONTEXT_LIMIT}, "
          f"NEWS_GENERATED_CONTEXT_PER_PERIOD={NEWS_GENERATED_CONTEXT_PER_PERIOD}, "
          f"EVOLVE_HISTORY_WINDOW={EVOLVE_HISTORY_WINDOW}, "
          f"CONCURRENCY={CONCURRENCY}, HDBSCAN_MIN_CLUSTER_SIZE={HDBSCAN_MIN_CLUSTER_SIZE}, "
          f"HDBSCAN_MIN_SAMPLES={HDBSCAN_MIN_SAMPLES}")
    print(f"  半年期总数: {len(quarter_seq)}")
    print(f"  新闻已完成: {len(news_done)}/{len(quarter_seq)}  待跑: {remaining_news or '无'}")
    print(f"  预测已完成: {len(preds_done)}/{len(quarter_seq)}  待跑: {remaining_preds or '无'}")
    print(f"  聚类已完成: {len(clust_done)}/{len(quarter_seq)}")

    if not remaining_news and not remaining_preds:
        print("  所有半年期已完成，仅重做全局聚类。")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    seed_news = load_seed_news(NEWS_CSV)
    qmap      = news_to_half_year_map(seed_news)

    print("从数据库加载人物档案...")
    profiles = load_profiles_from_db(DB_PATH, limit=1_000_000)
    print(f"加载 {len(profiles)} 个人物档案")

    # 恢复已生成的新闻到内存（供后续季度的 evolve_news 使用）
    generated_by_quarter: dict[str, list[NewsEvent]] = {}
    for q in news_done:
        news_path = run_dir / "news" / f"{q}.jsonl"
        if news_path.exists():
            evts = []
            with open(news_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        evts.append(NewsEvent(**json.loads(line)))
            generated_by_quarter[q] = evts

    # ── 启动后台任务（内部均已幂等）──────────────────────────────────────────
    # 重置全局聚类标志，让 _poll_and_cluster 重做全局重聚
    all_done_flag = run_dir / "clusters" / "all.done"
    if all_done_flag.exists():
        all_done_flag.unlink()
        print("  已重置 clusters/all.done，将重做全局重聚")

    preds_done_event = asyncio.Event()
    agent_task   = asyncio.create_task(
        _poll_and_run_predictions(run_dir, quarter_seq, profiles, preds_done_event)
    )
    cluster_task = asyncio.create_task(
        _poll_and_cluster(run_dir, quarter_seq, preds_done_event)
    )

    # ── 主进程：补跑缺失的 evolve_news ────────────────────────────────────────
    news_timeline: dict[str, list[NewsEvent]] = {}
    t0_news = time.time()
    news_times: list[float] = []

    for i, q_label in enumerate(quarter_seq):
        if q_label in news_done:
            # 已有新闻，加载到 timeline 以便最终写入
            evts = generated_by_quarter.get(q_label, [])
            news_timeline[q_label] = evts
            continue

        t0_q = time.time()
        print(f"\n{'='*50}")
        print(f"[evolve_news] 半年期 {q_label} ({i+1}/{len(quarter_seq)})")
        print(f"{'='*50}")

        quarter_real = qmap.get(q_label, [])
        print(f"  真实新闻 {len(quarter_real)} 条")

        # 历史阶段（<= HISTORY_END）：跳过 evolve_news，直接用真实新闻
        is_history = q_label <= HISTORY_END
        if is_history:
            print(f"  [历史期] 跳过 evolve_news，直接使用真实新闻")
            generated_this = quarter_real
            generated_by_quarter[q_label] = generated_this
            news_timeline[q_label] = generated_this

            news_path = run_dir / "news" / f"{q_label}.jsonl"
            with open(news_path, "w", encoding="utf-8") as f:
                for e in generated_this:
                    f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
            (run_dir / "news" / f"{q_label}.done").write_text("")
            _write_time(run_dir / "news" / f"{q_label}.time", t0_q, {"generated": len(generated_this), "type": "history"})
            elapsed_q = time.time() - t0_q
            print(f"  [历史期] {q_label} 完成，{len(generated_this)} 条真实新闻，耗时 {elapsed_q:.1f}s")
            continue

        # 未来阶段：正常调用 evolve_news。真实/种子新闻数量少，默认全部放入上下文；
        # 已推演新闻数量较大，先按最近期分层抽样成候选池，再由每次调用独立采样。
        real_for_evolve = collect_real_news_context(qmap, quarter_seq, i)
        sampled_for_evolve = collect_generated_news_context(generated_by_quarter, quarter_seq, i)
        print(
            f"  真实/种子上下文 {len(real_for_evolve)} 条；"
            f"已推演新闻候选 {len(sampled_for_evolve)} 条，"
            f"每次调用最多采样 {NEWS_SAMPLE_FOR_EVOLVE} 条"
        )

        generated_this = await evolve_news(real_for_evolve, sampled_for_evolve, q_label)
        elapsed_q = time.time() - t0_q
        news_times.append(elapsed_q)
        print(f"  推演新闻 {len(generated_this)} 条")
        generated_by_quarter[q_label] = generated_this
        news_timeline[q_label] = generated_this

        news_path = run_dir / "news" / f"{q_label}.jsonl"
        with open(news_path, "w", encoding="utf-8") as f:
            for e in generated_this:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
        (run_dir / "news" / f"{q_label}.done").write_text("")
        _write_time(run_dir / "news" / f"{q_label}.time", t0_q, {"generated": len(generated_this)})
        avg_t = sum(news_times) / len(news_times)
        remaining = len(remaining_news) - len(news_times)
        eta_total = avg_t * remaining if remaining > 0 else 0
        print(f"  [evolve_news] {q_label} 完成，耗时 {elapsed_q:.1f}s  |  新闻待跑 {remaining} 个半年期，ETA {eta_total:.0f}s")

    print("\n[evolve_news] 全部半年期新闻生成完毕！")

    # ── 等待聚类完成 ──────────────────────────────────────────────────────────
    await cluster_task
    agent_task.cancel()

    # ── 更新 news_timeline.json（补充新季度）────────────────────────────────
    timeline_path = run_dir / "news_timeline.json"
    existing_timeline: dict = {}
    if timeline_path.exists():
        existing_timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    for q, evts in news_timeline.items():
        if evts:
            existing_timeline[q] = [asdict(e) for e in evts]
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(existing_timeline, f, ensure_ascii=False, indent=2)
    print(f"新闻时间线 → {timeline_path}")

    # ── 更新 meta ─────────────────────────────────────────────────────────────
    total_preds = sum(
        sum(1 for line in open(run_dir / "predictions" / f"{q}.jsonl", encoding="utf-8") if line.strip())
        for q in quarter_seq
        if (run_dir / "predictions" / f"{q}.jsonl").exists()
    )
    timing = _collect_timing(run_dir, quarter_seq)
    news_total  = sum(t.get("news",        {}).get("elapsed_s", 0) for t in timing.get("periods", {}).values())
    pred_total  = sum(t.get("predictions", {}).get("elapsed_s", 0) for t in timing.get("periods", {}).values())
    clust_total = sum(t.get("cluster",     {}).get("elapsed_s", 0) for t in timing.get("periods", {}).values())
    g_total = timing.get("global_cluster", {}).get("elapsed_s", 0)
    print(f"\n计时报告:")
    print(f"  evolve_news   累计 {news_total:.1f}s")
    print(f"  预测Agent     累计 {pred_total:.1f}s")
    print(f"  单期聚类      累计 {clust_total:.1f}s")
    print(f"  全局聚类              {g_total:.1f}s")
    meta["status"]           = "completed"
    meta["resumed_at"]       = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta["total_predictions"] = total_preds
    meta["timing"] = timing
    meta["timing_summary"] = {
        "news_total_s": round(news_total, 1),
        "predictions_total_s": round(pred_total, 1),
        "cluster_total_s": round(clust_total, 1),
        "global_cluster_s": round(g_total, 1),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 续跑完成 ===  运行目录: {run_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "--resume":
        if len(sys.argv) < 3:
            print("用法: python job_sim.py --resume <run_dir>")
            print("示例: python job_sim.py --resume output/runs/run_20260320_145554")
            sys.exit(1)
        asyncio.run(resume_run(Path(sys.argv[2])))
    else:
        asyncio.run(run())
