from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from lxml import html


BASE_FIELDS = ["时间", "领域", "事件类别", "新闻标题", "新闻摘要", "原文链接"]
CANDIDATE_FIELDS = [
    "record_id",
    "date",
    "half_year",
    "domain",
    "category",
    "title_zh",
    "summary_zh",
    "event_text_en",
    "section",
    "source_label",
    "source_url",
    "source_domain",
    "source_type",
    "confidence",
    "score",
    "major_reason",
    "review_status",
    "duplicate_reason",
    "similarity_to_existing",
]

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

INCLUDE_SECTIONS = {
    "Science and technology",
    "Business and economy",
    "Health and environment",
    "International relations",
    "Law and crime",
    "Politics and elections",
}

SECTION_PRIOR = {
    "Science and technology": 22,
    "Health and environment": 14,
    "Business and economy": 12,
    "International relations": 8,
    "Law and crime": 7,
    "Politics and elections": 6,
}

TECH_TERMS = {
    "AI": [
        "artificial intelligence",
        "ai",
        "chatgpt",
        "openai",
        "deepmind",
        "large language model",
        "language model",
        "machine learning",
        "generative ai",
        "neural network",
        "foundation model",
        "google gemini",
        "claude",
        "llama",
    ],
    "芯片/半导体": [
        "semiconductor",
        "microchip",
        "chip",
        "processor",
        "gpu",
        "cpu",
        "nvidia",
        "tsmc",
        "intel",
        "amd",
        "lithography",
    ],
    "量子/计算": [
        "quantum",
        "qubit",
        "supercomputer",
        "exascale",
        "cloud computing",
        "edge computing",
        "high-performance computing",
        "github",
    ],
    "航天/天文": [
        "spacecraft",
        "space",
        "satellite",
        "rocket",
        "space launch",
        "lunar",
        "moon",
        "mars",
        "nasa",
        "spacex",
        "artemis",
        "james webb",
        "webb telescope",
        "telescope",
        "asteroid",
        "iss",
        "starship",
    ],
    "生物/医药": [
        "vaccine",
        "mrna",
        "gene",
        "genome",
        "genomic",
        "crispr",
        "biotech",
        "biotechnology",
        "fda approves",
        "clinical trial",
        "protein",
        "alphafold",
        "neuralink",
        "brain implant",
    ],
    "能源/材料": [
        "battery",
        "fusion",
        "nuclear",
        "solar",
        "hydrogen",
        "renewable",
        "electric vehicle",
        "ev",
        "tesla",
        "graphene",
    ],
    "机器人/自动化": [
        "robot",
        "robotics",
        "autonomous",
        "self-driving",
        "drone",
        "waymo",
        "automation",
    ],
    "网络安全/治理": [
        "cyber",
        "ransomware",
        "data breach",
        "hack",
        "privacy",
        "antitrust",
        "app store",
        "digital markets",
        "ai act",
    ],
    "消费电子/数字产业": [
        "iphone",
        "ipad",
        "mac",
        "vr",
        "augmented reality",
        "virtual reality",
        "playstation",
        "xbox",
        "streaming",
    ],
}

BAD_CONTEXT = [
    "rocket-propelled grenade",
    "missile strike",
    "airstrike",
    "militant",
    "shooting",
    "bombing",
    "killed",
    "dead",
    "death toll",
    "plane crash",
    "train crash",
]

OFFICIAL_DOMAINS = [
    "nasa.gov",
    "esa.int",
    "nih.gov",
    "fda.gov",
    "who.int",
    "ec.europa.eu",
    "europarl.europa.eu",
    "whitehouse.gov",
    "nist.gov",
    "energy.gov",
    "openai.com",
    "deepmind.google",
    "blog.google",
    "research.google",
    "microsoft.com",
    "github.blog",
    "apple.com",
    "nvidia.com",
    "intel.com",
    "amd.com",
    "tsmc.com",
    "samsung.com",
    "ibm.com",
    "meta.com",
    "about.fb.com",
    "anthropic.com",
    "moderna.com",
    "pfizer.com",
    "biontech.com",
]

AUTHORITATIVE_DOMAINS = [
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "nature.com",
    "science.org",
    "technologyreview.com",
    "spectrum.ieee.org",
    "space.com",
    "spacenews.com",
    "techcrunch.com",
    "theverge.com",
    "arstechnica.com",
    "scientificamerican.com",
    "newscientist.com",
    "statnews.com",
    "fiercebiotech.com",
]

VERB_PATTERNS = [
    (r"\bannounces?\b", "宣布"),
    (r"\breleases?\b", "发布"),
    (r"\blaunches?\b", "发射/启动"),
    (r"\bunveils?\b", "发布"),
    (r"\bintroduces?\b", "推出"),
    (r"\bapproves?\b", "批准"),
    (r"\bauthori[sz]es?\b", "授权"),
    (r"\bselects?\b", "选定"),
    (r"\bdevelops?\b", "开发"),
    (r"\bcompletes?\b", "完成"),
    (r"\bpublishes?\b", "发布"),
    (r"\bacquires?\b", "收购"),
    (r"\bsigns?\b", "签署"),
    (r"\breveals?\b", "公布"),
    (r"\bdiscovers?\b", "发现"),
    (r"\bfinds?\b", "发现"),
    (r"\bends support\b", "停止支持"),
]

BASE_ROW_OVERRIDES = {
    # Original seed rows used month-level dates. Keep the original event text
    # but normalize the simulation input to day-level dates/sources when a
    # traceable source date is available.
    "https://elevenlabs.io/blog/elevenlabs-launches-new-generative-voice-ai-products-and-announces-dollar19m-series-a-round-led-by-nat-friedman-daniel-gross-and-andreessen-horowitz": {
        "时间": "2023-01-23",
    },
    "https://openai.com/index/dall-e-3": {
        "时间": "2023-09-20",
    },
    "https://ir.kuaishou.com/news-releases/news-release-details/kuaishou-technology-announces-first-quarter-2025-unaudited": {
        "时间": "2025-04-15",
        "原文链接": "https://ir.kuaishou.com/news-releases/news-release-details/kling-ai-advances-20-era-empowering-everyone-tell-great-stories",
    },
}


@dataclass
class Candidate:
    record_id: str
    date: str
    half_year: str
    domain: str
    category: str
    title_zh: str
    summary_zh: str
    event_text_en: str
    section: str
    source_label: str
    source_url: str
    source_domain: str
    source_type: str
    confidence: str
    score: float
    major_reason: str
    review_status: str
    duplicate_reason: str = ""
    similarity_to_existing: float = 0.0

    def as_row(self) -> dict[str, str]:
        return {
            "record_id": self.record_id,
            "date": self.date,
            "half_year": self.half_year,
            "domain": self.domain,
            "category": self.category,
            "title_zh": self.title_zh,
            "summary_zh": self.summary_zh,
            "event_text_en": self.event_text_en,
            "section": self.section,
            "source_label": self.source_label,
            "source_url": self.source_url,
            "source_domain": self.source_domain,
            "source_type": self.source_type,
            "confidence": self.confidence,
            "score": f"{self.score:.1f}",
            "major_reason": self.major_reason,
            "review_status": self.review_status,
            "duplicate_reason": self.duplicate_reason,
            "similarity_to_existing": f"{self.similarity_to_existing:.3f}",
        }


def half_year_for_date(value: str) -> str:
    year, month, *_ = value.split("-")
    return f"{year}-S{1 if int(month) <= 6 else 2}"


def iter_periods(end_period: str) -> list[str]:
    periods: list[str] = []
    for year in range(2020, 2027):
        for half in (1, 2):
            period = f"{year}-S{half}"
            if period > end_period:
                return periods
            periods.append(period)
    return periods


def normalize_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url.strip())
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=False)
    query = [(k, v) for k, v in query if not k.lower().startswith("utm_")]
    normalized = parsed._replace(
        scheme=parsed.scheme.lower() or "https",
        netloc=parsed.netloc.lower(),
        query=urllib.parse.urlencode(query),
        fragment="",
    )
    text = urllib.parse.urlunsplit(normalized)
    return text.rstrip("/")


def apply_base_overrides(row: dict[str, str]) -> dict[str, str]:
    normalized = dict(row)
    original_url = normalize_url(normalized.get("原文链接", ""))
    override = BASE_ROW_OVERRIDES.get(original_url)
    if override:
        normalized.update(override)
    normalized["原文链接"] = normalize_url(normalized.get("原文链接", ""))
    return normalized


def domain_for_url(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc.lower().removeprefix("www.")


def domain_matches(domain: str, patterns: Iterable[str]) -> bool:
    return any(domain == item or domain.endswith(f".{item}") for item in patterns)


def source_type_for_domain(domain: str) -> tuple[str, str]:
    if domain_matches(domain, OFFICIAL_DOMAINS):
        return "official_or_primary", "high"
    if domain_matches(domain, AUTHORITATIVE_DOMAINS):
        return "authoritative_media", "medium_high"
    return "reference_media", "medium"


def normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"[\W_]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def clean_event_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*\([^()]{1,60}\)\s*$", "", text).strip()
    return text


def is_leaf_li(node) -> bool:
    return not node.xpath("./ul")


def terms_for_text(text: str) -> dict[str, int]:
    lowered = text.lower()
    hits: dict[str, int] = {}
    for domain, terms in TECH_TERMS.items():
        count = 0
        for term in terms:
            clean_term = term.strip().lower()
            pattern = r"(?<![a-z0-9])" + re.escape(clean_term).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
            if re.search(pattern, lowered):
                count += 1
        if count:
            hits[domain] = count
    return hits


def has_bad_context(text: str) -> bool:
    lowered = text.lower()
    if "ransomware" in lowered or "cyber" in lowered or "satellite" in lowered:
        return False
    return any(term in lowered for term in BAD_CONTEXT)


def score_candidate(text: str, section: str, source_type: str) -> float:
    hits = terms_for_text(text)
    if not hits:
        return -100.0
    if has_bad_context(text):
        return -50.0
    score = float(SECTION_PRIOR.get(section, 0))
    score += 5.0 * sum(hits.values())
    if source_type == "official_or_primary":
        score += 14.0
    elif source_type == "authoritative_media":
        score += 8.0
    else:
        score += 2.0
    lowered = text.lower()
    for strong in [
        "first",
        "record",
        "approves",
        "launches",
        "announces",
        "releases",
        "unveils",
        "breakthrough",
        "successfully",
        "completed",
        "law",
        "ban",
        "acquires",
    ]:
        if strong in lowered:
            score += 3.0
    return score


def classify_domain_and_category(text: str, section: str) -> tuple[str, str]:
    hits = terms_for_text(text)
    if hits:
        domain = sorted(hits.items(), key=lambda item: (-item[1], item[0]))[0][0]
    else:
        domain = "科技/前沿产业"
    lowered = text.lower()
    if any(term in lowered for term in ["approves", "authorizes", "law", "act", "ban", "antitrust", "regulation"]):
        category = "监管/审批/治理"
    elif any(term in lowered for term in ["launches", "rocket", "spacecraft", "satellite", "mission"]):
        category = "发射/任务/基础设施"
    elif any(term in lowered for term in ["announces", "releases", "unveils", "introduces"]):
        category = "发布/产品/平台"
    elif any(term in lowered for term in ["discovers", "finds", "study", "researchers", "breakthrough"]):
        category = "科研发现/技术突破"
    elif any(term in lowered for term in ["acquires", "merger", "investment", "factory", "plant"]):
        category = "产业投资/并购/产能"
    elif section in {"Law and crime", "Politics and elections", "International relations"}:
        category = "政策/治理/安全"
    else:
        category = "重大事件"
    return domain, category


def make_title_zh(text: str) -> str:
    trimmed = clean_event_text(text)
    first_sentence = trimmed[:180].strip()
    for pattern, verb_cn in VERB_PATTERNS:
        match = re.search(pattern, first_sentence, flags=re.IGNORECASE)
        if match:
            left = first_sentence[: match.start()].strip(" ,.;:")
            right = first_sentence[match.end() :].strip(" ,.;:")
            left = left[:48]
            right = right[:80]
            if left and right:
                return f"{left}{verb_cn}{right}"
    return f"科技前沿事件：{first_sentence[:110]}"


def make_summary_zh(candidate_date: str, text: str, domain: str, category: str, source_label: str) -> str:
    clean = clean_event_text(text)
    return (
        f"{candidate_date}，该事件被记录为{domain}方向的{category}。"
        f"英文事实摘录：{clean[:520]}。"
        f"来源引用为{source_label or 'Wikipedia Current Events外部来源'}，用于补充历史科技事实新闻语境。"
    )


def fetch_month_html(year: int, month: str, cache_dir: Path, sleep_seconds: float, retries: int) -> str | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{year}_{month}.html"
    if cache_path.exists() and cache_path.stat().st_size > 1000:
        return cache_path.read_text(encoding="utf-8")
    title = f"Portal:Current_events/{month}_{year}"
    url = "https://en.wikipedia.org/api/rest_v1/page/html/" + urllib.parse.quote(title, safe="")
    headers = {"User-Agent": "job-sim-frontier-news-enrichment/0.1 (local research dataset)"}
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=45) as response:
                body = response.read().decode("utf-8", errors="replace")
            cache_path.write_text(body, encoding="utf-8")
            time.sleep(sleep_seconds)
            return body
        except Exception as exc:
            wait = min(90.0, (2**attempt) * max(5.0, sleep_seconds))
            print(f"fetch_error {year} {month} attempt={attempt + 1} error={type(exc).__name__}: {exc}")
            time.sleep(wait)
    return None


def extract_candidates_from_month(year: int, month_index: int, body: str, current_date: date) -> list[Candidate]:
    doc = html.fromstring(body)
    candidates: list[Candidate] = []
    for day in doc.xpath('//div[contains(concat(" ", normalize-space(@class), " "), " current-events-main ")]'):
        date_text = day.xpath('string((.//span[contains(@class, "bday")])[1])').strip()
        if not date_text:
            continue
        try:
            event_date = date.fromisoformat(date_text)
        except ValueError:
            continue
        if event_date > current_date:
            continue
        content_nodes = day.xpath('.//div[contains(concat(" ", normalize-space(@class), " "), " current-events-content ")]')
        if not content_nodes:
            continue
        section = ""
        for child in content_nodes[0]:
            if child.tag == "p":
                section = " ".join(child.xpath(".//b//text()")).strip()
                continue
            if child.tag == "div" and "current-events-content-heading" in (child.get("class") or ""):
                section = " ".join(child.xpath(".//text()")).strip()
                continue
            if child.tag != "ul" or section not in INCLUDE_SECTIONS:
                continue
            for li in child.xpath(".//li"):
                if not is_leaf_li(li):
                    continue
                text = clean_event_text(" ".join(li.text_content().split()))
                if len(text) < 45:
                    continue
                external_urls = li.xpath('.//a[contains(concat(" ", normalize-space(@class), " "), " external ")]/@href')
                source_url = normalize_url(external_urls[0]) if external_urls else ""
                if not source_url:
                    continue
                source_label = li.xpath('string((.//a[contains(concat(" ", normalize-space(@class), " "), " external ")])[1])')
                source_label = source_label.strip(" ()")
                source_domain = domain_for_url(source_url)
                source_type, confidence = source_type_for_domain(source_domain)
                score = score_candidate(text, section, source_type)
                if score < 12.0:
                    continue
                domain, category = classify_domain_and_category(text, section)
                record_seed = f"{date_text}|{source_url}|{text}"
                record_id = "wiki_" + hashlib.sha1(record_seed.encode("utf-8")).hexdigest()[:14]
                title_zh = make_title_zh(text)
                summary_zh = make_summary_zh(date_text, text, domain, category, source_label)
                major_reason = f"Wikipedia Current Events/{section} 条目，匹配科技关键词并保留外部引用。"
                review_status = (
                    "assistant_screened_official_or_authoritative"
                    if source_type in {"official_or_primary", "authoritative_media"}
                    else "assistant_screened_reference"
                )
                candidates.append(
                    Candidate(
                        record_id=record_id,
                        date=date_text,
                        half_year=half_year_for_date(date_text),
                        domain=domain,
                        category=category,
                        title_zh=title_zh,
                        summary_zh=summary_zh,
                        event_text_en=text,
                        section=section,
                        source_label=source_label,
                        source_url=source_url,
                        source_domain=source_domain,
                        source_type=source_type,
                        confidence=confidence,
                        score=score,
                        major_reason=major_reason,
                        review_status=review_status,
                    )
                )
    return candidates


def read_base(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_pilot(path: Path | None) -> list[dict[str, str]]:
    if not path or not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pilot_to_base(row: dict[str, str]) -> dict[str, str]:
    return {
        "时间": row["date"],
        "领域": row["domain"],
        "事件类别": row["category"],
        "新闻标题": row["title_zh"],
        "新闻摘要": row["summary_zh"],
        "原文链接": row["source_url"],
    }


def candidate_to_base(row: Candidate) -> dict[str, str]:
    return {
        "时间": row.date,
        "领域": row.domain,
        "事件类别": row.category,
        "新闻标题": row.title_zh,
        "新闻摘要": row.summary_zh,
        "原文链接": row.source_url,
    }


def max_similarity(text: str, corpus: list[str]) -> float:
    norm = normalize_text(text)
    if not norm:
        return 0.0
    best = 0.0
    for other in corpus:
        other_norm = normalize_text(other)
        if not other_norm:
            continue
        if norm in other_norm or other_norm in norm:
            best = max(best, 0.92)
            continue
        if len(set(norm.split()) & set(other_norm.split())) < 2:
            continue
        best = max(best, SequenceMatcher(None, norm, other_norm).ratio())
    return best


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_source_catalog(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    catalog = []
    seen = set()
    for row in rows:
        url = row.get("原文链接") or row.get("source_url") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        domain = domain_for_url(url)
        source_type, confidence = source_type_for_domain(domain)
        catalog.append(
            {
                "source_id": "src_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12],
                "url": url,
                "domain": domain,
                "source_type": source_type,
                "confidence": confidence,
                "title_or_event": row.get("新闻标题") or row.get("title_zh") or "",
            }
        )
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich frontier technology news to about 50 items per half-year.")
    parser.add_argument("--base", default="data/base_seed_news.csv")
    parser.add_argument("--pilot", default="data/news_research_pilot_20260520/frontier_news_candidates_pilot.csv")
    parser.add_argument("--output", default="data/seed_news_enriched.csv")
    parser.add_argument("--output-dir", default="data/news_research_enriched_20260520")
    parser.add_argument("--target-per-period", type=int, default=50)
    parser.add_argument("--end-period", default="2026-S1")
    parser.add_argument("--current-date", default="2026-05-20")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--similarity-threshold", type=float, default=0.86)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache" / "wiki_current_events"
    current_date = date.fromisoformat(args.current_date)
    periods = iter_periods(args.end_period)

    base_rows = read_base(Path(args.base))
    pilot_rows = read_pilot(Path(args.pilot))

    merged: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    semantic_corpus: list[str] = []
    provenance_rows: list[dict[str, str]] = []
    skipped_base_duplicates: list[dict[str, str]] = []

    for row in base_rows:
        normalized_row = apply_base_overrides(row)
        url = normalized_row.get("原文链接", "")
        title = normalized_row.get("新闻标题", "").strip()
        if (url and url in seen_urls) or (title and title in seen_titles):
            skipped_base_duplicates.append(
                {
                    "时间": normalized_row.get("时间", ""),
                    "新闻标题": title,
                    "原文链接": url,
                    "duplicate_reason": "duplicate_seed_url_or_title",
                }
            )
            continue
        merged.append(normalized_row)
        if url:
            seen_urls.add(url)
        if title:
            seen_titles.add(title)
        semantic_corpus.append(f"{title} {row.get('新闻摘要', '')} {url}")
        provenance_rows.append(
            {
                "date": row.get("时间", ""),
                "half_year": half_year_for_date(row.get("时间", "1900-01-01")),
                "title": title,
                "source_url": url,
                "origin": "base_csv",
                "review_status": "existing_seed",
            }
        )

    for row in pilot_rows:
        if row.get("review_status") != "candidate_verified_primary":
            continue
        mapped = pilot_to_base(row)
        url = normalize_url(mapped["原文链接"])
        title = mapped["新闻标题"].strip()
        if url in seen_urls or title in seen_titles:
            continue
        mapped["原文链接"] = url
        merged.append(mapped)
        seen_urls.add(url)
        seen_titles.add(title)
        semantic_corpus.append(f"{title} {mapped.get('新闻摘要', '')} {url}")
        provenance_rows.append(
            {
                "date": mapped["时间"],
                "half_year": half_year_for_date(mapped["时间"]),
                "title": title,
                "source_url": url,
                "origin": "pilot_primary_source",
                "review_status": row.get("review_status", ""),
            }
        )

    all_candidates: list[Candidate] = []
    for year in range(2020, 2027):
        for month_index, month in enumerate(MONTHS, start=1):
            if f"{year}-S{1 if month_index <= 6 else 2}" > args.end_period:
                continue
            if date(year, month_index, 1) > current_date:
                continue
            body = fetch_month_html(year, month, cache_dir, args.sleep, args.retries)
            if not body:
                continue
            all_candidates.extend(extract_candidates_from_month(year, month_index, body, current_date))

    write_jsonl(output_dir / "wiki_current_events_candidates_all.jsonl", [c.as_row() for c in all_candidates])
    write_csv(output_dir / "wiki_current_events_candidates_all.csv", [c.as_row() for c in all_candidates], CANDIDATE_FIELDS)

    selected: list[Candidate] = []
    skipped: list[dict[str, str]] = []
    counts = Counter(half_year_for_date(row["时间"]) for row in merged)
    candidates_by_period: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in all_candidates:
        if candidate.half_year in periods:
            candidates_by_period[candidate.half_year].append(candidate)

    for period in periods:
        candidates = sorted(
            candidates_by_period.get(period, []),
            key=lambda item: (-item.score, item.date, item.title_zh),
        )
        for candidate in candidates:
            if counts[period] >= args.target_per_period:
                break
            url = normalize_url(candidate.source_url)
            title = candidate.title_zh.strip()
            if url in seen_urls:
                candidate.duplicate_reason = "duplicate_url"
            elif title in seen_titles:
                candidate.duplicate_reason = "duplicate_title"
            else:
                candidate.similarity_to_existing = max_similarity(
                    f"{candidate.title_zh} {candidate.summary_zh} {candidate.event_text_en}",
                    semantic_corpus,
                )
                if candidate.similarity_to_existing >= args.similarity_threshold:
                    candidate.duplicate_reason = "semantic_near_duplicate"
            if candidate.duplicate_reason:
                skipped.append(candidate.as_row())
                continue
            mapped = candidate_to_base(candidate)
            mapped["原文链接"] = url
            merged.append(mapped)
            selected.append(candidate)
            seen_urls.add(url)
            seen_titles.add(title)
            semantic_corpus.append(f"{candidate.title_zh} {candidate.summary_zh} {candidate.event_text_en} {url}")
            counts[period] += 1
            provenance_rows.append(
                {
                    "date": candidate.date,
                    "half_year": candidate.half_year,
                    "title": candidate.title_zh,
                    "source_url": url,
                    "origin": "wikipedia_current_events_reference",
                    "review_status": candidate.review_status,
                }
            )

    merged.sort(key=lambda row: (row.get("时间", ""), row.get("新闻标题", "")))
    output_path = Path(args.output)
    write_csv(output_path, merged, BASE_FIELDS)
    write_csv(output_dir / output_path.name, merged, BASE_FIELDS)
    write_csv(output_dir / "selected_wiki_candidates.csv", [c.as_row() for c in selected], CANDIDATE_FIELDS)
    write_csv(output_dir / "skipped_candidates_dedupe.csv", skipped, CANDIDATE_FIELDS)
    write_csv(
        output_dir / "skipped_base_duplicates.csv",
        skipped_base_duplicates,
        ["时间", "新闻标题", "原文链接", "duplicate_reason"],
    )
    write_csv(output_dir / "provenance_review_table.csv", provenance_rows, ["date", "half_year", "title", "source_url", "origin", "review_status"])

    source_catalog = build_source_catalog(merged)
    (output_dir / "source_catalog_enriched.json").write_text(
        json.dumps(source_catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    final_counts = Counter(half_year_for_date(row["时间"]) for row in merged)
    source_counts = Counter(row["origin"] for row in provenance_rows)
    status_counts = Counter(row["review_status"] for row in provenance_rows)
    period_rows = [
        {
            "half_year": period,
            "final_count": str(final_counts[period]),
            "target": str(args.target_per_period),
            "gap": str(max(0, args.target_per_period - final_counts[period])),
        }
        for period in periods
    ]
    write_csv(output_dir / "period_count_report.csv", period_rows, ["half_year", "final_count", "target", "gap"])

    audit_lines = [
        "# Frontier News Enrichment Audit",
        "",
        f"Base rows: {len(base_rows)}",
        f"Base rows retained after exact URL/title dedupe: {len(base_rows) - len(skipped_base_duplicates)}",
        f"Base duplicate rows skipped: {len(skipped_base_duplicates)}",
        f"Pilot primary-source rows accepted: {len([r for r in pilot_rows if r.get('review_status') == 'candidate_verified_primary'])}",
        f"Wikipedia filtered candidates: {len(all_candidates)}",
        f"Wikipedia selected rows: {len(selected)}",
        f"Final enriched rows: {len(merged)}",
        f"Output CSV: {output_path}",
        "",
        "## Counts by Period",
        "",
    ]
    for row in period_rows:
        audit_lines.append(f"- {row['half_year']}: {row['final_count']} (gap {row['gap']})")
    audit_lines.extend(
        [
            "",
            "## Provenance Counts",
            "",
        ]
    )
    for key, value in sorted(source_counts.items()):
        audit_lines.append(f"- {key}: {value}")
    audit_lines.extend(["", "## Review Status Counts", ""])
    for key, value in sorted(status_counts.items()):
        audit_lines.append(f"- {key}: {value}")
    audit_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The original base seed-news CSV is not overwritten.",
            "- The enriched CSV keeps the same six-column schema expected by job_sim.py.",
            "- Three month-level seed dates are normalized with traceable source dates before output.",
            "- Wikipedia Current Events is used as a dated discovery/reference layer; selected rows keep external source URLs.",
            "- Semantic near-duplicate filtering uses normalized text similarity plus exact title and URL checks.",
            "- Rows marked assistant_screened_reference should be sampled more heavily in any final human sign-off.",
        ]
    )
    (output_dir / "enrichment_audit_report.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    print(f"base_rows={len(base_rows)}")
    print(f"pilot_rows={len(pilot_rows)}")
    print(f"wiki_candidates={len(all_candidates)}")
    print(f"wiki_selected={len(selected)}")
    print(f"final_rows={len(merged)}")
    print(f"output={output_path}")
    for row in period_rows:
        print(f"{row['half_year']} final={row['final_count']} gap={row['gap']}")


if __name__ == "__main__":
    main()
