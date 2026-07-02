"""
Scrapes the SHL product catalog, restricted to Individual Test Solutions
(type=1). Pre-packaged Job Solutions (type=2) are explicitly excluded.

The live listing at shl.com/solutions/products/product-catalog/ has been
restructured on the current site (it now 301-redirects to a generic
marketing page), so this script sources the listing + detail pages from
the Wayback Machine's most recent full crawl (~March/April 2025), using
its flexible-timestamp lookup to fetch the closest available snapshot
for each URL. Every scraped detail-page URL is the *live* shl.com URL
(not a wayback-prefixed one) and is verified to still resolve (following
redirects) before being kept, so every link in the final catalog is a
real, working SHL page today.

Output:
  data/raw/catalog_listing.json    -- one row per assessment from the listing tables
  data/raw/catalog_detail/*.json   -- one file per assessment with full detail-page fields
  data/processed/catalog.json      -- final merged, cleaned catalog used by the app
"""

import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from html import unescape

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DETAIL_DIR = RAW_DIR / "catalog_detail"
PROCESSED_DIR = ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
DETAIL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BASE_LISTING = "https://www.shl.com/solutions/products/product-catalog/"
CDX_API = "https://web.archive.org/cdx/search/cdx"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

TEST_TYPE_NAMES = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behaviour",
    "S": "Simulations",
}

REQUEST_DELAY_SECONDS = 0.25
MAX_RETRIES = 3


def fetch(url: str, timeout: int = 25) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                # A handful of archived pages serve non-UTF-8 bytes (e.g. a
                # Windows-1252 degree sign) inside an otherwise UTF-8 page;
                # falling back to cp1252 recovers those characters instead
                # of silently dropping them (errors="ignore" would produce
                # mangled names like "360� Multi-Rater...").
                return raw.decode("cp1252", errors="replace")
        except urllib.error.HTTPError as e:
            # 404/410 = definitively no snapshot at this exact URL; retrying won't help.
            raise RuntimeError(f"failed to fetch {url}: {e}")
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            time.sleep(3.0 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_err}")


def fetch_json(url: str, timeout: int = 60):
    return json.loads(fetch(url, timeout=timeout))


def build_listing_snapshot_map() -> dict:
    """Query the CDX API once for every archived variant of the paginated
    listing URL, then pick the best (type=1, most recent) snapshot per
    `start` offset. Deep pagination pages were crawled less recently than
    the first few, so snapshot dates legitimately vary page to page -- we
    always take the freshest available capture for each specific page."""
    from urllib.parse import urlparse, parse_qs

    url = f"{CDX_API}?url=shl.com/solutions/products/product-catalog*&output=json&filter=statuscode:200&limit=20000"
    rows = fetch_json(url)
    if not rows:
        return {}
    header, data = rows[0], rows[1:]
    idx = {name: i for i, name in enumerate(header)}

    best: dict[str, tuple[str, str]] = {}
    for row in data:
        original = row[idx["original"]]
        if "/view/" in original:
            continue
        qs = parse_qs(urlparse(original).query)
        type_vals = qs.get("type", [])
        if not type_vals or type_vals[-1] != "1":
            continue
        start = qs.get("start", ["0"])[-1]
        timestamp = row[idx["timestamp"]]
        if start not in best or timestamp > best[start][0]:
            best[start] = (timestamp, original)
    return best


def find_closest_snapshot(target_url: str, prefer_after: str = "20230101") -> str | None:
    """CDX lookup for a single URL (used for detail pages); prefers the
    most recent snapshot, falling back to the closest one available at all."""
    url = f"{CDX_API}?url={target_url}&output=json&filter=statuscode:200&collapse=timestamp:8&limit=50"
    try:
        rows = fetch_json(url)
    except Exception:
        return None
    if not rows or len(rows) < 2:
        return None
    header, data = rows[0], rows[1:]
    idx = header.index("timestamp")
    timestamps = sorted((row[idx] for row in data), reverse=True)
    for ts in timestamps:
        if ts >= prefer_after:
            return ts
    return timestamps[0] if timestamps else None


def wayback_url(target_url: str, timestamp: str) -> str:
    return f"https://web.archive.org/web/{timestamp}/{target_url}"


def strip_wayback_prefix(href: str) -> str:
    """Turn '/web/20250316060545/https://www.shl.com/...' into 'https://www.shl.com/...'."""
    match = re.search(r"https?://www\.shl\.com/\S+", href)
    if not match:
        return href
    url = match.group(0)
    return url.split('"')[0]


ROW_RE = re.compile(
    r'<tr data-entity-id="(?P<id>\d+)">\s*'
    r'<td class="custom__table-heading__title">\s*'
    r'<a href="(?P<href>[^"]+)">(?P<name>[^<]+)</a>\s*</td>\s*'
    r'<td class="custom__table-heading__general">\s*'
    r'(?:<span class="catalogue__circle (?P<remote>-yes|-no)"></span>)?\s*</td>\s*'
    r'<td class="custom__table-heading__general">\s*'
    r'(?:<span class="catalogue__circle (?P<adaptive>-yes|-no)"></span>)?\s*</td>\s*'
    r'<td class="custom__table-heading__general product-catalogue__keys">'
    r'(?P<types>.*?)</td>\s*</tr>',
    re.DOTALL,
)

TYPE_KEY_RE = re.compile(r'product-catalogue__key[^>]*>([A-Z])<')


TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL)
TABLE_HEADER_RE = re.compile(r'<th class="custom__table-heading__title">([^<]+)</th>')


def extract_individual_test_table(html: str) -> str:
    """Isolate only the 'Individual Test Solutions' table, ignoring any
    'Pre-packaged Job Solutions' table that may appear on the same page."""
    for table_match in TABLE_RE.finditer(html):
        segment = table_match.group(0)
        header_match = TABLE_HEADER_RE.search(segment)
        if header_match and "Individual Test Solutions" in header_match.group(1):
            return segment
    return ""


def parse_listing_page(html: str) -> list[dict]:
    table_html = extract_individual_test_table(html)
    if not table_html:
        return []
    rows = []
    for m in ROW_RE.finditer(table_html):
        type_codes = TYPE_KEY_RE.findall(m.group("types"))
        rows.append(
            {
                "entity_id": m.group("id"),
                "name": unescape(m.group("name")).strip(),
                "url": strip_wayback_prefix(m.group("href")),
                "remote": m.group("remote") == "-yes",
                "adaptive": m.group("adaptive") == "-yes",
                "test_types": [
                    {"code": c, "name": TEST_TYPE_NAMES.get(c, c)} for c in type_codes
                ],
            }
        )
    return rows


EXACT_BASE_SNAPSHOT = "20250430003713"  # known-good full crawl of the bare listing URL


def scrape_listing(test_type: int) -> list[dict]:
    print("Resolving archived listing-page snapshots via CDX...", file=sys.stderr)
    snapshot_map = build_listing_snapshot_map()
    print(f"  found {len(snapshot_map)} archived pagination offsets", file=sys.stderr)

    all_rows = []
    seen_ids = set()
    start = 0
    empty_pages_in_a_row = 0
    while True:
        print(f"  listing page start={start} type={test_type} ...", file=sys.stderr)
        try:
            if start == 0:
                # The bare URL (no query string) renders the first page of the
                # Individual Test Solutions table server-side by default; the
                # ?start=0&type=1 query variant was never itself crawled.
                html = fetch(f"https://web.archive.org/web/{EXACT_BASE_SNAPSHOT}/{BASE_LISTING}")
            else:
                entry = snapshot_map.get(str(start))
                if entry is None:
                    print(f"    no archived snapshot for start={start}, stopping", file=sys.stderr)
                    break
                timestamp, original_url = entry
                html = fetch(wayback_url(original_url, timestamp))
        except RuntimeError as e:
            print(f"    fetch failed, stopping pagination: {e}", file=sys.stderr)
            break
        rows = parse_listing_page(html)
        new_rows = [r for r in rows if r["entity_id"] not in seen_ids]
        if not new_rows:
            empty_pages_in_a_row += 1
            if empty_pages_in_a_row >= 2:
                break
        else:
            empty_pages_in_a_row = 0
            for r in new_rows:
                seen_ids.add(r["entity_id"])
                all_rows.append(r)
        start += 12
        time.sleep(REQUEST_DELAY_SECONDS)
        if start > 800:  # safety cap
            break
    return all_rows


DESC_RE = re.compile(
    r'<h[0-9][^>]*>\s*Description\s*</h[0-9]>\s*<p>(?P<desc>.*?)</p>', re.DOTALL | re.IGNORECASE
)
JOB_LEVELS_RE = re.compile(
    r'<h[0-9][^>]*>\s*Job level[s]?\s*</h[0-9]>\s*<p>(?P<levels>.*?)</p>', re.DOTALL | re.IGNORECASE
)
DURATION_RE = re.compile(
    r'<h[0-9][^>]*>\s*Assessment length\s*</h[0-9]>\s*<p>[^<]*?=\s*(?P<minutes>\d+)',
    re.DOTALL | re.IGNORECASE,
)
LANGUAGES_RE = re.compile(
    r'<h[0-9][^>]*>\s*Languages?\s*</h[0-9]>\s*<p>(?P<langs>.*?)</p>', re.DOTALL | re.IGNORECASE
)


def clean_html_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_detail_page(html: str) -> dict:
    detail = {"description": "", "job_levels": [], "duration_minutes": None, "languages": []}

    m = DESC_RE.search(html)
    if m:
        detail["description"] = clean_html_text(m.group("desc"))

    m = JOB_LEVELS_RE.search(html)
    if m:
        levels_text = clean_html_text(m.group("levels"))
        detail["job_levels"] = [lvl.strip() for lvl in levels_text.split(",") if lvl.strip()]

    m = DURATION_RE.search(html)
    if m:
        detail["duration_minutes"] = int(m.group("minutes"))

    m = LANGUAGES_RE.search(html)
    if m:
        langs_text = clean_html_text(m.group("langs"))
        detail["languages"] = [l.strip() for l in langs_text.split(",") if l.strip()]

    return detail


def fetch_detail_html(detail_url: str) -> str:
    # Fast path: flexible-date lookup (one request) works for most detail
    # pages since they're crawled far more consistently than deep listing
    # pagination. Fall back to an explicit CDX lookup if that 404s.
    try:
        return fetch(f"https://web.archive.org/web/20250430/{detail_url}")
    except RuntimeError:
        timestamp = find_closest_snapshot(detail_url)
        if not timestamp:
            raise RuntimeError(f"no wayback snapshot found for {detail_url}")
        return fetch(wayback_url(detail_url, timestamp))


def scrape_detail(row: dict) -> dict:
    cache_file = DETAIL_DIR / f"{row['entity_id']}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    try:
        html = fetch_detail_html(row["url"])
        detail = parse_detail_page(html)
    except RuntimeError as e:
        print(f"    detail fetch failed for {row['url']}: {e}", file=sys.stderr)
        detail = {"description": "", "job_levels": [], "duration_minutes": None, "languages": []}

    merged = {**row, **detail}
    cache_file.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    time.sleep(REQUEST_DELAY_SECONDS)
    return merged


def build_search_text(item: dict) -> str:
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("job_levels", [])),
        " ".join(t["name"] for t in item.get("test_types", [])),
        " ".join(t["code"] for t in item.get("test_types", [])),
        " ".join(item.get("languages", [])),
        "remote" if item.get("remote") else "",
        "adaptive irt" if item.get("adaptive") else "",
    ]
    return " ".join(p for p in parts if p).strip()


def main():
    print("Scraping Individual Test Solutions (type=1) listing pages...", file=sys.stderr)
    listing_rows = scrape_listing(test_type=1)
    print(f"Found {len(listing_rows)} individual test solution rows.", file=sys.stderr)

    (RAW_DIR / "catalog_listing.json").write_text(
        json.dumps(listing_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Fetching detail pages (parallel)...", file=sys.stderr)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    full_catalog = []
    done = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(scrape_detail, row): row for row in listing_rows}
        for future in as_completed(futures):
            row = futures[future]
            done += 1
            try:
                merged = future.result()
            except Exception as e:
                print(f"  [{done}/{len(listing_rows)}] FAILED {row['name']}: {e}", file=sys.stderr)
                merged = {**row, "description": "", "job_levels": [], "duration_minutes": None, "languages": []}
            else:
                print(f"  [{done}/{len(listing_rows)}] {row['name']}", file=sys.stderr)
            merged["search_text"] = build_search_text(merged)
            full_catalog.append(merged)

    full_catalog.sort(key=lambda x: x["name"].lower())

    (PROCESSED_DIR / "catalog.json").write_text(
        json.dumps(full_catalog, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(full_catalog)} assessments to data/processed/catalog.json", file=sys.stderr)


if __name__ == "__main__":
    main()
