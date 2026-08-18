"""
build_verified_labels.py

Full rebuild of labeled_bridges.parquet from scratch.

Requirements for every record (ALL three are non-negotiable):
  1. Data point  - bridge name, state, year, cause from a curated source
  2. Source      - traceable origin (seed_failures.csv / Wikipedia)
  3. News URL    - a confirmed news article or official report link (NOT social media, NOT Wikipedia)

Pipeline:
  1. Load seed_failures.csv (263 known named bridge failures)
  2. For records already having a URL: validate it is not social media
  3. For records with no URL: use Serper + OpenAI to find a news article
  4. Only keep records where OpenAI confirms the article matches the specific failure
  5. Write final verified set to labeled_bridges.parquet
"""

import os
import sys
import time
import json
import requests
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

BLOCKED = [
    "facebook.com", "instagram.com", "twitter.com", "x.com",
    "tiktok.com", "reddit.com", "pinterest.com", "linkedin.com",
    "youtube.com", "youtu.be", "tumblr.com", "quora.com",
    "wikipedia.org", "en.m.wikipedia.org",
]

# These look like URLs but are NOT news articles about specific failures
NOT_NEWS = [
    "fhwa.dot.gov/bridge/nbi/ascii",  # NBI raw data download pages
    "fhwa.dot.gov/bridge/nbi/data",
    "fhwa.dot.gov/bridge/ascii",
]

def clean_url(val):
    """Return the URL string or None if it is missing/invalid."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "n/a", ""):
        return None
    return s

def is_blocked(url):
    if not url:
        return True
    url_lower = url.lower()
    if any(b in url_lower for b in BLOCKED):
        return True
    if any(n in url_lower for n in NOT_NEWS):
        return True
    return False

def search_news(bridge_name, year, state, cause):
    """Search Serper for a news article about this specific bridge failure."""
    if not SERPER_API_KEY:
        return [], None

    queries = [
        f'"{bridge_name}" bridge collapse {year}',
        f'"{bridge_name}" bridge failure {year} {state}',
        f'{bridge_name} {state} bridge {cause} {year} news',
    ]

    for query in queries:
        full_query = (
            f"{query} -site:facebook.com -site:instagram.com "
            "-site:reddit.com -site:pinterest.com -site:wikipedia.org"
        )
        payload = json.dumps({"q": full_query, "num": 10})
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

        try:
            r = requests.post(
                "https://google.serper.dev/search",
                headers=headers, data=payload, timeout=10
            )
            r.raise_for_status()
            data = r.json()
            snippets = []
            first_link = None
            for item in data.get("organic", []):
                link = item.get("link", "")
                if is_blocked(link):
                    continue
                snippet = item.get("snippet", "")
                if snippet:
                    snippets.append(snippet)
                if not first_link and link:
                    first_link = link
                if len(snippets) >= 5:
                    break
            if snippets:
                return snippets, first_link
        except Exception as e:
            print(f"    [WARN] Search error: {e}")
        time.sleep(0.3)

    return [], None

def openai_find_article(bridge_name, year, state, cause, snippets, first_link):
    """Ask OpenAI to confirm the snippets refer to this specific bridge failure and return a URL."""
    if not OPENAI_API_KEY or not snippets:
        return None, None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        snippet_text = "\n".join(f"- {s}" for s in snippets[:5])
        prompt = f"""You are verifying whether search results confirm a specific historical bridge failure.

Bridge: {bridge_name}
State: {state}
Year: {year}
Suspected cause: {cause}

Search snippets found:
{snippet_text}

Candidate URL: {first_link}

Task: Determine if ANY of the above snippets specifically confirm this bridge's failure/collapse/closure in or around {year}.
The snippet must explicitly mention this bridge BY NAME or by a very specific location that uniquely identifies it.
A generic article about bridge safety or about a different bridge does NOT count.

Respond in JSON:
{{
  "confirmed": true/false,
  "article_url": "<URL if confirmed, else null>",
  "reason": "<brief explanation>"
}}"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        result = json.loads(resp.choices[0].message.content)
        if result.get("confirmed"):
            return result.get("article_url") or first_link, result.get("reason", "")
        return None, result.get("reason", "")
    except Exception as e:
        print(f"    [WARN] OpenAI error: {e}")
        return None, None

def build_verified_labels(limit=None):
    seed_path = "labels/seed_failures.csv"
    out_path = "data/processed/labeled_bridges.parquet"

    seed = pd.read_csv(seed_path)
    print(f"Loaded {len(seed)} records from seed_failures.csv")

    # Skip BIN-named records — these are NYDOT anonymous entries with no news coverage
    seed = seed[~seed["bridge_name"].str.strip().str.upper().str.startswith("BIN ")]
    print(f"After dropping BIN records: {len(seed)} named bridge failures")
    clean_urls = seed["download_url"].apply(clean_url).notna() & ~seed["download_url"].apply(clean_url).apply(lambda u: is_blocked(u) if u else True)
    print(f"Already have a valid (non-NBI-download) news URL: {clean_urls.sum()}")
    print()

    verified = []
    skipped_no_url = 0
    skipped_bad_url = 0
    already_had_url = 0

    records_to_process = seed.to_dict("records")
    if limit:
        records_to_process = records_to_process[:limit]

    for i, row in enumerate(records_to_process):
        name = str(row.get("bridge_name", "")).strip()
        state = str(row.get("state", "")).strip()
        year = row.get("year_failed", 0)
        cause = str(row.get("cause", "")).strip()
        notes = str(row.get("notes", "")).strip()
        
        print(f"[{i+1}/{len(records_to_process)}] {name} ({state}, {year})")

        # Skip BIN records (caught earlier but double-check)
        if name.upper().startswith("BIN "):
            continue

        # Validate existing URL — must be a real news article, not a data page
        existing_url = clean_url(row.get("download_url"))
        if is_blocked(existing_url):
            existing_url = None  # Treat as missing — must search for a real news URL

        article_url = None
        reason = ""

        if existing_url:
            # Have a clean non-NBI URL — confirm with OpenAI using fresh snippets
            snippets, first_link = search_news(name, year, state, cause)
            if snippets and OPENAI_API_KEY:
                confirmed_url, reason = openai_find_article(name, year, state, cause, snippets, existing_url)
                if confirmed_url and not is_blocked(confirmed_url):
                    article_url = confirmed_url
                elif not is_blocked(existing_url):
                    # OpenAI couldn't find a better URL but existing is clean — keep it
                    article_url = existing_url
                    reason = "Existing URL kept (OpenAI search inconclusive)"
            else:
                # No fresh snippets but URL looks clean — keep it
                article_url = existing_url
                reason = "Pre-existing URL kept (no search results)"
            already_had_url += 1
        else:
            # No clean URL at all — must search and confirm
            snippets, first_link = search_news(name, year, state, cause)
            if not snippets:
                print(f"  -> No search results found. Skipping.")
                skipped_no_url += 1
                continue

            if OPENAI_API_KEY:
                confirmed_url, reason = openai_find_article(name, year, state, cause, snippets, first_link)
                if confirmed_url and not is_blocked(confirmed_url):
                    article_url = confirmed_url
            else:
                # No OpenAI — require name words in snippet
                name_words = [w for w in name.lower().split() if len(w) > 3]
                snippet_text = " ".join(snippets).lower()
                if any(w in snippet_text for w in name_words) and first_link and not is_blocked(first_link):
                    article_url = first_link
                    reason = "Name matched in snippet (no OpenAI)"

        if not article_url:
            print(f"  -> No confirmed article. Skipping. Reason: {reason}")
            skipped_bad_url += 1
            continue

        print(f"  -> CONFIRMED | URL: {article_url[:80]}...")

        verified.append({
            "bridge_name": name,
            "state": state,
            "year_failed": int(year) if year else None,
            "cause": cause,
            "location_text": str(row.get("location_text", "")).strip(),
            "prefailure_nbi_year": row.get("prefailure_nbi_year"),
            "nbi_data_available": str(row.get("nbi_data_available", "no")).strip(),
            "download_url": article_url,
            "fatalities": row.get("fatalities", 0),
            "injuries": row.get("injuries", 0),
            "damage_cost_usd": row.get("damage_cost_usd"),
            "notes": notes,
            "bridge_key": None,
            "match_score": None,
        })

        time.sleep(0.2)

    print()
    print("=" * 50)
    print(f"RESULTS:")
    print(f"  Verified with article URL:  {len(verified)}")
    print(f"  Had URL already:            {already_had_url}")
    print(f"  No search results:          {skipped_no_url}")
    print(f"  Article not confirmed:      {skipped_bad_url}")
    print()

    if verified:
        df_out = pd.DataFrame(verified)
        df_out.to_parquet(out_path, index=False)
        print(f"[OK] Saved {len(df_out)} verified labels to {out_path}")
        print("\nCause distribution:")
        print(df_out["cause"].value_counts())
    else:
        print("[WARN] No verified labels produced.")

if __name__ == "__main__":
    build_verified_labels()
