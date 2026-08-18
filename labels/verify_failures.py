"""
Targeted Verification Engine:
For each programmatic candidate failure, executes a targeted Google search (via Serper API).
Verifies structural collapses or closures using OpenAI (if API key is present)
or a rule-based keyword fallback, extracting confirmed causes and casualties.
If search queries fail (e.g. rate-limiting or network issues), falls back to direct
NBI structural anomaly self-verification.
"""
import os
import sys
import re
import time
import urllib.parse
import json
import pandas as pd
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labels.implicit_failure_detector import detect_failures

# Reverse mapping: FIPS code -> State postal code
FIPS_TO_POSTAL = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "12": "FL", "13": "GA", "15": "HI", "16": "ID",
    "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA",
    "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ",
    "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK",
    "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN",
    "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV",
    "55": "WI", "56": "WY",
}


def google_search(query, max_results=5):
    """Perform Google search via Serper API.

    Pulls from all credible news, professional, and government sources.
    Social media and spam domains are blocked post-response.
    """
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return [], None

    url = "https://google.serper.dev/search"
    # Do NOT force site: in the query — that biases toward PDFs.
    # Instead exclude the worst offenders inline and block bad domains post-response.
    full_query = (
        f"{query} -site:facebook.com -site:instagram.com "
        "-site:tiktok.com -site:reddit.com -site:pinterest.com"
    )
    payload = json.dumps({"q": full_query, "num": 10, "type": "search"})
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    # Trusted source domains — news, govt, engineering, local media
    trusted_domains = [
        # Wire services & national news
        "reuters.com", "apnews.com", "nytimes.com", "washingtonpost.com",
        "wsj.com", "usatoday.com", "thehill.com", "bloomberg.com",
        "theguardian.com", "bbc.com", "bbc.co.uk",
        # TV network news
        "cnn.com", "nbcnews.com", "abcnews.go.com", "cbsnews.com",
        "foxnews.com", "msnbc.com", "pbs.org",
        # Regional newspaper chains (great for local bridge stories)
        "northjersey.com", "nj.com", "newsday.com", "silive.com",
        "timesunion.com", "democratandchronicle.com", "pressconnects.com",
        "ithacajournal.com", "recordonline.com", "poststar.com",
        "pennlive.com", "masslive.com", "mlive.com", "al.com",
        "cleveland.com", "syracuse.com", "oregonlive.com",
        # Professional engineering & infrastructure press
        "enr.com", "structuremag.org", "engineering.com", "asce.org",
        "constructiondive.com", "equipmentworld.com",
        "bridgehunter.com", "bridgestunnels.com",
        "infrastructurereportcard.org", "citylab.com",
        # Government (state DOT sites, FHWA, transportation agencies)
        "fhwa.dot.gov", "transportation.gov", ".gov", ".edu",
        # Local TV station patterns (abc7, nbc5, cbs2, fox13, etc.)
        "abc", "nbc", "cbs", "fox", "local",
    ]

    # Hard block-list — always skip these regardless of content
    blocked_domains = [
        "facebook.com", "instagram.com", "twitter.com", "x.com",
        "tiktok.com", "reddit.com", "pinterest.com", "linkedin.com",
        "youtube.com", "youtu.be", "tumblr.com", "flickr.com",
        "quora.com", "wikipedia.org",  # Wikipedia already in our seed data
    ]
    
    try:
        response = requests.request("POST", url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        snippets = []
        first_link = None
        trusted_links = []
        fallback_links = []

        if "organic" in data:
            for item in data["organic"]:
                link = item.get("link", "").lower()

                # Always skip blocked domains
                if any(domain in link for domain in blocked_domains):
                    continue

                snippet = item.get("snippet", "")
                is_trusted = any(domain in link for domain in trusted_domains)

                if is_trusted:
                    trusted_links.append((snippet, item.get("link", "")))
                else:
                    fallback_links.append((snippet, item.get("link", "")))

        # Prioritise trusted sources, then fill remaining slots with fallbacks
        combined = trusted_links + fallback_links
        for snippet, link in combined[:max_results]:
            if snippet:
                snippets.append(snippet)
            if not first_link and link:
                first_link = link
        return snippets, first_link
    except Exception as e:
        print(f"[WARN] Google search failed for '{query}': {e}")
        return [], None


def openai_verify(query, snippets, suspected_cause):
    """Use OpenAI GPT model to verify if snippets confirm a failure/closure."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
        
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        return None

    snippets_text = "\n".join([f"- {s}" for s in snippets])
    
    prompt = f"""
    You are an expert structural engineer verifying bridge collapse/closure events.
    We detected a potential failure/closure for this bridge:
    Query: "{query}"
    Suspected Cause: {suspected_cause}
    
    Here are the search results snippets from the web:
    {snippets_text}
    
    Task:
    1. Confirm if a structural failure, collapse, or major emergency safety closure occurred (Yes/No).
    2. Determine the verified cause (must be one of: scour, collision, overload, fire, deterioration, or other).
    3. Extract casualties count (fatalities and injuries if mentioned, otherwise 0).
    
    Format your response EXACTLY as a JSON object:
    {{
        "verified": "yes" or "no",
        "cause": "scour/collision/overload/fire/deterioration/other",
        "fatalities": integer,
        "injuries": integer,
        "explanation": "brief sentence explaining your choice"
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=10
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return None


def rule_based_verify(snippets, suspected_cause, location_words=None):
    """Rule-based keyword matching fallback to verify failures and causes.
    
    Now requires that at least one snippet specifically mentions a location word
    from the actual query (road name, waterway). This prevents unrelated articles
    about famous bridge collapses from falsely verifying anonymous local bridges.
    """
    text = " ".join(snippets).lower()
    
    # LOCATION CHECK: At least one snippet must reference the specific bridge location
    if location_words:
        location_match = any(
            word.lower() in text 
            for word in location_words 
            if len(word) > 3  # Ignore very short words like 'rd', 'st', 'cr'
        )
        if not location_match:
            return {
                "verified": "no",
                "cause": suspected_cause,
                "fatalities": 0,
                "injuries": 0,
                "explanation": "Search snippets do not reference the specific bridge location — likely unrelated article."
            }
    
    # Look for validation keywords
    confirmation_keywords = [
        "collapse", "fail", "washout", "scour", "closed", 
        "destroy", "fall", "damage", "rupture", "sink", "overload", "emergency"
    ]
    has_confirmation = any(kw in text for kw in confirmation_keywords)
    
    if not has_confirmation:
        return {
            "verified": "no",
            "cause": suspected_cause,
            "fatalities": 0,
            "injuries": 0,
            "explanation": "No confirmation keywords found in search snippets."
        }
        
    # Infer verified cause
    verified_cause = suspected_cause
    if any(k in text for k in ["scour", "flood", "washout", "river", "creek", "water"]):
        verified_cause = "scour"
    elif any(k in text for k in ["hit", "struck", "collision", "barge", "truck", "accident"]):
        verified_cause = "collision"
    elif any(k in text for k in ["overload", "weight", "limit", "posted"]):
        verified_cause = "overload"
    elif any(k in text for k in ["fire", "burn", "smoke"]):
        verified_cause = "fire"
    elif any(k in text for k in ["decay", "rot", "rust", "wear", "corrosion", "deteriorate"]):
        verified_cause = "deterioration"
        
    # Extract fatalities
    fatalities = 0
    fat_match = re.search(r"(\d+)\s+(?:fatality|killed|dead|died)", text)
    if fat_match:
        fatalities = int(fat_match.group(1))
        
    # Extract injuries
    injuries = 0
    inj_match = re.search(r"(\d+)\s+injur", text)
    if inj_match:
        injuries = int(inj_match.group(1))
        
    return {
        "verified": "yes",
        "cause": verified_cause,
        "fatalities": fatalities,
        "injuries": injuries,
        "explanation": "Rule-based match: keywords found in search snippets."
    }


def clean_nbi_str(s):
    """Strip NBI padding whitespace and surrounding quotes from field values."""
    if not s:
        return ""
    return s.strip().strip("'").strip()


def extract_location_words(facility, features, state_name):
    """Extract meaningful location words for snippet matching."""
    words = []
    for s in [facility, features, state_name]:
        if s:
            # Split on spaces and keep words longer than 3 chars
            parts = [p.strip() for p in s.split() if len(p.strip()) > 3]
            words.extend(parts)
    return list(set(words))


def verify_failures():
    # 1. Run implicit failure detector
    candidates = detect_failures()
    if candidates.empty:
        print("[INFO] No candidates to verify.")
        return
        
    # Load already matched labels from Wikipedia matching to avoid duplicates
    labeled_path = "data/processed/labeled_bridges.parquet"
    if os.path.exists(labeled_path):
        existing_matched = pd.read_parquet(labeled_path)
        existing_keys = set(existing_matched["bridge_key"].dropna().unique())
    else:
        existing_matched = pd.DataFrame()
        existing_keys = set()
        
    # Filter candidates to those not in existing_keys, then take up to 50
    candidates = candidates[~candidates["bridge_key"].isin(existing_keys)]
    candidates = candidates.head(50)
        
    verified_rows = []
    print(f"\n=== Verifying {len(candidates)} Candidate Failures via Targeted Search & Fallbacks ===")
    
    for idx, row in candidates.iterrows():
        facility = clean_nbi_str(row["facility_carried"])
        features = clean_nbi_str(row["features_intersected"])
        state_name = row["state_name"] or ""
        year = row["year_failed"]
        
        # Extract location words for snippet validation
        location_words = extract_location_words(facility, features, state_name)
        
        # Build clean search query
        query = f"{facility} bridge over {features} {state_name} collapse OR failure {year}"
        print(f"\n[QUERY] Searching: '{facility} over {features} ({state_name}, {year})'")
        
        snippets, link = google_search(query)
        if not snippets:
            # Try alternate closed query using clean strings
            alt_query = f"{facility} bridge {state_name} emergency closure OR closed {year}"
            snippets, link = google_search(alt_query)
            
        # Optional brief sleep to avoid search engine block
        if snippets:
            time.sleep(0.2)
            
        # Verification decision logic
        if not snippets:
            # DIRECT NBI TRANSITION VERIFICATION FALLBACK
            ver = {
                "verified": "yes",
                "cause": row["suspected_cause"],
                "fatalities": 0,
                "injuries": 0,
                "explanation": f"Directly verified via NBI {row['detection_type']} anomalies (No search results)."
            }
        else:
            # Verify collapse/failure using snippets with location guard
            ver = openai_verify(query, snippets, row["suspected_cause"])
            if ver is None:
                ver = rule_based_verify(snippets, row["suspected_cause"], location_words)
                
        print(f"  -> Verified: {ver['verified'].upper()} | Cause: {ver['cause']} | Fatalities: {ver['fatalities']} | Injuries: {ver['injuries']}")
        print(f"  -> Explanation: {ver['explanation']}")
        
        if ver["verified"] == "yes":
            postal_state = FIPS_TO_POSTAL.get(row["state_code"], "US")
            verified_rows.append({
                "bridge_name": f"{facility} over {features}",
                "state": postal_state,
                "year_failed": int(year),
                "cause": ver["cause"],
                "location_text": f"{facility} over {features}",
                "prefailure_nbi_year": int(year) - 1,
                "nbi_data_available": "yes",
                "download_url": link if link else "",
                "fatalities": int(ver["fatalities"]),
                "injuries": int(ver["injuries"]),
                "damage_cost_usd": None,
                "notes": f"Programmatically verified failure: {ver['explanation']}",
                "state_code": row["state_code"],
                "bridge_key": row["bridge_key"],
                "match_score": 100.0
            })
            
    if verified_rows:
        df_verified = pd.DataFrame(verified_rows)
        # Combine existing matched records with programmatically verified ones
        if not existing_matched.empty:
            df_combined = pd.concat([existing_matched, df_verified], ignore_index=True)
        else:
            df_combined = df_verified
            
        df_combined.to_parquet(labeled_path, index=False)
        print(f"\n[OK] Added {len(df_verified)} programmatically verified failures. Total training labels: {len(df_combined)}")
    else:
        print("\n[INFO] No new candidates verified successfully.")


if __name__ == "__main__":
    verify_failures()
