import re
from io import StringIO
import pandas as pd
import requests

# State mapping from full name to 2-letter postal code
STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "Y", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


def parse_year(date_str) -> int:
    """Extract a 4-digit year from a date string."""
    if not isinstance(date_str, str):
        return None
    match = re.search(r"\b(19\d{2}|20\d{2})\b", date_str)
    return int(match.group(1)) if match else None


def identify_cause(cause_str) -> str:
    """Categorize the cause of failure using keyword matching."""
    if not isinstance(cause_str, str):
        return "unknown"
    
    cause_lower = cause_str.lower()
    
    # Collision (check first to avoid matching water body names)
    if any(k in cause_lower for k in ["collision", "barge", "ship", "boat", "struck", "truck", "impact", "vehicle"]):
        return "collision"
    # Hydraulic/Scour
    if any(k in cause_lower for k in ["scour", "flood", "hydraulic", "washout", "erosion", "undermin"]):
        return "scour"
    # Overload
    if any(k in cause_lower for k in ["overload", "weight", "heavy", "overweight"]):
        return "overload"
    # Deterioration
    if any(k in cause_lower for k in ["corrosion", "rust", "deterioration", "maintenance", "decay", "poor condition", "structural"]):
        return "deterioration"
    # Fracture Critical
    if any(k in cause_lower for k in ["fatigue", "crack", "fracture", "weld", "member", "redundancy"]):
        return "fracture_critical"
    # Fire
    if any(k in cause_lower for k in ["fire", "explosion", "burn"]):
        return "fire"
    # Extreme Weather
    if any(k in cause_lower for k in ["earthquake", "seismic", "tornado", "hurricane", "wind", "storm", "landslide", "lightning"]):
        return "extreme_weather"
    # Construction
    if any(k in cause_lower for k in ["construction", "design", "erection", "falsework"]):
        return "construction"
        
    return "misc"


def detect_state(location_str) -> str:
    """Identify the US state code from a location string."""
    if not isinstance(location_str, str):
        return None
    
    # Check for direct state name match
    for state_name, code in STATES.items():
        if state_name.lower() in location_str.lower():
            return code
            
    # Check for 2-letter state code boundaries, e.g. ", CA" or " CA "
    for code in STATES.values():
        if re.search(r"\b" + code + r"\b", location_str):
            return code
            
    return None


def scrape_wikipedia_failures() -> pd.DataFrame:
    """
    Scrapes Wikipedia's 'List of bridge failures' page, filters for US incidents,
    maps causes, and formats the output ready for pre-failure mapping.
    """
    url = "https://en.wikipedia.org/wiki/List_of_bridge_failures"
    print(f"[INFO] Fetching Wikipedia bridge failures: {url}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        # read_html parses all tables on the page
        tables = pd.read_html(StringIO(response.text))
    except Exception as e:
        print(f"[ERROR] Failed to scrape Wikipedia: {e}")
        return pd.DataFrame()

    all_bridges = []
    
    for table in tables:
        # Wikipedia table headers are usually: Bridge, Location, Date, Cause, etc.
        cols = [str(c).lower() for c in table.columns]
        if "bridge" in cols and "location" in cols:
            table.columns = cols
            all_bridges.append(table)

    if not all_bridges:
        print("[WARN] No tables matching NBI criteria were parsed from Wikipedia.")
        return pd.DataFrame()

    df = pd.concat(all_bridges, ignore_index=True)
    print(f"[INFO] Found {len(df)} total bridge failures across Wikipedia tables.")

    cleaned_incidents = []
    for _, row in df.iterrows():
        location = row.get("location")
        state_code = detect_state(location)
        
        # We only care about US bridges
        if not state_code:
            continue
            
        bridge_name = str(row.get("bridge", "")).split("[")[0].strip()
        date_str = str(row.get("date", ""))
        year_failed = parse_year(date_str)
        
        if not year_failed:
            continue
            
        cause_desc = str(row.get("reason", ""))
        cause_category = identify_cause(cause_desc)
        
        # Calculate pre-failure NBI year (1 year prior to failure)
        prefailure_year = year_failed - 1
        nbi_available = "yes" if prefailure_year >= 1996 else "no"
        download_url = f"https://www.fhwa.dot.gov/bridge/nbi/ascii{prefailure_year}.cfm" if prefailure_year >= 1996 else ""
        
        # Extract losses (casualties)
        losses = str(row.get("casualties", ""))
        fatalities = 0
        injuries = ""
        
        match_fat = re.search(r"(\d+)\s*(?:dead|killed|fatalit|death)", losses.lower())
        if match_fat:
            fatalities = int(match_fat.group(1))
        elif "none" in losses.lower() or "no " in losses.lower() or "0" in losses.lower():
            fatalities = 0
            
        match_inj = re.search(r"(\d+)\s*(?:injur|wound)", losses.lower())
        if match_inj:
            injuries = int(match_inj.group(1))
            
        cleaned_incidents.append({
            "bridge_name": bridge_name,
            "state": state_code,
            "year_failed": year_failed,
            "cause": cause_category,
            "location_text": f"{bridge_name} {location}".split("[")[0].strip(),
            "prefailure_nbi_year": prefailure_year,
            "nbi_data_available": nbi_available,
            "download_url": download_url,
            "fatalities": fatalities,
            "injuries": injuries,
            "damage_cost_usd": "",
            "notes": f"Scraped from Wikipedia. Original cause: {cause_desc}"
        })

    result_df = pd.DataFrame(cleaned_incidents)
    # Remove duplicates
    result_df = result_df.drop_duplicates(subset=["bridge_name", "state", "year_failed"])
    print(f"[OK] Extracted {len(result_df)} US-based bridge failures.")
    return result_df


if __name__ == "__main__":
    df = scrape_wikipedia_failures()
    if not df.empty:
        df.to_csv("labels/scraped_failures.csv", index=False)
        print("[OK] Saved scraped failures to labels/scraped_failures.csv")
