"""
Maximize NYDOT Data Usage:
Match every NYDOT BIN record to its pre-failure NBI parquet year by bridge_key,
map cause codes to ML categories, and append all matches as confirmed labeled training examples.

NYDOT cause codes:
  pc  = partial collapse (deterioration/overload)
  tc  = total collapse   (deterioration/scour depending on scour_flag)
  scour = scour failure

This replaces the current approach where NYDOT records were added as labeled_bridges rows
WITHOUT verifying the bridge_key match to the NBI. Now we directly confirm the link.
"""
import os
import sys
import pandas as pd
import glob
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CAUSE_MAP = {
    "pc": "deterioration",   # partial collapse = structural deterioration
    "tc": "deterioration",   # total collapse   = severe structural deterioration
    "scour": "scour",
}

def bin_to_bridge_key(bin_str, state_code="36"):
    """Convert a NYSDOT BIN to the NBI bridge_key format."""
    bin_str = str(bin_str).strip()
    padded = bin_str.zfill(15)
    return f"{state_code}-{padded}"

def maximize_nydot():
    nydot_path = "labels/nydot_failures.csv"
    labeled_path = "data/processed/labeled_bridges.parquet"
    processed_dir = "data/processed"

    nydot = pd.read_csv(nydot_path)
    print(f"Loaded {len(nydot)} NYDOT records")

    # Build bridge_key from BIN
    nydot["bin_num"] = nydot["bridge_name"].str.replace("BIN ", "", regex=False).str.strip()
    nydot["bridge_key"] = nydot["bin_num"].apply(lambda b: bin_to_bridge_key(b, "36"))

    # Map cause codes to ML categories
    nydot["ml_cause"] = nydot["cause"].str.strip().str.lower().map(CAUSE_MAP).fillna("other")

    # Load index of all processed parquets
    files = glob.glob(os.path.join(processed_dir, "bridges_clean_*.parquet"))
    year_map = {}
    for f in files:
        m = re.search(r"bridges_clean_(\d{4})\.parquet", f)
        if m:
            year_map[int(m.group(1))] = f

    # Load existing labeled data to avoid duplicates
    if os.path.exists(labeled_path):
        existing = pd.read_parquet(labeled_path)
        existing_keys = set(existing["bridge_key"].dropna().unique())
    else:
        existing = pd.DataFrame()
        existing_keys = set()

    print(f"Existing bridge_keys in labeled_bridges: {len(existing_keys)}")

    matched_rows = []
    unmatched = []

    for _, row in nydot.iterrows():
        bridge_key = row["bridge_key"]
        prefailure_year = row["prefailure_nbi_year"]

        # Skip if already in labeled data
        if bridge_key in existing_keys:
            continue

        # Try to find in prefailure year first, then adjacent years
        search_years = [prefailure_year, prefailure_year - 1, prefailure_year + 1]
        matched = False

        for yr in search_years:
            if yr not in year_map:
                continue
            df_yr = pd.read_parquet(
                year_map[yr],
                columns=["bridge_key", "state_code", "facility_carried", "features_intersected",
                         "lowest_major_rating", "scour_flag", "year_built"]
            )
            hit = df_yr[df_yr["bridge_key"] == bridge_key]
            if not hit.empty:
                h = hit.iloc[0]
                # Override cause: if scour_flag is set, prefer scour over pc/tc
                ml_cause = row["ml_cause"]
                if h["scour_flag"] == 1 and ml_cause in ("deterioration", "other"):
                    ml_cause = "scour"

                matched_rows.append({
                    "bridge_name": row["bridge_name"],
                    "state": row["state"],
                    "year_failed": int(row["year_failed"]),
                    "cause": ml_cause,
                    "location_text": row["location_text"],
                    "prefailure_nbi_year": int(yr),
                    "nbi_data_available": "yes",
                    "download_url": "",
                    "fatalities": int(row["fatalities"]),
                    "injuries": int(row["injuries"]),
                    "damage_cost_usd": None,
                    "notes": "Ingested from NYDOT/Scour Database (NBI key matched)",
                    "state_code": "36",
                    "bridge_key": bridge_key,
                    "match_score": 100.0,
                })
                matched = True
                break

        if not matched:
            unmatched.append(bridge_key)

    print(f"\nMatched {len(matched_rows)} new NYDOT bridges to NBI records")
    print(f"Unmatched (no NBI record found): {len(unmatched)}")

    if matched_rows:
        df_new = pd.DataFrame(matched_rows)
        if not existing.empty:
            df_combined = pd.concat([existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_parquet(labeled_path, index=False)
        print(f"\n[OK] Total labels now: {len(df_combined)}")
        print("\nNew cause distribution from NYDOT matches:")
        print(df_new["cause"].value_counts())
    else:
        print("\n[INFO] No new NYDOT matches to add.")


if __name__ == "__main__":
    maximize_nydot()
