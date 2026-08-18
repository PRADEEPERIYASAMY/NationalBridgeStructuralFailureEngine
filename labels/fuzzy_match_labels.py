"""
For each failure incident, match it against the NBI snapshot from the
YEAR BEFORE it failed (prefailure_nbi_year), not against 2025. This
requires you to have run ingest/load_nbi.py + ingest/clean.py for each
distinct prefailure_nbi_year present in seed_failures.csv.

If a required year's Parquet file doesn't exist on disk yet, that incident is
skipped with a clear message -- it is NOT silently matched against the
wrong year.
"""
import os
import pandas as pd
from rapidfuzz import fuzz, process

STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "FL": "12", "GA": "13", "HI": "15", "ID": "16",
    "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22",
    "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34",
    "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40",
    "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47",
    "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56",
}


def match_labels(seed_csv="labels/seed_failures.csv", match_threshold=60):
    incidents = pd.read_csv(seed_csv)
    incidents["state_code"] = incidents["state"].map(STATE_FIPS)

    all_matches = []
    for prefailure_year, group in incidents.groupby("prefailure_nbi_year"):
        parquet_path = f"data/processed/bridges_clean_{prefailure_year}.parquet"
        if not os.path.exists(parquet_path):
            for _, row in group.iterrows():
                print(f"[SKIP] '{row['bridge_name']}' needs {parquet_path}, which hasn't been loaded yet.")
                all_matches.append({**row.to_dict(), "bridge_key": None, "match_score": None})
            continue

        # Load only 4 columns from the processed clean Parquet file to save memory/time
        bridges = pd.read_parquet(
            parquet_path,
            columns=["bridge_key", "state_code", "facility_carried", "features_intersected"]
        )

        for _, incident in group.iterrows():
            candidates = bridges[bridges["state_code"] == incident["state_code"]].copy()
            if candidates.empty:
                all_matches.append({**incident.to_dict(), "bridge_key": None, "match_score": 0})
                continue

            candidates["search_text"] = (
                candidates["facility_carried"].fillna("") + " " +
                candidates["features_intersected"].fillna("")
            )
            # Use combined query (bridge name + location text) for higher accuracy
            query = f"{incident['bridge_name']} {incident['location_text']}"
            best = process.extractOne(
                query, candidates["search_text"], scorer=fuzz.partial_token_set_ratio
            )
            
            # Map default threshold of 60 to 85 for partial_token_set_ratio
            effective_threshold = 85 if match_threshold == 60 else match_threshold
            
            if best and best[1] >= effective_threshold:
                matched_row = candidates.loc[best[2]]
                all_matches.append({**incident.to_dict(), "bridge_key": matched_row["bridge_key"], "match_score": best[1]})
            else:
                all_matches.append({**incident.to_dict(), "bridge_key": None, "match_score": best[1] if best else 0})

    result = pd.DataFrame(all_matches)
    matched = result[result["bridge_key"].notna()]
    print(f"\n[OK] Matched {len(matched)}/{len(result)} incidents total")
    print(result[["bridge_name", "prefailure_nbi_year", "cause", "match_score", "bridge_key"]])

    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "labeled_bridges.parquet")
    matched.to_parquet(out_path, index=False)
    print(f"[OK] Saved matched labeled bridges to {out_path}")
    return result


if __name__ == "__main__":
    match_labels()
