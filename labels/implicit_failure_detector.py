"""
Scan processed NBI Parquet files consecutively to detect structural failure candidates:
1. Inventory Disappearances (present in Year T with CRITICAL ratings + scour/load flags, missing in T+1)
2. Emergency Safety Closures (status changes from open to closed-for-safety 'K')
3. Sudden Condition Drops (lowest rating drops by >=4 points down to critical level <=2)
4. Scour Code Drops (scour_code drops from safe >=6 to critical <=3)

Thresholds are intentionally strict to avoid false positives from bridge re-keying,
demolition, or routine maintenance closures.
"""
import os
import glob
import re
import pandas as pd

FIPS_TO_STATE_NAME = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas", "06": "California",
    "08": "Colorado", "09": "Connecticut", "10": "Delaware", "12": "Florida", "13": "Georgia",
    "15": "Hawaii", "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa",
    "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine", "24": "Maryland",
    "25": "Massachusetts", "26": "Michigan", "27": "Minnesota", "28": "Mississippi",
    "29": "Missouri", "30": "Montana", "31": "Nebraska", "32": "Nevada", "33": "New Hampshire",
    "34": "New Jersey", "35": "New Mexico", "36": "New York", "37": "North Carolina",
    "38": "North Dakota", "39": "Ohio", "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania",
    "44": "Rhode Island", "45": "South Carolina", "46": "South Dakota", "47": "Tennessee",
    "48": "Texas", "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
    "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming",
}


def detect_failures(
    processed_dir="data/processed",
    disappearance_rating_thresh=3,   # Only flag critical (<=3) bridges, not just poor ones
    drop_start_thresh=6,             # Must start from satisfactory or above (>=6)
    drop_end_thresh=2,               # Must land at serious/critical (<=2)
    drop_size_thresh=4,              # Must drop at least 4 points
    scour_code_drop_safe=6,          # Scour was safe (>=6)
    scour_code_drop_critical=3       # Scour dropped to critical (<=3)
):
    # Find all bridges_clean_YYYY.parquet files
    pattern = os.path.join(processed_dir, "bridges_clean_*.parquet")
    files = glob.glob(pattern)
    
    # Extract years and sort
    year_files = []
    for f in files:
        match = re.search(r"bridges_clean_(\d{4})\.parquet", f)
        if match:
            year_files.append((int(match.group(1)), f))
            
    year_files = sorted(year_files, key=lambda x: x[0])
    if len(year_files) < 2:
        print("[INFO] Need at least 2 consecutive processed years to detect failures.")
        return pd.DataFrame()
        
    candidates = []
    cols = [
        "bridge_key", "state_code", "lowest_major_rating", "scour_flag", 
        "scour_code", "load_deficient_flag", "posting_status", 
        "facility_carried", "features_intersected", "year_built"
    ]
    
    for i in range(len(year_files) - 1):
        year_t, path_t = year_files[i]
        year_t1, path_t1 = year_files[i + 1]
        
        # We only compare if they are consecutive years (e.g. 2005 and 2006, or 2020 and 2021)
        if year_t1 - year_t > 1:
            # Skip gaps (non-consecutive years)
            continue
            
        print(f"[INFO] Detecting structural failures between {year_t} -> {year_t1}...")
        df_t = pd.read_parquet(path_t, columns=cols)
        df_t1 = pd.read_parquet(path_t1, columns=cols)
        
        # State overlap guard: Only match states that are loaded in BOTH years!
        valid_states = set(df_t["state_code"].dropna().unique()) & set(df_t1["state_code"].dropna().unique())
        df_t = df_t[df_t["state_code"].isin(valid_states)].copy()
        df_t1 = df_t1[df_t1["state_code"].isin(valid_states)].copy()
        
        if df_t.empty or df_t1.empty:
            continue
            
        # Merge on bridge_key
        merged = df_t.merge(df_t1, on="bridge_key", how="outer", suffixes=("_t", "_t1"))
        
        # YEAR_BUILT GUARD: Impossible for a bridge to fail before it was built.
        # Filter out records where year_built > year_t (data artifacts, re-keyed entries,
        # or NBI records attached to a road segment before the bridge was constructed).
        # This directly prevents cases like a bridge built in 1997 being flagged as failing in 1993.
        valid_built = (
            merged["year_built_t"].isna() |  # Keep if year_built unknown
            (merged["year_built_t"] <= year_t)  # Must have existed by comparison year
        )
        merged = merged[valid_built]
        
        # 1. Disappearances (active in T with CRITICAL rating + structural flag, missing in T1)
        # Strict: rating must be <=3 (critical) AND at least one structural flag must be set
        # This avoids flagging bridges that were re-keyed, demolished intentionally, or re-inventoried
        disappeared = merged[
            merged["state_code_t"].notna() & 
            merged["state_code_t1"].isna() & 
            (merged["lowest_major_rating_t"] <= disappearance_rating_thresh) &
            (merged["lowest_major_rating_t"] > 0) &  # Exclude already-dead (rating=0) bridges
            (
                (merged["scour_flag_t"] == 1) |
                (merged["load_deficient_flag_t"] == 1) |
                (merged["scour_code_t"].notna() & (merged["scour_code_t"] <= 3))
            )
        ]
        for _, row in disappeared.iterrows():
            candidates.append({
                "bridge_key": row["bridge_key"],
                "state_code": row["state_code_t"],
                "year_failed": year_t1,  # assumed failure year
                "facility_carried": row["facility_carried_t"],
                "features_intersected": row["features_intersected_t"],
                "lowest_rating_before": row["lowest_major_rating_t"],
                "detection_type": "disappearance",
                "suspected_cause": "scour" if row["scour_flag_t"] == 1 or (row["scour_code_t"] is not None and row["scour_code_t"] <= 3) else "deterioration"
            })
            
        # 2. Emergency Closures (status changes from open/restricted to closed-for-safety 'K')
        closed = merged[
            merged["state_code_t"].notna() & 
            merged["state_code_t1"].notna() & 
            (merged["posting_status_t"] != "K") & 
            (merged["posting_status_t1"] == "K")
        ]
        for _, row in closed.iterrows():
            candidates.append({
                "bridge_key": row["bridge_key"],
                "state_code": row["state_code_t"],
                "year_failed": year_t1,
                "facility_carried": row["facility_carried_t"],
                "features_intersected": row["features_intersected_t"],
                "lowest_rating_before": row["lowest_major_rating_t"],
                "detection_type": "emergency_closure",
                "suspected_cause": "scour" if row["scour_flag_t"] == 1 or (row["scour_code_t"] is not None and row["scour_code_t"] <= 3) else "overload" if row["load_deficient_flag_t"] == 1 else "deterioration"
            })
            
        # 3. Sudden Rating Drops — must be severe: from satisfactory (>=6) to serious/critical (<=2)
        # A 4+ point drop from 6->2 or 9->3 is much more credible than a 3-point 5->2 drop
        drops = merged[
            merged["state_code_t"].notna() & 
            merged["state_code_t1"].notna() & 
            (merged["lowest_major_rating_t"] >= drop_start_thresh) & 
            (merged["lowest_major_rating_t1"] <= drop_end_thresh) &
            (merged["lowest_major_rating_t"] - merged["lowest_major_rating_t1"] >= drop_size_thresh)
        ]
        for _, row in drops.iterrows():
            candidates.append({
                "bridge_key": row["bridge_key"],
                "state_code": row["state_code_t"],
                "year_failed": year_t1,
                "facility_carried": row["facility_carried_t"],
                "features_intersected": row["features_intersected_t"],
                "lowest_rating_before": row["lowest_major_rating_t"],
                "detection_type": "sudden_rating_drop",
                "suspected_cause": "scour" if row["scour_flag_t"] == 1 or (row["scour_code_t"] is not None and row["scour_code_t"] <= 3) else "deterioration"
            })
            
        # 4. Scour Code Drops — from safe (>=6) to critical (<=3)
        # This represents a genuine sudden scour vulnerability, not just routine re-inspection
        scour_drops = merged[
            merged["state_code_t"].notna() & 
            merged["state_code_t1"].notna() & 
            merged["scour_code_t1"].notna() &
            merged["scour_code_t"].notna() &
            (merged["scour_code_t1"] <= scour_code_drop_critical) &
            (merged["scour_code_t"] >= scour_code_drop_safe)
        ]
        for _, row in scour_drops.iterrows():
            candidates.append({
                "bridge_key": row["bridge_key"],
                "state_code": row["state_code_t"],
                "year_failed": year_t1,
                "facility_carried": row["facility_carried_t"],
                "features_intersected": row["features_intersected_t"],
                "lowest_rating_before": row["lowest_major_rating_t"],
                "detection_type": "scour_code_drop",
                "suspected_cause": "scour"
            })
            
    df_candidates = pd.DataFrame(candidates)
    if not df_candidates.empty:
        # Map state names for better query generation
        df_candidates["state_name"] = df_candidates["state_code"].map(FIPS_TO_STATE_NAME)
        # Drop duplicates in case a bridge triggered multiple rules
        df_candidates = df_candidates.drop_duplicates(subset=["bridge_key"])
        print(f"[OK] Detected {len(df_candidates)} failure candidates from NBI yearly changes.")
    else:
        print("[INFO] No failure candidates detected.")
        
    return df_candidates


if __name__ == "__main__":
    detect_failures()
