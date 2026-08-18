"""
Build the feature table for a given snapshot year -> bridge_features_{year}.
Call this once per year you loaded+cleaned (each prefailure year, plus 2025).
"""
import os
import sys
import duckdb

FEATURE_COLUMNS = [
    "scour_code", "scour_flag", "waterway_adequacy", "channel_cond",
    "deck_cond", "superstructure_cond", "substructure_cond", "culvert_cond",
    "lowest_major_rating", "bridge_age", "reconstruction_age",
    "operating_rating", "inventory_rating", "load_deficient_flag",
    "adt", "pct_truck_traffic",
    "fracture_critical_flag", "structure_kind", "structure_type", "design_load",
]


def build_features(year: int):
    clean_path = f"data/processed/bridges_clean_{year}.parquet"
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    feature_path = os.path.join(out_dir, f"bridge_features_{year}.parquet")

    con = duckdb.connect(':memory:')
    cols = ", ".join(FEATURE_COLUMNS)

    con.execute(f"""
        CREATE OR REPLACE TABLE feature_tmp AS
        SELECT bridge_key, state_code, snapshot_year, {cols}
        FROM read_parquet('{clean_path}')
    """)

    con.execute(f"COPY feature_tmp TO '{feature_path}' (FORMAT PARQUET)")
    n = con.execute("SELECT COUNT(*) FROM feature_tmp").fetchone()[0]
    con.close()

    print(f"[OK] bridge_features_{year} built: {n} rows, {len(FEATURE_COLUMNS)} features -> {feature_path}")


if __name__ == "__main__":
    build_features(int(sys.argv[1]))
