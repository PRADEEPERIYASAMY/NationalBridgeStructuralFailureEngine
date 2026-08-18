"""
Clean a single year's raw table -> bridges_clean_{year}.
'current_year' controls the age calc: for a 1986 snapshot, age should be
computed as of 1986, not today -- that's the bridge's age *at the time it
was inspected right before failing*, which is the actual signal we want.
"""
import os
import sys
import duckdb


def clean(year: int):
    raw_path = f"data/raw/bridges_raw_{year}.parquet"
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    clean_path = os.path.join(out_dir, f"bridges_clean_{year}.parquet")

    con = duckdb.connect(':memory:')

    con.execute(f"""
    CREATE OR REPLACE TABLE clean_tmp AS
    SELECT
        state_code,
        structure_number,
        state_code || '-' || structure_number AS bridge_key,
        {year} AS snapshot_year,
        TRY_CAST(latitude AS DOUBLE) AS latitude,
        TRY_CAST(longitude AS DOUBLE) AS longitude,
        TRY_CAST(year_built AS INTEGER) AS year_built,
        NULLIF(TRY_CAST(year_reconstructed AS INTEGER), 0) AS year_reconstructed,
        {year} - TRY_CAST(year_built AS INTEGER) AS bridge_age,
        COALESCE({year} - NULLIF(TRY_CAST(year_reconstructed AS INTEGER), 0), {year} - TRY_CAST(year_built AS INTEGER)) AS reconstruction_age,

        TRY_CAST(deck_cond AS INTEGER) AS deck_cond,
        TRY_CAST(superstructure_cond AS INTEGER) AS superstructure_cond,
        TRY_CAST(substructure_cond AS INTEGER) AS substructure_cond,
        TRY_CAST(culvert_cond AS INTEGER) AS culvert_cond,
        TRY_CAST(channel_cond AS INTEGER) AS channel_cond,
        TRY_CAST(waterway_adequacy AS INTEGER) AS waterway_adequacy,

        CASE WHEN scour_critical IN ('N', '') THEN NULL
             ELSE TRY_CAST(scour_critical AS INTEGER) END AS scour_code,
        CASE WHEN scour_critical IN ('0','1','2','3') THEN 1 ELSE 0 END AS scour_flag,

        TRY_CAST(operating_rating AS DOUBLE) AS operating_rating,
        TRY_CAST(inventory_rating AS DOUBLE) AS inventory_rating,
        design_load,
        posting_status,
        CASE WHEN posting_status IN ('P','R') THEN 1 ELSE 0 END AS load_deficient_flag,

        TRY_CAST(adt AS INTEGER) AS adt,
        TRY_CAST(pct_truck_traffic AS DOUBLE) AS pct_truck_traffic,

        structure_kind,
        structure_type,
        CASE WHEN fracture_critical IN ('N','') THEN 0 ELSE 1 END AS fracture_critical_flag,

        facility_carried,
        features_intersected,

        LEAST(
            COALESCE(TRY_CAST(deck_cond AS INTEGER), 9),
            COALESCE(TRY_CAST(superstructure_cond AS INTEGER), 9),
            COALESCE(TRY_CAST(substructure_cond AS INTEGER), 9),
            COALESCE(TRY_CAST(culvert_cond AS INTEGER), 9)
        ) AS lowest_major_rating

    FROM read_parquet('{raw_path}')
    """)

    con.execute(f"COPY clean_tmp TO '{clean_path}' (FORMAT PARQUET)")
    n = con.execute("SELECT COUNT(*) FROM clean_tmp").fetchone()[0]
    con.close()

    print(f"[OK] bridges_clean_{year} built with {n} rows -> {clean_path}")


if __name__ == "__main__":
    clean(int(sys.argv[1]))
