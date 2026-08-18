import os
import sys
import duckdb


def shortlist(year: int, rating_threshold: int = 4):
    feature_path = f"data/processed/bridge_features_{year}.parquet"
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    parquet_path = os.path.join(out_dir, f"shortlist_{year}.parquet")

    con = duckdb.connect(':memory:')

    con.execute(f"""
        CREATE OR REPLACE TABLE shortlist_tmp AS
        SELECT bridge_key, lowest_major_rating, scour_flag, load_deficient_flag, adt
        FROM read_parquet('{feature_path}')
        WHERE lowest_major_rating <= 3
           OR (lowest_major_rating <= {rating_threshold} AND (scour_flag = 1 OR load_deficient_flag = 1))
           OR (scour_code IS NOT NULL AND scour_code <= 2)
    """)

    total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{feature_path}')").fetchone()[0]
    shortlisted = con.execute("SELECT COUNT(*) FROM shortlist_tmp").fetchone()[0]
    pct = 100 * shortlisted / total if total else 0

    con.execute(f"COPY shortlist_tmp TO '{parquet_path}' (FORMAT PARQUET)")
    print(f"[OK] {shortlisted}/{total} bridges ({pct:.1f}%) shortlisted "
          f"(condition rating <= {rating_threshold}, or scour/load flagged) -> {parquet_path}")
    con.close()


if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    shortlist(year, threshold)
