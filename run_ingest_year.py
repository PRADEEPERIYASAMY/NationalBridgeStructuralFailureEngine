"""
Multi-year pipeline runner.

Step A (repeat once per NBI year you have -- 2025 for scoring, plus every
distinct prefailure_nbi_year in labels/seed_failures.csv):
    python run_ingest_year.py data/raw/nbi_<year>.csv <year>

Step B (once, after all needed years are ingested+cleaned+featured):
    python run_pipeline.py
"""
import sys
from ingest.load_nbi import load_nbi
from ingest.clean import clean
from features.build_features import build_features


def ingest_one_year(raw_path: str, year: int):
    print(f"\n=== Ingesting {year} ===")
    load_nbi(raw_path, year)
    clean(year)
    build_features(year)


if __name__ == "__main__":
    ingest_one_year(sys.argv[1], int(sys.argv[2]))
