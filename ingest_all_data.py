import os
import pandas as pd
from ingest.nbi_downloader import download_nbi_year
from ingest.load_nbi import load_nbi
from ingest.clean import clean
from features.build_features import build_features


def run_ingestion():
    # 1. Load seed failures to find what years and states we need
    failures = pd.read_csv("labels/seed_failures.csv")
    usable = failures[failures["nbi_data_available"].str.lower() == "yes"]
    
    # Group by prefailure_nbi_year and collect target states
    year_states = {}
    for year, group in usable.groupby("prefailure_nbi_year"):
        year_states[int(year)] = group["state"].unique().tolist()
        
    print(f"[INFO] Found {len(year_states)} historical pre-failure years to ingest.")
    
    # 2. Ingest scoring year (2025) first (all states)
    print("\n=== Processing Scoring Target: 2025 (All States) ===")
    raw_dir_2025 = "data/raw/nbi_2025"
    if not os.path.exists(raw_dir_2025) or not os.listdir(raw_dir_2025):
        raw_dir_2025 = download_nbi_year(2025)
    load_nbi(raw_dir_2025, 2025)
    clean(2025)
    build_features(2025)
    
    # 3. Ingest each historical pre-failure year (only relevant states)
    for year, states in sorted(year_states.items()):
        print(f"\n=== Processing Training Year: {year} (States: {states}) ===")
        raw_dir = f"data/raw/nbi_{year}"
        try:
            if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
                raw_dir = download_nbi_year(year)
            # Ingest only the specific states where failures occurred
            loaded_map = load_nbi(raw_dir, year, states_to_load=states)
            if loaded_map:
                clean(year)
                build_features(year)
        except Exception as e:
            print(f"[ERROR] Failed to process year {year}: {e}")
            
    print("\n=== Ingestion Completed Successfully ===")


if __name__ == "__main__":
    run_ingestion()
