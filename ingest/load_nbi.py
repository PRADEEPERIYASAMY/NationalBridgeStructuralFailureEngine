import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.column_aliases import ALIASES
from ingest.xml_parser import parse_nbi_xml


def resolve_columns(columns):
    resolved = {}
    for canonical, patterns in ALIASES.items():
        match = None
        for pattern in patterns:
            for real_col in columns:
                if pattern.upper() in real_col.upper():
                    match = real_col
                    break
            if match:
                break
        resolved[canonical] = match
    return resolved


def load_nbi(raw_path: str, year: int, states_to_load: list = None):
    # Multi-format and directory handling
    if os.path.isdir(raw_path):
        files = []
        for f in os.listdir(raw_path):
            if f.lower().endswith(('.txt', '.csv', '.xml')):
                # Filter by state postal code (first 2 chars of file name)
                state_prefix = f[:2].upper()
                if states_to_load is None or state_prefix in states_to_load:
                    files.append(os.path.join(raw_path, f))
                    
        if not files:
            print(f"[WARN] No state files matched states {states_to_load} in {raw_path}")
            return None
            
        dfs = []
        for file in files:
            if file.lower().endswith(".xml"):
                dfs.append(parse_nbi_xml(file))
            else:
                dfs.append(pd.read_csv(file, low_memory=False, dtype=str))
        df = pd.concat(dfs, ignore_index=True)
    else:
        if raw_path.lower().endswith(".xml"):
            df = parse_nbi_xml(raw_path)
        else:
            df = pd.read_csv(raw_path, low_memory=False, dtype=str)

    colmap = resolve_columns(df.columns)

    # Validate that critical primary keys are present
    if colmap.get("state_code") is None or colmap.get("structure_number") is None:
        raise ValueError(
            f"Ingestion failed for year {year}: Could not resolve critical primary key columns "
            f"'state_code' (Item 1) or 'structure_number' (Item 8) from the input file. "
            f"Please verify column mappings. Available columns: {list(df.columns)[:10]}..."
        )

    missing = [k for k, v in colmap.items() if v is None]
    if missing:
        print(f"[WARN] year={year}: could not resolve {len(missing)} fields: {missing}")

    found = {k: v for k, v in colmap.items() if v is not None}
    clean = df[list(found.values())].rename(columns={v: k for k, v in found.items()})
    clean["snapshot_year"] = year

    # Save to raw Parquet file
    out_dir = "data/raw"
    os.makedirs(out_dir, exist_ok=True)
    parquet_path = os.path.join(out_dir, f"bridges_raw_{year}.parquet")
    clean.to_parquet(parquet_path, index=False)

    print(f"[OK] year={year}: loaded {len(clean)} rows, {len(found)}/{len(ALIASES)} fields resolved -> {parquet_path}")
    return colmap


if __name__ == "__main__":
    raw_path = sys.argv[1]
    year = int(sys.argv[2])
    load_nbi(raw_path, year)
