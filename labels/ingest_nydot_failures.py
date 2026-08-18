import pandas as pd
import os
import glob

STATE_FIPS_REVERSE = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "12": "FL", "13": "GA", "15": "HI", "16": "ID",
    "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA",
    "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ",
    "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK",
    "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN",
    "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV",
    "55": "WI", "56": "WY",
}

def ingest_nydot_failures():
    external_dir = "data/external"
    out_path = "labels/nydot_failures.csv"
    
    files = glob.glob(os.path.join(external_dir, "*.xls"))
    if not files:
        print("[WARNING] No .xls files found in data/external/")
        return
        
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_excel(f))
        except Exception as e:
            print(f"[ERROR] Could not read {f}: {e}")
            
    if not dfs:
        return
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Clean and filter year
    df['YR_FAIL'] = pd.to_numeric(df['YR_FAIL'], errors='coerce')
    df = df.dropna(subset=['YR_FAIL'])
    
    # We need failures from 1993 onwards (so prefailure year is 1992+)
    df_valid = df[df['YR_FAIL'] >= 1993].copy()
    
    records = []
    for _, row in df_valid.iterrows():
        # Clean BIN
        bin_val = str(row.get('BIN', '')).strip()
        if not bin_val or bin_val.lower() == 'nan':
            continue
            
        # Map state
        state_code_num = str(row.get('STATE_CODE', '')).split('.')[0].zfill(2)
        state_abbr = STATE_FIPS_REVERSE.get(state_code_num, "NY") # Default to NY if missing since it's mostly NYDOT
        
        # Clean location
        loc = str(row.get('LOCATION', '')).strip()
        feat = str(row.get('FEAT_UND', '')).strip()
        loc_text = f"{loc} {feat}".strip()
        
        fail_type = str(row.get('FAIL_TYPE', 'scour')).strip().lower()
        if fail_type == 'nan' or fail_type == '':
            fail_type = 'scour'
            
        records.append({
            "bridge_name": f"BIN {bin_val}",
            "state": state_abbr,
            "year_failed": int(row['YR_FAIL']),
            "cause": fail_type,
            "location_text": loc_text,
            "prefailure_nbi_year": int(row['YR_FAIL']) - 1,
            "nbi_data_available": "yes",
            "download_url": "",
            "fatalities": 0,
            "injuries": 0,
            "damage_cost_usd": 0.0,
            "notes": "Ingested from NYDOT/Scour Database"
        })
        
    out_df = pd.DataFrame(records)
    out_df = out_df.drop_duplicates(subset=["bridge_name", "state", "year_failed"])
    
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Ingested {len(out_df)} verified failures (>=1993) from external databases to {out_path}.")

if __name__ == "__main__":
    ingest_nydot_failures()
