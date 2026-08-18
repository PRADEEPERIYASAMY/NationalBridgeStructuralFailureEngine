import pandas as pd

def merge_failures(seed_path="labels/seed_failures.csv", scraped_path="labels/scraped_failures.csv", nydot_path="labels/nydot_failures.csv"):
    seed = pd.read_csv(seed_path)
    scraped = pd.read_csv(scraped_path)
    
    dfs_to_concat = [seed, scraped]
    import os
    if os.path.exists(nydot_path):
        nydot = pd.read_csv(nydot_path)
        dfs_to_concat.append(nydot)
        
    # Combine datasets
    merged = pd.concat(dfs_to_concat, ignore_index=True)
    
    # Fill NaN notes
    merged["notes"] = merged["notes"].fillna("")
    
    # Smart classification of scour/hydraulic failures
    # If the notes contain keywords like "flood", "washout", or "scour", it belongs to the hydraulic/scour category
    mask_scour = (
        merged["notes"].str.lower().str.contains("flood|washout|scour|riverbed|undermin") |
        (merged["cause"] == "scour")
    )
    merged.loc[mask_scour, "cause"] = "scour"
    
    # Deduplicate keeping the first occurrence (usually the seed one, but now updated with cause)
    # Sort by cause so 'scour' is prioritized for deduplication
    merged["is_scour"] = (merged["cause"] == "scour").astype(int)
    merged = merged.sort_values(by="is_scour", ascending=False)
    merged = merged.drop_duplicates(subset=["bridge_name", "state", "year_failed"], keep="first")
    merged = merged.drop(columns=["is_scour"])
    
    # Sort by year_failed
    merged = merged.sort_values(by=["year_failed", "bridge_name"])
    
    # Save back to seed_failures.csv
    merged.to_csv(seed_path, index=False)
    print(f"[OK] Merged and updated {seed_path}. Total records: {len(merged)}")
    print("Cause distribution:")
    print(merged["cause"].value_counts())

if __name__ == "__main__":
    merge_failures()
