"""
Generates fake NBI files for SEVERAL years (simulating 2025 + a few
pre-failure years) so we can test the multi-year pipeline logic before
using real downloads.
"""
import numpy as np
import pandas as pd

np.random.seed(42)


def make_year_file(year, n, path):
    states = ["01", "36", "48", "06", "13", "12", "47", "24", "42", "26"]
    df = pd.DataFrame({
        "STATE_CODE_001": np.random.choice(states, n),
        "STRUCTURE_NUMBER_008": [f"{i:07d}" for i in range(n)],
        "COUNTY_CODE_003": np.random.randint(1, 50, n),
        "LAT_016": np.random.uniform(25, 49, n),
        "LONG_017": np.random.uniform(-124, -70, n),
        "YEAR_BUILT_027": np.random.randint(1930, year, n),
        "YEAR_RECONSTRUCTED_106": np.random.choice([0] * 6 + list(range(1990, year)), n),
        "DECK_COND_058": np.random.randint(3, 9, n),
        "SUPERSTRUCTURE_COND_059": np.random.randint(3, 9, n),
        "SUBSTRUCTURE_COND_060": np.random.randint(3, 9, n),
        "CULVERT_COND_062": np.random.choice([0, 5, 6, 7, 8], n),
        "CHANNEL_COND_061": np.random.randint(2, 9, n),
        "WATERWAY_EVAL_071": np.random.randint(2, 9, n),
        "SCOUR_CRITICAL_113": np.random.choice(["N", "8", "5", "3", "2", "0"], n, p=[.3,.2,.2,.15,.1,.05]),
        "OPERATING_RATING_064": np.random.uniform(20, 80, n),
        "INVENTORY_RATING_066": np.random.uniform(15, 60, n),
        "DESIGN_LOAD_031": np.random.choice(["1", "3", "5", "6"], n),
        "OPEN_CLOSED_POSTED_041": np.random.choice(["A", "P", "R"], n, p=[.85, .1, .05]),
        "ADT_029": np.random.randint(50, 50000, n),
        "YEAR_ADT_030": np.random.randint(year - 3, year, n),
        "PERCENT_ADT_TRUCK_109": np.random.uniform(1, 25, n),
        "STRUCTURE_KIND_043A": np.random.choice(["1", "2", "3", "4", "5"], n),
        "STRUCTURE_TYPE_043B": np.random.choice(["02", "04", "10", "19"], n),
        "FRACTURE_CRIT_DETAIL_092A": np.random.choice(["N", "1", "2"], n, p=[.7, .2, .1]),
        # embed a real-ish location string for one bridge per state so fuzzy match has something to find
        "FEATURES_DESC_006A": [f"Creek near county {c}" for c in np.random.randint(1, 50, n)],
        "FACILITY_CARRIED_007": np.random.choice(["US 90", "SR 12", "I-40", "CR 4"], n),
    })
    # Inject one bridge per state with realistic location text matching a seed incident,
    # so the fuzzy matcher has a real target to find during testing.
    df.loc[0, "FEATURES_DESC_006A"] = "Mississippi River near Minneapolis"
    df.loc[0, "FACILITY_CARRIED_007"] = "Interstate 35W"
    df.loc[0, "STATE_CODE_001"] = "27"  # MN

    df.to_csv(path, index=False)
    print(f"Wrote {n} synthetic rows for year {year} -> {path}")


if __name__ == "__main__":
    make_year_file(2025, 1500, "data/raw/nbi_2025.csv")
    make_year_file(2006, 800, "data/raw/nbi_2006.csv")   # pre-failure year for I-35W (failed 2007)
    make_year_file(1986, 500, "data/raw/nbi_1986.csv")   # pre-failure year for Schoharie Creek (failed 1987)
