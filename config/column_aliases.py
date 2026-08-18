# Maps our clean internal field names -> substrings that appear in the
# real FHWA NBI delimited file's header (item name + item number suffix).
# We match by substring because exact header spelling varies slightly by
# year/state export. Verify against your actual downloaded header once
# and add any missing aliases here.

ALIASES = {
    "state_code":        ["STATE_CODE_001"],
    "structure_number":  ["STRUCTURE_NUMBER_008"],
    "county_code":       ["COUNTY_CODE_003"],
    "latitude":          ["LAT_016", "LATITUDE"],
    "longitude":         ["LONG_017", "LONGITUD"],
    "year_built":        ["YEAR_BUILT_027"],
    "year_reconstructed":["YEAR_RECONSTRUCTED_106"],
    "deck_cond":         ["DECK_COND_058"],
    "superstructure_cond":["SUPERSTRUCTURE_COND_059"],
    "substructure_cond": ["SUBSTRUCTURE_COND_060"],
    "culvert_cond":      ["CULVERT_COND_062"],
    "channel_cond":      ["CHANNEL_COND_061"],
    "waterway_adequacy": ["WATERWAY_EVAL_071"],
    "scour_critical":    ["SCOUR_CRITICAL_113"],
    "operating_rating":  ["OPERATING_RATING_064"],
    "inventory_rating":  ["INVENTORY_RATING_066"],
    "design_load":       ["DESIGN_LOAD_031"],
    "posting_status":    ["OPEN_CLOSED_POSTED_041"],
    "adt":               ["ADT_029"],
    "adt_year":          ["YEAR_ADT_030"],
    "pct_truck_traffic": ["PERCENT_ADT_TRUCK_109"],
    "structure_kind":    ["STRUCTURE_KIND_043A"],
    "structure_type":    ["STRUCTURE_TYPE_043B"],
    "fracture_critical": ["FRACTURE_CRIT_DETAIL_092A", "FRACTURE_092A"],
    "facility_carried":  ["FACILITY_CARRIED_007"],
    "features_intersected": ["FEATURES_DESC_006A"],
}
