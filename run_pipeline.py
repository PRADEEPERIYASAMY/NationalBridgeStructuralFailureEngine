"""
Run after all needed years have been ingested via run_ingest_year.py.
Matches Wikipedia labels, programmatically detects and verifies implicit
failures from yearly NBI transitions, trains per-category models on the combined
labeled dataset, and scores the 2025 snapshot.
"""
from labels.fuzzy_match_labels import match_labels
from labels.verify_failures import verify_failures
from labels.ingest_nydot_failures import ingest_nydot_failures
from labels.merge_failures import merge_failures
from models.train_models import run as train_and_score
from features.shortlist_candidates import shortlist

if __name__ == "__main__":
    print("=== Ingest external databases and merge failures ===")
    ingest_nydot_failures()
    merge_failures()

    print("\n=== Match failure labels to their pre-failure NBI snapshots ===")
    match_labels()

    print("\n=== Programmatically detect and verify implicit failures from NBI transitions ===")
    verify_failures()

    print("\n=== Train per-category models, rank drivers, score 2025 ===")
    train_and_score()

    print("\n=== Preliminary shortlist (condition rating filter) ===")
    shortlist(2025)

    print("\n=== Done. See data/driver_ranking_*.csv, data/bridge_risk_scores_2025.csv, data/shortlist_2025.csv ===")
