"""
Two distinct data sources, never mixed:
  - TRAINING features come from each incident's own pre-failure snapshot
    year (bridge_features_{prefailure_year}.parquet) -- this is what "about to fail"
    actually looked like.
  - SCORING features come from bridge_features_2025.parquet -- today's bridges,
    scored against the pattern the model learned.
"""
import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.build_features import FEATURE_COLUMNS

CATEGORIES = ["scour", "deterioration", "overload", "fracture_critical", "collision", "fire", "extreme_weather"]


def prep_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Converts categorical columns to category type and numeric to float/int."""
    cat_cols = ["structure_kind", "structure_type", "design_load"]
    X = df[feature_cols].copy()
    
    for col in feature_cols:
        if col in cat_cols:
            if not isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype(str).str.strip()
                X[col] = X[col].replace(["nan", "None", ""], None).astype("category")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            
    return X


def build_training_set():
    """Pull each labeled incident's features from ITS OWN prefailure year Parquet file."""
    labeled_path = "data/processed/labeled_bridges.parquet"
    if not os.path.exists(labeled_path):
        raise RuntimeError(f"Labeled bridges file not found at {labeled_path}. Run matching first.")
    labeled = pd.read_parquet(labeled_path)

    rows = []
    for year, group in labeled.groupby("prefailure_nbi_year"):
        feature_path = f"data/processed/bridge_features_{int(year)}.parquet"
        if not os.path.exists(feature_path):
            print(f"[SKIP] {feature_path} not found on disk.")
            continue
        features = pd.read_parquet(feature_path)
        merged = group.merge(features, on="bridge_key", how="inner")
        rows.append(merged)

    if not rows:
        raise RuntimeError("No training rows available -- no prefailure feature files were found on disk.")
    return pd.concat(rows, ignore_index=True)


def load_scoring_set(score_year=2025):
    """Load scoring features for the standing inventory from its feature Parquet file."""
    feature_path = f"data/processed/bridge_features_{score_year}.parquet"
    if not os.path.exists(feature_path):
        raise RuntimeError(f"Features file {feature_path} not found on disk.")
    df = pd.read_parquet(feature_path)
    return df


def train_one_category(train_df, category, feature_cols):
    y = (train_df["cause"] == category).astype(int)
    X = prep_features(train_df, feature_cols)

    if y.sum() < 5:
        print(f"[SKIP] '{category}': only {y.sum()} positive labels (need >=5). "
              f"Add more incidents to seed_failures.csv for this category.")
        return None, None

    n_splits = min(5, y.sum())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, praucs = [], []
    for train_idx, test_idx in skf.split(X, y):
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, 
            eval_metric="logloss", enable_categorical=True
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict_proba(X.iloc[test_idx])[:, 1]
        
        try:
            aucs.append(roc_auc_score(y.iloc[test_idx], preds))
        except ValueError:
            aucs.append(0.5)
            
        try:
            praucs.append(average_precision_score(y.iloc[test_idx], preds))
        except ValueError:
            praucs.append(y.iloc[test_idx].mean())

    print(f"[{category}] ROC-AUC={np.mean(aucs):.3f}  PR-AUC={np.mean(praucs):.3f}  (n_pos={y.sum()})")

    final_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, 
        eval_metric="logloss", enable_categorical=True
    )
    final_model.fit(X, y)
    return final_model, {"roc_auc": np.mean(aucs), "pr_auc": np.mean(praucs), "n_pos": int(y.sum())}


def rank_drivers(model, X, feature_cols, category, out_dir="data"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap})
    ranking = ranking.sort_values("mean_abs_shap", ascending=False)
    ranking["failure_category"] = category
    ranking.to_parquet(f"{out_dir}/driver_ranking_{category}.parquet", index=False)
    print(f"  top drivers for {category}: {ranking['feature'].head(3).tolist()}")
    return ranking


def score_2025(models, score_df, feature_cols, out_dir="data"):
    X = prep_features(score_df, feature_cols)
    scores = score_df[["bridge_key"]].copy()
    for category, model in models.items():
        scores[f"{category}_risk"] = model.predict_proba(X)[:, 1] if model is not None else np.nan
    scores.to_parquet(f"{out_dir}/bridge_risk_scores_2025.parquet", index=False)
    print(f"[OK] Scored {len(scores)} bridges (2025 snapshot) -> {out_dir}/bridge_risk_scores_2025.parquet")
    return scores


def run(score_year=2025):
    train_df = build_training_set()
    score_df = load_scoring_set(score_year)
    feature_cols = FEATURE_COLUMNS

    # Align CategoryDtype categories for training and scoring datasets
    cat_cols = ["structure_kind", "structure_type", "design_load"]
    for col in cat_cols:
        train_vals = train_df[col].dropna().astype(str).str.strip().unique()
        score_vals = score_df[col].dropna().astype(str).str.strip().unique()
        all_vals = sorted(list(set(train_vals) | set(score_vals) - {"", "nan", "None", "<NA>"}))
        
        cat_type = pd.CategoricalDtype(categories=all_vals)
        train_df[col] = train_df[col].astype(str).str.strip().replace(["nan", "None", "", "<NA>"], None).astype(cat_type)
        score_df[col] = score_df[col].astype(str).str.strip().replace(["nan", "None", "", "<NA>"], None).astype(cat_type)

    print(f"\nTraining set: {len(train_df)} labeled bridges across {train_df['prefailure_nbi_year'].nunique()} snapshot years")

    models, rankings = {}, []
    for category in CATEGORIES:
        model, metrics = train_one_category(train_df, category, feature_cols)
        models[category] = model
        if model is not None:
            X_train = prep_features(train_df, feature_cols)
            rankings.append(rank_drivers(model, X_train, feature_cols, category))

    score_2025(models, score_df, feature_cols)
    return models, rankings


if __name__ == "__main__":
    run()
