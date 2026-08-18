import unittest
import pandas as pd
import numpy as np
import xgboost as xgb
from models.train_models import prep_features, train_one_category
from features.build_features import FEATURE_COLUMNS


class TestModelingPipeline(unittest.TestCase):
    
    def setUp(self):
        # Create a synthetic dataset containing both features and labels
        self.num_records = 20
        np.random.seed(42)
        
        # Build training set features
        data = {
            "bridge_key": [f"99-{i:06d}" for i in range(self.num_records)],
            "cause": ["scour"] * 6 + ["collision"] * 14,  # 6 positive scour labels (meets >=5 check)
            "scour_code": np.random.choice(["N", "8", "3", "1"], self.num_records),
            "scour_flag": np.random.choice([0, 1], self.num_records),
            "waterway_adequacy": np.random.randint(2, 9, self.num_records),
            "channel_cond": np.random.randint(2, 9, self.num_records),
            "deck_cond": np.random.randint(2, 9, self.num_records),
            "superstructure_cond": np.random.randint(2, 9, self.num_records),
            "substructure_cond": np.random.randint(2, 9, self.num_records),
            "culvert_cond": np.random.randint(2, 9, self.num_records),
            "lowest_major_rating": np.random.randint(2, 9, self.num_records),
            "bridge_age": np.random.randint(5, 80, self.num_records),
            "reconstruction_age": np.random.randint(5, 80, self.num_records),
            "operating_rating": np.random.uniform(20, 80, self.num_records),
            "inventory_rating": np.random.uniform(15, 60, self.num_records),
            "load_deficient_flag": np.random.choice([0, 1], self.num_records),
            "adt": np.random.randint(100, 10000, self.num_records),
            "pct_truck_traffic": np.random.uniform(1, 20, self.num_records),
            "fracture_critical_flag": np.random.choice([0, 1], self.num_records),
            # Categorical fields as text
            "structure_kind": np.random.choice(["1", "2", "3"], self.num_records),
            "structure_type": np.random.choice(["01", "02", "19"], self.num_records),
            "design_load": np.random.choice(["1", "5", "A", "C"], self.num_records),
        }
        self.df = pd.DataFrame(data)
        
    def test_prep_features(self):
        """Verify that prep_features assigns correct categorical and numeric types."""
        X = prep_features(self.df, FEATURE_COLUMNS)
        
        # Verify columns exist
        for col in FEATURE_COLUMNS:
            self.assertIn(col, X.columns)
            
        # Verify categorical columns are category dtype
        self.assertEqual(X["structure_kind"].dtype, "category")
        self.assertEqual(X["structure_type"].dtype, "category")
        self.assertEqual(X["design_load"].dtype, "category")
        
        # Verify numeric columns are numeric dtype
        self.assertTrue(pd.api.types.is_numeric_dtype(X["bridge_age"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(X["lowest_major_rating"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(X["scour_flag"]))

    def test_train_one_category_success(self):
        """Verify model trains successfully when positive class count is met."""
        model, metrics = train_one_category(self.df, "scour", FEATURE_COLUMNS)
        
        # Verify model is a fitted XGBClassifier
        self.assertIsNotNone(model)
        self.assertIsInstance(model, xgb.XGBClassifier)
        
        # Verify metrics
        self.assertIn("roc_auc", metrics)
        self.assertIn("pr_auc", metrics)
        self.assertEqual(metrics["n_pos"], 6)
        self.assertTrue(0.0 <= metrics["roc_auc"] <= 1.0)
        
    def test_train_one_category_skipped(self):
        """Verify training is skipped when positive class count is < 5."""
        # Only 2 positive 'fire' labels
        self.df.loc[0:1, "cause"] = "fire"
        self.df.loc[2:, "cause"] = "collision"
        
        model, metrics = train_one_category(self.df, "fire", FEATURE_COLUMNS)
        
        self.assertIsNone(model)
        self.assertIsNone(metrics)


if __name__ == "__main__":
    unittest.main()
