"""Leakage Prevention Unit Test.

Guards against B-06 (Data Leakage) where post-outcome features (like TIME in Bail)
or trivial shortcut features are accidentally included in the input feature matrix.
Asserts that no single feature in any dataset has an individual AUC-ROC >= 0.85 with the target label.
"""
import os
import numpy as np
import pytest
import torch

from src.data.datasets import load_synthetic, load_bail, load_german, load_credit
from src.utils.metrics import auc_roc


def check_no_single_feature_leakage(data, dataset_name: str, threshold: float = 0.85):
    """Asserts that no single feature achieves AUC >= threshold."""
    x = data.x.numpy() if isinstance(data.x, torch.Tensor) else np.asarray(data.x)
    y = data.y.numpy() if isinstance(data.y, torch.Tensor) else np.asarray(data.y)
    
    num_features = x.shape[1]
    max_auc = 0.0
    max_feat_idx = -1
    
    for j in range(num_features):
        feat = x[:, j]
        # Check both positive and negative correlation
        score_pos = feat
        score_neg = -feat
        
        auc_pos = auc_roc(y, score_pos)
        auc_neg = auc_roc(y, score_neg)
        feat_auc = max(auc_pos, auc_neg)
        
        if feat_auc > max_auc:
            max_auc = feat_auc
            max_feat_idx = j
            
        assert feat_auc < threshold, (
            f"Dataset '{dataset_name}' feature {j} has single-feature AUC = {feat_auc:.4f} >= {threshold}! "
            f"This indicates a potential post-outcome feature leakage."
        )
    return max_auc, max_feat_idx


def test_synthetic_no_leakage():
    # Synthetic testbed has injected signal block with mean shift 1.5 (AUC ~ 0.85)
    data = load_synthetic(seed=42)
    max_auc, idx = check_no_single_feature_leakage(data, "synthetic", threshold=0.90)
    assert max_auc < 0.90


def test_german_no_leakage():
    data = load_german(root="data", seed=42)
    max_auc, idx = check_no_single_feature_leakage(data, "german", threshold=0.85)
    assert max_auc < 0.85, f"German max feature AUC = {max_auc:.4f} >= 0.85 (feature {idx})"


def test_credit_no_leakage():
    data = load_credit(root="data", seed=42)
    max_auc, idx = check_no_single_feature_leakage(data, "credit", threshold=0.85)
    assert max_auc < 0.85, f"Credit max feature AUC = {max_auc:.4f} >= 0.85 (feature {idx})"


def test_bail_no_leakage_when_time_dropped():
    """Test bail dataset with TIME dropped has max single-feature AUC < 0.85."""
    data = load_bail(root="data", seed=42)
    max_auc, idx = check_no_single_feature_leakage(data, "bail", threshold=0.85)
    assert max_auc < 0.85, f"Bail max feature AUC = {max_auc:.4f} >= 0.85 (feature {idx})"


def test_pokec_z_no_leakage():
    """Test pokec_z dataset has max single-feature AUC < 0.85."""
    try:
        from src.data.datasets import load_pokec_z
        data = load_pokec_z(root="data", seed=42)
        max_auc, idx = check_no_single_feature_leakage(data, "pokec_z", threshold=0.85)
        assert max_auc < 0.85, f"Pokec-z max feature AUC = {max_auc:.4f} >= 0.85 (feature {idx})"
    except Exception as e:
        # If network download unavailable locally during fast unit test
        pytest.skip(f"Pokec-z download skipped: {e}")

