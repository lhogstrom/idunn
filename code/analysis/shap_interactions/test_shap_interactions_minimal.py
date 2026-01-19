"""
Minimal test case to isolate SHAP interaction calculation issue.

This script tests whether SHAP interaction values work with a simple
Random Forest binary classifier on synthetic data.
"""

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("=" * 70)
print("MINIMAL SHAP INTERACTION TEST")
print("=" * 70)

# 1. Create simple synthetic data
np.random.seed(42)
n_samples = 1000
n_features = 5

X = np.random.randn(n_samples, n_features)
# Create binary outcome with some signal
y = ((X[:, 0] + X[:, 1] * X[:, 2]) > 0).astype(int)

print(f"\nData shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")

# 2. Train simple Random Forest
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

print(f"\nModel trained. Train accuracy: {rf.score(X_train, y_train):.3f}")
print(f"Test accuracy: {rf.score(X_test, y_test):.3f}")

# 3. Test SHAP interaction values
print("\n" + "=" * 70)
print("TESTING SHAP INTERACTION VALUES")
print("=" * 70)

# Test 1: TreeExplainer without background data
print("\n[Test 1] TreeExplainer without background data:")
try:
    explainer1 = shap.TreeExplainer(rf)
    interaction_values1 = explainer1.shap_interaction_values(X_test[:10])

    print(f"  Shape: {np.array(interaction_values1).shape}")
    print(f"  Type: {type(interaction_values1)}")

    if isinstance(interaction_values1, list):
        print(f"  List length: {len(interaction_values1)}")
        for i, arr in enumerate(interaction_values1):
            print(f"  Class {i} - min: {np.min(arr):.6f}, max: {np.max(arr):.6f}, mean_abs: {np.mean(np.abs(arr)):.6f}")
    else:
        print(f"  Min: {np.min(interaction_values1):.6f}")
        print(f"  Max: {np.max(interaction_values1):.6f}")
        print(f"  Mean abs: {np.mean(np.abs(interaction_values1)):.6f}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 2: TreeExplainer with background data
print("\n[Test 2] TreeExplainer with background data:")
try:
    explainer2 = shap.TreeExplainer(rf, data=X_train[:100])
    interaction_values2 = explainer2.shap_interaction_values(X_test[:10])

    print(f"  Shape: {np.array(interaction_values2).shape}")
    print(f"  Type: {type(interaction_values2)}")

    if isinstance(interaction_values2, list):
        print(f"  List length: {len(interaction_values2)}")
        for i, arr in enumerate(interaction_values2):
            print(f"  Class {i} - min: {np.min(arr):.6f}, max: {np.max(arr):.6f}, mean_abs: {np.mean(np.abs(arr)):.6f}")
    else:
        print(f"  Min: {np.min(interaction_values2):.6f}")
        print(f"  Max: {np.max(interaction_values2):.6f}")
        print(f"  Mean abs: {np.mean(np.abs(interaction_values2)):.6f}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 3: Standard SHAP values (for comparison)
print("\n[Test 3] Standard SHAP values (for comparison):")
try:
    explainer3 = shap.TreeExplainer(rf)
    shap_values3 = explainer3.shap_values(X_test[:10])

    print(f"  Shape: {np.array(shap_values3).shape}")
    print(f"  Type: {type(shap_values3)}")

    if isinstance(shap_values3, list):
        print(f"  List length: {len(shap_values3)}")
        for i, arr in enumerate(shap_values3):
            print(f"  Class {i} - min: {np.min(arr):.6f}, max: {np.max(arr):.6f}, mean_abs: {np.mean(np.abs(arr)):.6f}")
    else:
        print(f"  Min: {np.min(shap_values3):.6f}")
        print(f"  Max: {np.max(shap_values3):.6f}")
        print(f"  Mean abs: {np.mean(np.abs(shap_values3)):.6f}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 70)
print("SHAP version:", shap.__version__)
print("=" * 70)
