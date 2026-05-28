#!/usr/bin/env python3
"""Test data loaders to identify hanging issues"""

import sys
sys.path.insert(0, "/Users/jay19/Desktop/Zidio Project")

from pathlib import Path
import time

# Test imports first
print("Testing imports...")
start = time.time()
try:
    from src.dashboard.data_access import (
        load_dashboard_summary,
        load_churn_outputs,
        load_inventory_outputs,
        load_monitoring_summary,
        load_forecast_comparison,
        load_ensemble_metrics,
    )
    print(f"✓ Imports successful ({time.time() - start:.2f}s)")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test each data loader individually
loaders = [
    ("load_churn_outputs", load_churn_outputs),
    ("load_inventory_outputs", load_inventory_outputs),
    ("load_monitoring_summary", load_monitoring_summary),
    ("load_forecast_comparison", load_forecast_comparison),
    ("load_ensemble_metrics", load_ensemble_metrics),
]

for name, loader in loaders:
    print(f"\nTesting {name}...")
    start = time.time()
    try:
        result = loader()
        elapsed = time.time() - start
        if isinstance(result, dict):
            print(f"✓ {name}: returned dict with {len(result)} keys ({elapsed:.2f}s)")
        else:
            print(f"✓ {name}: returned {type(result).__name__} ({elapsed:.2f}s)")
            if hasattr(result, 'shape'):
                print(f"  Shape: {result.shape}")
    except Exception as e:
        print(f"✗ {name} failed: {e}")
        import traceback
        traceback.print_exc()

# Test the combined loader
print(f"\nTesting load_dashboard_summary...")
start = time.time()
try:
    summary = load_dashboard_summary()
    elapsed = time.time() - start
    print(f"✓ load_dashboard_summary: returned dict with {len(summary)} keys ({elapsed:.2f}s)")
    for key in summary:
        val = summary[key]
        if hasattr(val, 'shape'):
            print(f"  {key}: {type(val).__name__} with shape {val.shape}")
        else:
            print(f"  {key}: {type(val).__name__}")
except Exception as e:
    print(f"✗ load_dashboard_summary failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ All tests completed!")
