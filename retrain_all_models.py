"""
Retrain All Models

Retrains all bridge and sensor health models with the latest code fixes.
This ensures all models have:
- trained_feature_cols attribute
- Updated empty DataFrame handling
- Latest code improvements

Usage:
    python retrain_all_models.py <DATA_DIR>

Example:
    python retrain_all_models.py "../STR Data - Merged"
"""

import sys
import os
import pickle
import pandas as pd
from datetime import datetime

sys.path.append('model')
from universal_model_v3 import UniversalModelV3
from sensor_health_model_v6 import SensorHealthMonitorV6


def retrain_structure(structure_id, data_dir):
    """Retrain both bridge and sensor models for a structure"""
    print(f"\n{'='*80}")
    print(f"Retraining: {structure_id}")
    print(f"{'='*80}")

    try:
        # Load data
        file_path = f"{data_dir}/{structure_id}_merged.xlsx"
        if not os.path.exists(file_path):
            print(f"  FAIL Data file not found: {file_path}")
            return False

        print(f"  Loading data...")
        df = pd.read_excel(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"  OK Loaded {len(df):,} records")

        # Train bridge model
        print(f"\n  Training bridge health model...")
        bridge_model = UniversalModelV3()
        bridge_model.train(df)

        # Verify trained_feature_cols exists
        if hasattr(bridge_model, 'trained_feature_cols'):
            print(f"  OK Model has trained_feature_cols ({len(bridge_model.trained_feature_cols)} features)")
        else:
            print(f"  WARNING  Model missing trained_feature_cols attribute")

        bridge_path = f'trained_models/bridge_health/{structure_id}_bridge_model.pkl'
        with open(bridge_path, 'wb') as f:
            pickle.dump(bridge_model, f)
        print(f"  OK Bridge model saved: {bridge_path}")

        # Train sensor model
        print(f"\n  Training sensor health model...")
        sensor_model = SensorHealthMonitorV6(structure_id=structure_id)
        sensor_model.train(df)

        sensor_path = f'trained_models/sensor_health/{structure_id}_sensor_model.pkl'
        with open(sensor_path, 'wb') as f:
            pickle.dump(sensor_model, f)
        print(f"  OK Sensor model saved: {sensor_path}")

        # Test empty DataFrame handling
        print(f"\n  Testing empty DataFrame handling...")
        try:
            empty_df = pd.DataFrame()
            sensor_result = sensor_model.score_sensor_daily(empty_df)
            bridge_features = bridge_model.generate_features(empty_df)
            print(f"  OK Empty DataFrame handling works correctly")
        except Exception as e:
            print(f"  FAIL Empty DataFrame test failed: {e}")
            return False

        print(f"\n  OKOKOK {structure_id} retrained successfully!")
        return True

    except Exception as e:
        print(f"  FAIL Error retraining {structure_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python retrain_all_models.py <DATA_DIR>")
        print("\nExample:")
        print("  python retrain_all_models.py '../STR Data - Merged'")
        sys.exit(1)

    data_dir = sys.argv[1]

    print("="*80)
    print("RETRAINING ALL BRIDGE & SENSOR HEALTH MODELS")
    print("="*80)
    print(f"\nData directory: {data_dir}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Find all structure files
    structures = []
    for file in os.listdir(data_dir):
        if file.endswith('_merged.xlsx'):
            structure_id = file.replace('_merged.xlsx', '')
            structures.append(structure_id)

    structures.sort()
    print(f"Found {len(structures)} structures: {', '.join(structures)}\n")

    # Retrain each structure
    results = {
        'success': [],
        'failed': []
    }

    for i, structure_id in enumerate(structures, 1):
        print(f"\n[{i}/{len(structures)}] Processing {structure_id}...")

        success = retrain_structure(structure_id, data_dir)

        if success:
            results['success'].append(structure_id)
        else:
            results['failed'].append(structure_id)

    # Summary
    print(f"\n\n{'='*80}")
    print("RETRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTotal structures: {len(structures)}")
    print(f"Successfully retrained: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")

    if results['success']:
        print(f"\nOK Successfully retrained:")
        for structure in results['success']:
            print(f"  - {structure}")

    if results['failed']:
        print(f"\nFAIL Failed to retrain:")
        for structure in results['failed']:
            print(f"  - {structure}")

    # Save report
    os.makedirs('test_output', exist_ok=True)
    with open('test_output/retraining_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL RETRAINING REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total: {len(structures)}\n")
        f.write(f"Success: {len(results['success'])}\n")
        f.write(f"Failed: {len(results['failed'])}\n\n")

        if results['success']:
            f.write("\nSuccessfully Retrained:\n")
            for structure in results['success']:
                f.write(f"  - {structure}\n")

        if results['failed']:
            f.write("\nFailed:\n")
            for structure in results['failed']:
                f.write(f"  - {structure}\n")

    print(f"\nReport saved: test_output/retraining_report.txt")

    print(f"\n{'='*80}")
    if len(results['failed']) == 0:
        print("OK ALL MODELS RETRAINED SUCCESSFULLY")
        print("="*80)
        return True
    else:
        print(f"FAIL {len(results['failed'])} MODELS FAILED")
        print("="*80)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
