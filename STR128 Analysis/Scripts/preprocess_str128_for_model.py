"""
Preprocessing Script for STR128 - Anomaly Detection Model Ready

This script:
1. Loads STR128 merged data
2. Creates combined displacement from paired sensors (DI549 + DI550)
3. Creates averaged dynamic tilt from redundant sensors (TI551, TI552, TI553)
4. Keeps stable tilt separate (TI554)
5. Adds sensor health metrics
6. Creates model-ready feature set
7. Saves preprocessed data

Author: Project Team
Date: 2025-12-05
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
INPUT_FILE = Path(r"c:\Users\pc\Downloads\Project_Anomaly detection IoT-20251203T010920Z-1-001\Project_Anomaly detection IoT\Shawon\Company Project\STR Data - Merged\STR128_merged.xlsx")
OUTPUT_DIR = Path(r"c:\Users\pc\Downloads\Project_Anomaly detection IoT-20251203T010920Z-1-001\Project_Anomaly detection IoT\Shawon\Company Project\Preprocessed Data")
OUTPUT_DIR.mkdir(exist_ok=True)

def preprocess_str128():
    """
    Preprocess STR128 data for anomaly detection model.

    Returns processed DataFrame with engineered features.
    """
    print("="*80)
    print("STR128 PREPROCESSING FOR ANOMALY DETECTION")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load data
    print("Loading STR128 merged data...")
    df = pd.read_excel(INPUT_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Loaded {len(df):,} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Sensors: {df['sensor_id'].nunique()}")
    print()

    # Step 1: Create wide format for easier feature engineering
    print("Step 1: Pivoting to wide format...")

    # Pivot each sensor type separately to handle different columns
    df_wide = df.pivot_table(
        index='timestamp',
        columns='sensor_id',
        values='value',
        aggfunc='first'
    )

    print(f"Wide format shape: {df_wide.shape}")
    print(f"Columns: {list(df_wide.columns)}")
    print()

    # Step 2: Create combined displacement (DI549 + DI550)
    print("Step 2: Creating combined displacement...")

    if 'DI549' in df_wide.columns and 'DI550' in df_wide.columns:
        df_wide['displacement_combined'] = df_wide['DI549'] + df_wide['DI550']

        # Statistics
        print(f"  DI549 stats: mean={df_wide['DI549'].mean():.2f}, std={df_wide['DI549'].std():.2f}")
        print(f"  DI550 stats: mean={df_wide['DI550'].mean():.2f}, std={df_wide['DI550'].std():.2f}")
        print(f"  Combined stats: mean={df_wide['displacement_combined'].mean():.2f}, std={df_wide['displacement_combined'].std():.2f}")
        print(f"  Valid combined measurements: {df_wide['displacement_combined'].notna().sum():,}")
    else:
        print("  WARNING: DI549 or DI550 not found!")
    print()

    # Step 3: Create averaged dynamic tilt (TI551, TI552, TI553)
    print("Step 3: Creating averaged dynamic tilt...")

    tilt_dynamic_sensors = ['TI551', 'TI552', 'TI553']
    available_dynamic = [s for s in tilt_dynamic_sensors if s in df_wide.columns]

    if len(available_dynamic) >= 2:
        df_wide['tilt_dynamic_avg'] = df_wide[available_dynamic].mean(axis=1)
        df_wide['tilt_dynamic_std'] = df_wide[available_dynamic].std(axis=1)

        print(f"  Averaging {len(available_dynamic)} sensors: {available_dynamic}")
        print(f"  Average mean: {df_wide['tilt_dynamic_avg'].mean():.2f}°")
        print(f"  Average std: {df_wide['tilt_dynamic_avg'].std():.2f}°")
        print(f"  Sensor agreement (avg std): {df_wide['tilt_dynamic_std'].mean():.4f}° (lower = better)")
        print(f"  Valid measurements: {df_wide['tilt_dynamic_avg'].notna().sum():,}")
    else:
        print(f"  WARNING: Only {len(available_dynamic)} dynamic tilt sensors found!")
    print()

    # Step 4: Keep stable tilt separate (TI554)
    print("Step 4: Renaming stable tilt sensor...")

    if 'TI554' in df_wide.columns:
        df_wide['tilt_stable'] = df_wide['TI554']
        print(f"  TI554 renamed to tilt_stable")
        print(f"  Mean: {df_wide['tilt_stable'].mean():.2f}°")
        print(f"  Std: {df_wide['tilt_stable'].std():.2f}°")
        print(f"  Valid measurements: {df_wide['tilt_stable'].notna().sum():,}")
    else:
        print("  WARNING: TI554 not found!")
    print()

    # Step 5: Process accelerometer data
    print("Step 5: Processing accelerometer data...")

    # Get accelerometer data separately (has p2p and rms)
    acc_data = df[df['sensor_type'] == 'accelerometer'].copy()

    if len(acc_data) > 0:
        # Pivot accelerometer p2p
        acc_p2p = acc_data.pivot_table(
            index='timestamp',
            columns='sensor_id',
            values='p2p',
            aggfunc='first'
        )

        # Pivot accelerometer rms
        acc_rms = acc_data.pivot_table(
            index='timestamp',
            columns='sensor_id',
            values='rms',
            aggfunc='first'
        )

        # Add to main dataframe with clear naming
        for col in acc_p2p.columns:
            df_wide[f'{col}_p2p'] = acc_p2p[col]

        for col in acc_rms.columns:
            df_wide[f'{col}_rms'] = acc_rms[col]

        print(f"  Accelerometers processed: {list(acc_p2p.columns)}")

        # Create averaged accelerometer features
        acc_sensors = list(acc_p2p.columns)
        if len(acc_sensors) >= 2:
            df_wide['vibration_p2p_avg'] = df_wide[[f'{s}_p2p' for s in acc_sensors]].mean(axis=1)
            df_wide['vibration_rms_avg'] = df_wide[[f'{s}_rms' for s in acc_sensors]].mean(axis=1)
            print(f"  Average vibration features created")
    else:
        print("  WARNING: No accelerometer data found!")
    print()

    # Step 6: Add temperature and other environmental data
    print("Step 6: Processing environmental sensors...")

    temp_data = df[df['sensor_type'] == 'temperature_probe'].copy()
    if len(temp_data) > 0:
        temp_pivot = temp_data.pivot_table(
            index='timestamp',
            columns='sensor_id',
            values='value',
            aggfunc='first'
        )

        for col in temp_pivot.columns:
            df_wide[f'{col}_temp'] = temp_pivot[col]

        print(f"  Temperature probes: {list(temp_pivot.columns)}")
    else:
        print("  No temperature probe data found")
    print()

    # Step 7: Add derived features
    print("Step 7: Creating derived features...")

    # Rate of change features (first-order difference)
    if 'displacement_combined' in df_wide.columns:
        df_wide['displacement_rate'] = df_wide['displacement_combined'].diff()
        print(f"  Added displacement_rate (rate of change)")

    if 'tilt_dynamic_avg' in df_wide.columns:
        df_wide['tilt_dynamic_rate'] = df_wide['tilt_dynamic_avg'].diff()
        print(f"  Added tilt_dynamic_rate (rate of change)")

    if 'tilt_stable' in df_wide.columns:
        df_wide['tilt_stable_rate'] = df_wide['tilt_stable'].diff()
        print(f"  Added tilt_stable_rate (rate of change)")
    print()

    # Step 8: Add temporal features
    print("Step 8: Adding temporal features...")

    df_wide['hour'] = df_wide.index.hour
    df_wide['day_of_week'] = df_wide.index.dayofweek
    df_wide['month'] = df_wide.index.month
    df_wide['is_weekend'] = (df_wide['day_of_week'] >= 5).astype(int)

    print(f"  Added: hour, day_of_week, month, is_weekend")
    print()

    # Step 9: Create sensor health metrics
    print("Step 9: Creating sensor health metrics...")

    # Displacement synchronization check
    if 'DI549' in df_wide.columns and 'DI550' in df_wide.columns:
        expected_combined = (df_wide['DI549'] + df_wide['DI550']).rolling(window=144, center=True).mean()
        df_wide['displacement_sync_error'] = abs(df_wide['displacement_combined'] - expected_combined)
        print(f"  Added displacement_sync_error (sensor health metric)")

    # Tilt sensor agreement (already have tilt_dynamic_std)
    if 'tilt_dynamic_std' in df_wide.columns:
        print(f"  tilt_dynamic_std tracks sensor agreement")
    print()

    # Step 10: Define model-ready feature set
    print("Step 10: Creating model-ready feature set...")

    model_features = []

    # Core structural features
    core_features = [
        'displacement_combined',
        'tilt_dynamic_avg',
        'tilt_stable'
    ]
    model_features.extend([f for f in core_features if f in df_wide.columns])

    # Vibration features
    vibration_features = [
        'vibration_p2p_avg',
        'vibration_rms_avg'
    ]
    model_features.extend([f for f in vibration_features if f in df_wide.columns])

    # Rate of change features
    rate_features = [
        'displacement_rate',
        'tilt_dynamic_rate',
        'tilt_stable_rate'
    ]
    model_features.extend([f for f in rate_features if f in df_wide.columns])

    # Sensor health features
    health_features = [
        'tilt_dynamic_std',
        'displacement_sync_error'
    ]
    model_features.extend([f for f in health_features if f in df_wide.columns])

    # Temporal features
    temporal_features = [
        'hour',
        'day_of_week',
        'month',
        'is_weekend'
    ]
    model_features.extend([f for f in temporal_features if f in df_wide.columns])

    print(f"  Model features selected: {len(model_features)}")
    for feat in model_features:
        print(f"    - {feat}")
    print()

    # Step 11: Save preprocessed data
    print("Step 11: Saving preprocessed data...")

    # Reset index to make timestamp a column
    df_processed = df_wide.reset_index()

    # Save full preprocessed data
    output_file = OUTPUT_DIR / "STR128_preprocessed.xlsx"
    df_processed.to_excel(output_file, index=False)
    print(f"  Full preprocessed data saved: {output_file}")
    print(f"  Shape: {df_processed.shape}")

    # Save model-ready data (only selected features, dropna)
    df_model = df_processed[['timestamp'] + model_features].copy()
    df_model_clean = df_model.dropna()

    output_model_file = OUTPUT_DIR / "STR128_model_ready.xlsx"
    df_model_clean.to_excel(output_model_file, index=False)
    print(f"  Model-ready data saved: {output_model_file}")
    print(f"  Shape: {df_model_clean.shape}")
    print(f"  Completeness: {len(df_model_clean) / len(df_processed) * 100:.1f}%")
    print()

    # Step 12: Generate summary statistics
    print("Step 12: Generating summary statistics...")

    summary_file = OUTPUT_DIR / "STR128_preprocessing_summary.txt"

    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STR128 PREPROCESSING SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("INPUT DATA:\n")
        f.write(f"  File: {INPUT_FILE.name}\n")
        f.write(f"  Original rows: {len(df):,}\n")
        f.write(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
        f.write(f"  Sensors: {df['sensor_id'].nunique()}\n\n")

        f.write("FEATURE ENGINEERING:\n")
        f.write(f"  Combined displacement: DI549 + DI550\n")
        f.write(f"  Dynamic tilt (avg): (TI551 + TI552 + TI553) / 3\n")
        f.write(f"  Stable tilt: TI554\n")
        f.write(f"  Total features created: {len(model_features)}\n\n")

        f.write("OUTPUT FILES:\n")
        f.write(f"  Full preprocessed: {output_file.name}\n")
        f.write(f"    - Shape: {df_processed.shape}\n")
        f.write(f"    - All sensors and derived features\n")
        f.write(f"  Model-ready: {output_model_file.name}\n")
        f.write(f"    - Shape: {df_model_clean.shape}\n")
        f.write(f"    - Selected features only, no missing values\n\n")

        f.write("BASELINE STATISTICS (for anomaly thresholds):\n")
        f.write("-"*80 + "\n")

        for feat in model_features:
            if feat in df_model_clean.columns and df_model_clean[feat].dtype in ['float64', 'int64']:
                mean = df_model_clean[feat].mean()
                std = df_model_clean[feat].std()
                min_val = df_model_clean[feat].min()
                max_val = df_model_clean[feat].max()

                f.write(f"\n{feat}:\n")
                f.write(f"  Mean: {mean:.4f}\n")
                f.write(f"  Std Dev: {std:.4f}\n")
                f.write(f"  Min: {min_val:.4f}\n")
                f.write(f"  Max: {max_val:.4f}\n")
                f.write(f"  Suggested threshold (mean ± 2*std): [{mean - 2*std:.4f}, {mean + 2*std:.4f}]\n")

        f.write("\n" + "="*80 + "\n")
        f.write("READY FOR ANOMALY DETECTION MODEL\n")
        f.write("="*80 + "\n")
        f.write("\nNext steps:\n")
        f.write("1. Load STR128_model_ready.xlsx\n")
        f.write("2. Split into train/test (80/20 or by time)\n")
        f.write("3. Train anomaly detection model (Isolation Forest, Autoencoder, etc.)\n")
        f.write("4. Use baseline statistics for threshold tuning\n")

    print(f"  Summary saved: {summary_file}")
    print()

    print("="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"\nFiles created:")
    print(f"  1. {output_file.name}")
    print(f"  2. {output_model_file.name}")
    print(f"  3. {summary_file.name}")
    print(f"\nYou can now load '{output_model_file.name}' and start building your anomaly detection model!")

    return df_processed, df_model_clean, model_features

if __name__ == "__main__":
    try:
        df_processed, df_model_clean, features = preprocess_str128()
        print("\nPreprocessing successful!")

    except Exception as e:
        print(f"\nERROR during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
