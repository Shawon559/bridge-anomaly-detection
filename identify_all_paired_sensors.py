"""
Script to identify ALL paired sensors across all 21 bridge structures.

This script:
1. Analyzes each structure's merged data
2. Identifies potential sensor pairs (displacement and tilt)
3. Checks timestamp synchronization
4. Checks for complementary value patterns
5. Generates comprehensive pairing report

Author: Project Team
Date: 2025-12-05
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Define paths
MERGED_DATA_DIR = Path(r"c:\Users\pc\Downloads\Project_Anomaly detection IoT-20251203T010920Z-1-001\Project_Anomaly detection IoT\Shawon\Company Project\STR Data - Merged")
OUTPUT_DIR = Path(r"c:\Users\pc\Downloads\Project_Anomaly detection IoT-20251203T010920Z-1-001\Project_Anomaly detection IoT\Shawon\Company Project\Sensor Pairing Analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_structure_data(file_path):
    """Load merged structure data."""
    print(f"  Loading {file_path.name}...")
    df = pd.read_excel(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def get_sensors_by_type(df, sensor_type):
    """Get all sensors of a specific type."""
    sensors = df[df['sensor_type'] == sensor_type]['sensor_id'].unique()
    return sorted(sensors)

def check_synchronization(df, sensor1_id, sensor2_id):
    """
    Check timestamp synchronization between two sensors.
    Returns: sync_percentage, matching_timestamps, total_timestamps
    """
    # Get timestamps for each sensor
    ts1 = set(df[df['sensor_id'] == sensor1_id]['timestamp'])
    ts2 = set(df[df['sensor_id'] == sensor2_id]['timestamp'])

    # Find matching timestamps
    matching = ts1.intersection(ts2)
    total = ts1.union(ts2)

    if len(total) == 0:
        return 0.0, 0, 0

    sync_percentage = (len(matching) / len(total)) * 100
    return sync_percentage, len(matching), len(total)

def analyze_value_patterns(df, sensor1_id, sensor2_id):
    """
    Analyze value patterns to identify complementary sensors.
    Returns: dict with statistics
    """
    # Get data for both sensors at matching timestamps
    df1 = df[df['sensor_id'] == sensor1_id][['timestamp', 'value']].rename(columns={'value': 'value1'})
    df2 = df[df['sensor_id'] == sensor2_id][['timestamp', 'value']].rename(columns={'value': 'value2'})

    # Merge on timestamp
    merged = pd.merge(df1, df2, on='timestamp', how='inner')

    if len(merged) == 0:
        return None

    # Calculate statistics
    stats = {
        'sensor1_min': merged['value1'].min(),
        'sensor1_max': merged['value1'].max(),
        'sensor1_mean': merged['value1'].mean(),
        'sensor1_std': merged['value1'].std(),
        'sensor2_min': merged['value2'].min(),
        'sensor2_max': merged['value2'].max(),
        'sensor2_mean': merged['value2'].mean(),
        'sensor2_std': merged['value2'].std(),
        'combined_min': (merged['value1'] + merged['value2']).min(),
        'combined_max': (merged['value1'] + merged['value2']).max(),
        'combined_mean': (merged['value1'] + merged['value2']).mean(),
        'combined_std': (merged['value1'] + merged['value2']).std(),
        'correlation': merged['value1'].corr(merged['value2']),
        'matching_count': len(merged)
    }

    return stats

def identify_pairs_in_structure(df, structure_name, sensor_type='displacement'):
    """
    Identify paired sensors in a structure.

    Criteria for pairing:
    1. High timestamp synchronization (>95%)
    2. Complementary value patterns (one positive, one negative OR both oscillating)
    3. Combined value shows smaller range than individual sensors
    """
    print(f"\n{'='*80}")
    print(f"Analyzing {structure_name} - {sensor_type.upper()} sensors")
    print(f"{'='*80}")

    # Get all sensors of this type
    sensors = get_sensors_by_type(df, sensor_type)

    if len(sensors) < 2:
        print(f"  Only {len(sensors)} {sensor_type} sensor(s) found. Need at least 2 for pairing.")
        return []

    print(f"  Found {len(sensors)} {sensor_type} sensors: {sensors}")

    # Check all possible pairs
    pairs = []
    checked_pairs = set()

    for i, sensor1 in enumerate(sensors):
        for sensor2 in sensors[i+1:]:
            pair_key = tuple(sorted([sensor1, sensor2]))
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)

            print(f"\n  Checking pair: {sensor1} + {sensor2}")

            # Check synchronization
            sync_pct, matching, total = check_synchronization(df, sensor1, sensor2)
            print(f"    Synchronization: {sync_pct:.2f}% ({matching}/{total} timestamps)")

            if sync_pct < 90:  # Require at least 90% synchronization
                print(f"    ❌ Low synchronization ({sync_pct:.2f}% < 90%)")
                continue

            # Analyze value patterns
            stats = analyze_value_patterns(df, sensor1, sensor2)

            if stats is None:
                print(f"    ❌ No matching data")
                continue

            # Check if this looks like a paired sensor system
            # Criteria: combined range is significantly smaller OR correlation is negative
            sensor1_range = stats['sensor1_max'] - stats['sensor1_min']
            sensor2_range = stats['sensor2_max'] - stats['sensor2_min']
            combined_range = stats['combined_max'] - stats['combined_min']

            print(f"    Sensor1 range: {sensor1_range:.2f} (mean: {stats['sensor1_mean']:.2f}, std: {stats['sensor1_std']:.2f})")
            print(f"    Sensor2 range: {sensor2_range:.2f} (mean: {stats['sensor2_mean']:.2f}, std: {stats['sensor2_std']:.2f})")
            print(f"    Combined range: {combined_range:.2f} (mean: {stats['combined_mean']:.2f}, std: {stats['combined_std']:.2f})")
            print(f"    Correlation: {stats['correlation']:.4f}")

            # Determine if paired
            is_paired = False
            pairing_reason = ""

            # Check for complementary patterns (opposite signs)
            if stats['sensor1_mean'] > 0 and stats['sensor2_mean'] < 0:
                is_paired = True
                pairing_reason = "Complementary (one positive, one negative)"
            elif stats['sensor1_mean'] < 0 and stats['sensor2_mean'] > 0:
                is_paired = True
                pairing_reason = "Complementary (one negative, one positive)"
            # Check for differential measurement (combined range << individual ranges)
            elif combined_range < min(sensor1_range, sensor2_range) * 0.5:
                is_paired = True
                pairing_reason = f"Differential (combined range {combined_range:.2f} << individual ranges)"
            # Check for negative correlation (common mode rejection)
            elif stats['correlation'] < -0.5:
                is_paired = True
                pairing_reason = f"Negative correlation ({stats['correlation']:.4f})"

            if is_paired:
                print(f"    ✅ PAIRED SENSORS IDENTIFIED: {pairing_reason}")
                pairs.append({
                    'structure': structure_name,
                    'sensor_type': sensor_type,
                    'sensor1': sensor1,
                    'sensor2': sensor2,
                    'sync_percentage': sync_pct,
                    'matching_timestamps': matching,
                    'pairing_reason': pairing_reason,
                    **stats
                })
            else:
                print(f"    ❌ Not paired (no clear complementary pattern)")

    return pairs

def analyze_all_structures():
    """Analyze all structures and identify all paired sensors."""
    print("="*80)
    print("PAIRED SENSOR IDENTIFICATION - ALL STRUCTURES")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    all_pairs = []
    structure_files = sorted(MERGED_DATA_DIR.glob("STR*_merged.xlsx"))

    print(f"Found {len(structure_files)} structure files\n")

    for file_path in structure_files:
        structure_name = file_path.stem.replace('_merged', '')

        try:
            # Load data
            df = load_structure_data(file_path)

            # Check displacement sensors
            displacement_pairs = identify_pairs_in_structure(df, structure_name, 'displacement')
            all_pairs.extend(displacement_pairs)

            # Check tiltmeter sensors
            tilt_pairs = identify_pairs_in_structure(df, structure_name, 'tilt')
            all_pairs.extend(tilt_pairs)

        except Exception as e:
            print(f"\n❌ ERROR processing {structure_name}: {str(e)}\n")
            continue

    return all_pairs

def generate_report(all_pairs):
    """Generate comprehensive report of all paired sensors."""

    # Convert to DataFrame
    if len(all_pairs) == 0:
        print("\n" + "="*80)
        print("NO PAIRED SENSORS FOUND")
        print("="*80)
        return

    pairs_df = pd.DataFrame(all_pairs)

    # Save detailed CSV
    csv_path = OUTPUT_DIR / "all_paired_sensors_detailed.csv"
    pairs_df.to_csv(csv_path, index=False)
    print(f"\n✅ Detailed CSV saved: {csv_path}")

    # Generate text report
    report_path = OUTPUT_DIR / "00_PAIRED_SENSORS_SUMMARY.txt"

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PAIRED SENSOR ANALYSIS - SUMMARY REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total paired sensor systems identified: {len(all_pairs)}\n")
        f.write(f"Structures with paired sensors: {pairs_df['structure'].nunique()}\n\n")

        # By sensor type
        f.write(f"BY SENSOR TYPE\n")
        f.write("-"*80 + "\n")
        type_counts = pairs_df['sensor_type'].value_counts()
        for sensor_type, count in type_counts.items():
            f.write(f"  {sensor_type.capitalize()}: {count} pairs\n")
        f.write("\n")

        # By structure
        f.write(f"BY STRUCTURE\n")
        f.write("-"*80 + "\n")
        for structure in sorted(pairs_df['structure'].unique()):
            struct_pairs = pairs_df[pairs_df['structure'] == structure]
            f.write(f"\n{structure}: {len(struct_pairs)} paired system(s)\n")

            for idx, row in struct_pairs.iterrows():
                f.write(f"  ├─ {row['sensor1']} + {row['sensor2']} ({row['sensor_type']})\n")
                f.write(f"  │  Sync: {row['sync_percentage']:.1f}% ({row['matching_timestamps']} timestamps)\n")
                f.write(f"  │  Reason: {row['pairing_reason']}\n")
                f.write(f"  │  Combined: mean={row['combined_mean']:.2f}, std={row['combined_std']:.2f}\n")
                f.write(f"  │  Correlation: {row['correlation']:.4f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED STATISTICS\n")
        f.write("="*80 + "\n\n")

        for idx, row in pairs_df.iterrows():
            f.write(f"{row['structure']} - {row['sensor1']} + {row['sensor2']}\n")
            f.write(f"  Type: {row['sensor_type']}\n")
            f.write(f"  Synchronization: {row['sync_percentage']:.2f}% ({row['matching_timestamps']} matching timestamps)\n")
            f.write(f"  Pairing Reason: {row['pairing_reason']}\n")
            f.write(f"  Sensor 1: range=[{row['sensor1_min']:.2f}, {row['sensor1_max']:.2f}], mean={row['sensor1_mean']:.2f}, std={row['sensor1_std']:.2f}\n")
            f.write(f"  Sensor 2: range=[{row['sensor2_min']:.2f}, {row['sensor2_max']:.2f}], mean={row['sensor2_mean']:.2f}, std={row['sensor2_std']:.2f}\n")
            f.write(f"  Combined: range=[{row['combined_min']:.2f}, {row['combined_max']:.2f}], mean={row['combined_mean']:.2f}, std={row['combined_std']:.2f}\n")
            f.write(f"  Correlation: {row['correlation']:.4f}\n")
            f.write("\n")

    print(f"✅ Summary report saved: {report_path}")

    # Print summary to console
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total paired sensor systems: {len(all_pairs)}")
    print(f"Structures with pairs: {pairs_df['structure'].nunique()}")
    print(f"\nBy sensor type:")
    for sensor_type, count in type_counts.items():
        print(f"  {sensor_type.capitalize()}: {count} pairs")
    print("\nStructures with paired sensors:")
    for structure in sorted(pairs_df['structure'].unique()):
        count = len(pairs_df[pairs_df['structure'] == structure])
        print(f"  {structure}: {count} pair(s)")

if __name__ == "__main__":
    print("Starting paired sensor identification...\n")

    # Analyze all structures
    all_pairs = analyze_all_structures()

    # Generate report
    generate_report(all_pairs)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
