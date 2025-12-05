import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def analyze_data_quality(structure_name, save_report=True):
    """
    Comprehensive data quality analysis for a structure.
    """
    merged_path = f"/Users/shawon/Downloads/Company Project/STR Data - Merged/{structure_name}_merged.xlsx"
    report_dir = "/Users/shawon/Downloads/Company Project/Data Quality Reports"

    # Create report directory
    os.makedirs(report_dir, exist_ok=True)

    report_file = os.path.join(report_dir, f"{structure_name}_quality_report.txt")

    # Log buffer
    log = []

    def log_print(message):
        """Print and save to log"""
        print(message)
        log.append(message)

    if not os.path.exists(merged_path):
        msg = f"❌ File not found: {structure_name}_merged.xlsx"
        log_print(msg)
        return None

    log_print("="*80)
    log_print(f"📊 DATA QUALITY REPORT: {structure_name}")
    log_print("="*80)

    # Load data
    print("\n⏳ Loading data...")
    df = pd.read_excel(merged_path)
    print(f"✓ Loaded {len(df):,} rows")

    report = {
        'structure': structure_name,
        'total_rows': len(df),
        'sensors': {},
        'issues': []
    }

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Overall info
    print(f"\n{'='*80}")
    print("📋 OVERALL INFORMATION")
    print("="*80)
    print(f"   Total rows: {len(df):,}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    print(f"   Unique sensors: {df['sensor_id'].nunique()}")
    print(f"   Sensor types: {', '.join(df['sensor_type'].unique())}")

    # Analysis per sensor
    print(f"\n{'='*80}")
    print("🔍 PER-SENSOR ANALYSIS")
    print("="*80)

    sensors = df['sensor_id'].unique()

    for sensor_id in sorted(sensors):
        sensor_df = df[df['sensor_id'] == sensor_id].copy()
        sensor_type = sensor_df['sensor_type'].iloc[0]

        print(f"\n📡 SENSOR: {sensor_id} ({sensor_type})")
        print("-"*80)

        sensor_info = {
            'type': sensor_type,
            'total_rows': len(sensor_df),
            'date_range': {
                'start': sensor_df['timestamp'].min(),
                'end': sensor_df['timestamp'].max(),
                'days': (sensor_df['timestamp'].max() - sensor_df['timestamp'].min()).days
            },
            'issues': []
        }

        # Basic stats
        print(f"   Rows: {len(sensor_df):,}")
        print(f"   Date range: {sensor_df['timestamp'].min()} to {sensor_df['timestamp'].max()}")
        print(f"   Duration: {sensor_info['date_range']['days']} days")

        # Check sampling rate
        sensor_df = sensor_df.sort_values('timestamp')
        time_diffs = sensor_df['timestamp'].diff()
        median_interval = time_diffs.median()

        print(f"\n   ⏱️  Sampling Rate:")
        print(f"      Median interval: {median_interval}")

        if pd.notna(median_interval):
            # Expected number of samples
            total_duration = sensor_df['timestamp'].max() - sensor_df['timestamp'].min()
            expected_samples = int(total_duration / median_interval)
            actual_samples = len(sensor_df)
            missing_pct = ((expected_samples - actual_samples) / expected_samples * 100) if expected_samples > 0 else 0

            print(f"      Expected samples: {expected_samples:,}")
            print(f"      Actual samples: {actual_samples:,}")
            print(f"      Missing: {missing_pct:.1f}%")

            sensor_info['sampling'] = {
                'median_interval': str(median_interval),
                'expected_samples': expected_samples,
                'actual_samples': actual_samples,
                'missing_pct': missing_pct
            }

            if missing_pct > 10:
                issue = f"High missing data rate: {missing_pct:.1f}%"
                sensor_info['issues'].append(issue)
                report['issues'].append(f"{sensor_id}: {issue}")
                print(f"      ⚠️  {issue}")

        # Identify data columns (exclude metadata)
        data_columns = [col for col in sensor_df.columns
                       if col not in ['timestamp', 'sensor_id', 'sensor_type']]

        # Check for gaps in data
        print(f"\n   📊 Data Gaps Analysis:")

        # Find large gaps (> 2x median interval)
        if pd.notna(median_interval):
            large_gaps = time_diffs[time_diffs > median_interval * 2]
            if len(large_gaps) > 0:
                print(f"      Large gaps found: {len(large_gaps)}")
                print(f"      Largest gap: {large_gaps.max()}")

                sensor_info['gaps'] = {
                    'count': len(large_gaps),
                    'largest': str(large_gaps.max())
                }

                if len(large_gaps) > 10:
                    issue = f"Multiple large data gaps detected: {len(large_gaps)}"
                    sensor_info['issues'].append(issue)
                    report['issues'].append(f"{sensor_id}: {issue}")
                    print(f"      ⚠️  {issue}")
            else:
                print(f"      ✓ No large gaps detected")

        # Analyze each data column
        print(f"\n   📈 Data Column Analysis:")
        sensor_info['columns'] = {}

        for col in data_columns:
            # Skip if column is all NaN
            if sensor_df[col].isna().all():
                continue

            col_data = sensor_df[col].dropna()

            if len(col_data) == 0:
                continue

            # Basic statistics
            stats = {
                'count': len(col_data),
                'missing': sensor_df[col].isna().sum(),
                'missing_pct': (sensor_df[col].isna().sum() / len(sensor_df)) * 100,
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'median': float(col_data.median())
            }

            print(f"\n      {col}:")
            print(f"         Count: {stats['count']:,} ({100-stats['missing_pct']:.1f}% complete)")
            print(f"         Mean: {stats['mean']:.4f}")
            print(f"         Std Dev: {stats['std']:.4f}")
            print(f"         Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"         Median: {stats['median']:.4f}")

            # Outlier detection using IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            outlier_pct = (len(outliers) / len(col_data)) * 100

            stats['outliers'] = {
                'count': len(outliers),
                'percent': outlier_pct,
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }

            print(f"         Outliers (3×IQR): {len(outliers)} ({outlier_pct:.2f}%)")

            if outlier_pct > 5:
                issue = f"High outlier rate in {col}: {outlier_pct:.2f}%"
                sensor_info['issues'].append(issue)
                print(f"         ⚠️  {issue}")

            # Check for constant values (sensor stuck)
            unique_values = col_data.nunique()
            unique_pct = (unique_values / len(col_data)) * 100

            if unique_values == 1:
                issue = f"Constant value in {col} - sensor may be stuck!"
                sensor_info['issues'].append(issue)
                report['issues'].append(f"{sensor_id}: {issue}")
                print(f"         ⚠️  {issue}")
            elif unique_pct < 1:
                issue = f"Low variance in {col}: only {unique_values} unique values"
                sensor_info['issues'].append(issue)
                print(f"         ⚠️  {issue}")

            # Check for sudden jumps
            if len(col_data) > 1:
                changes = col_data.diff().abs()
                mean_change = changes.mean()
                std_change = changes.std()

                if pd.notna(mean_change) and pd.notna(std_change) and std_change > 0:
                    large_jumps = changes[changes > mean_change + 5 * std_change]

                    if len(large_jumps) > 0:
                        print(f"         ⚠️  Sudden jumps detected: {len(large_jumps)} instances")
                        stats['sudden_jumps'] = len(large_jumps)

            sensor_info['columns'][col] = stats

        # Overall sensor health
        print(f"\n   🏥 Sensor Health:")
        if len(sensor_info['issues']) == 0:
            print(f"      ✅ No major issues detected")
        else:
            print(f"      ⚠️  {len(sensor_info['issues'])} issue(s) detected:")
            for issue in sensor_info['issues']:
                print(f"         - {issue}")

        report['sensors'][sensor_id] = sensor_info

    # Summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print("="*80)
    print(f"\n   Total sensors analyzed: {len(sensors)}")
    print(f"   Total data rows: {len(df):,}")

    if len(report['issues']) == 0:
        print(f"\n   ✅ No critical issues detected!")
    else:
        print(f"\n   ⚠️  Total issues found: {len(report['issues'])}")
        print(f"\n   Issues by sensor:")
        for issue in report['issues'][:10]:  # Show first 10
            print(f"      - {issue}")
        if len(report['issues']) > 10:
            print(f"      ... and {len(report['issues']) - 10} more")

    print("\n" + "="*80)

    return report


def generate_summary_report():
    """Generate summary report for all structures"""
    merged_path = "/Users/shawon/Downloads/Company Project/STR Data - Merged"
    structures = sorted([f.replace('_merged.xlsx', '')
                        for f in os.listdir(merged_path)
                        if f.endswith('_merged.xlsx')])

    print("="*80)
    print("🚀 GENERATING DATA QUALITY REPORTS FOR ALL STRUCTURES")
    print("="*80)
    print(f"\nFound {len(structures)} structures to analyze\n")

    all_reports = []

    for idx, structure in enumerate(structures, 1):
        print(f"\n\n{'#'*80}")
        print(f"# ANALYZING {idx}/{len(structures)}: {structure}")
        print(f"{'#'*80}\n")

        report = analyze_data_quality(structure)
        if report:
            all_reports.append(report)

        print(f"\n✅ {structure} analysis complete")

    # Overall summary
    print("\n\n" + "="*80)
    print("📊 OVERALL SUMMARY - ALL STRUCTURES")
    print("="*80)

    total_sensors = sum(len(r['sensors']) for r in all_reports)
    total_rows = sum(r['total_rows'] for r in all_reports)
    total_issues = sum(len(r['issues']) for r in all_reports)

    print(f"\n✅ Structures analyzed: {len(all_reports)}")
    print(f"✅ Total sensors: {total_sensors}")
    print(f"✅ Total data rows: {total_rows:,}")
    print(f"⚠️  Total issues detected: {total_issues}")

    # Structures with most issues
    structures_with_issues = [(r['structure'], len(r['issues']))
                             for r in all_reports if len(r['issues']) > 0]
    structures_with_issues.sort(key=lambda x: x[1], reverse=True)

    if structures_with_issues:
        print(f"\n⚠️  Structures with most issues:")
        for struct, issue_count in structures_with_issues[:5]:
            print(f"   {struct}: {issue_count} issues")

    print("\n" + "="*80)


if __name__ == "__main__":
    # You can analyze one structure or all
    import sys

    if len(sys.argv) > 1:
        # Analyze specific structure
        structure_name = sys.argv[1]
        analyze_data_quality(structure_name)
    else:
        # Analyze all structures
        generate_summary_report()
