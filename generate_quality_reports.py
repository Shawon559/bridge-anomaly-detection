import pandas as pd
import numpy as np
import os
from datetime import datetime

def analyze_structure_quality(structure_name, report_dir):
    """
    Analyze data quality for one structure and save report.
    """
    merged_path = f"/Users/shawon/Downloads/Company Project/STR Data - Merged/{structure_name}_merged.xlsx"
    report_file = os.path.join(report_dir, f"{structure_name}_quality_report.txt")

    log = []

    def log_line(message):
        log.append(message)
        print(message)

    if not os.path.exists(merged_path):
        log_line(f"❌ File not found: {structure_name}_merged.xlsx")
        return None

    log_line("="*80)
    log_line(f"📊 DATA QUALITY REPORT: {structure_name}")
    log_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_line("="*80)

    # Load data
    print("⏳ Loading data...")
    df = pd.read_excel(merged_path)
    log_line(f"\n✓ Loaded {len(df):,} rows")

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Overall info
    log_line(f"\n{'='*80}")
    log_line("📋 OVERALL INFORMATION")
    log_line("="*80)
    log_line(f"   Total rows: {len(df):,}")
    log_line(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    log_line(f"   Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    log_line(f"   Unique sensors: {df['sensor_id'].nunique()}")
    log_line(f"   Sensor types: {', '.join(df['sensor_type'].unique())}")

    report_data = {
        'structure': structure_name,
        'total_rows': len(df),
        'date_range': {'start': df['timestamp'].min(), 'end': df['timestamp'].max()},
        'sensors': {},
        'issues_count': 0
    }

    # Per-sensor analysis
    log_line(f"\n{'='*80}")
    log_line("🔍 PER-SENSOR ANALYSIS")
    log_line("="*80)

    sensors = sorted(df['sensor_id'].unique())

    for sensor_id in sensors:
        sensor_df = df[df['sensor_id'] == sensor_id].copy()
        sensor_type = sensor_df['sensor_type'].iloc[0]

        log_line(f"\n📡 SENSOR: {sensor_id} ({sensor_type})")
        log_line("-"*80)

        sensor_info = {
            'type': sensor_type,
            'total_rows': len(sensor_df),
            'issues': []
        }

        log_line(f"   Rows: {len(sensor_df):,}")
        log_line(f"   Date range: {sensor_df['timestamp'].min()} to {sensor_df['timestamp'].max()}")

        # Sampling rate
        sensor_df = sensor_df.sort_values('timestamp')
        time_diffs = sensor_df['timestamp'].diff()
        median_interval = time_diffs.median()

        log_line(f"\n   ⏱️  Sampling Rate:")
        log_line(f"      Median interval: {median_interval}")

        if pd.notna(median_interval):
            total_duration = sensor_df['timestamp'].max() - sensor_df['timestamp'].min()
            expected_samples = int(total_duration / median_interval)
            actual_samples = len(sensor_df)
            missing_pct = ((expected_samples - actual_samples) / expected_samples * 100) if expected_samples > 0 else 0

            log_line(f"      Expected samples: {expected_samples:,}")
            log_line(f"      Actual samples: {actual_samples:,}")
            log_line(f"      Data completeness: {100-missing_pct:.1f}%")

            if missing_pct > 10:
                issue = f"Missing data: {missing_pct:.1f}%"
                sensor_info['issues'].append(issue)
                log_line(f"      ⚠️  {issue}")

        # Data gaps
        if pd.notna(median_interval):
            large_gaps = time_diffs[time_diffs > median_interval * 2]
            if len(large_gaps) > 0:
                log_line(f"\n   📊 Data Gaps:")
                log_line(f"      Large gaps: {len(large_gaps)}")
                log_line(f"      Largest gap: {large_gaps.max()}")
                if len(large_gaps) > 10:
                    sensor_info['issues'].append(f"{len(large_gaps)} data gaps")

        # Data columns analysis
        data_columns = [col for col in sensor_df.columns
                       if col not in ['timestamp', 'sensor_id', 'sensor_type']]

        log_line(f"\n   📈 Data Statistics:")

        for col in data_columns:
            if sensor_df[col].isna().all():
                continue

            col_data = sensor_df[col].dropna()
            if len(col_data) == 0:
                continue

            log_line(f"\n      {col}:")
            log_line(f"         Mean: {col_data.mean():.4f}, Std: {col_data.std():.4f}")
            log_line(f"         Range: [{col_data.min():.4f}, {col_data.max():.4f}]")

            # Outliers
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = col_data[(col_data < Q1 - 3*IQR) | (col_data > Q3 + 3*IQR)]
            outlier_pct = (len(outliers) / len(col_data)) * 100

            log_line(f"         Outliers: {len(outliers)} ({outlier_pct:.2f}%)")

            if outlier_pct > 5:
                sensor_info['issues'].append(f"{col}: {outlier_pct:.1f}% outliers")

        # Sensor health
        log_line(f"\n   🏥 Sensor Health:")
        if len(sensor_info['issues']) == 0:
            log_line(f"      ✅ No issues detected")
        else:
            log_line(f"      ⚠️  {len(sensor_info['issues'])} issue(s):")
            for issue in sensor_info['issues']:
                log_line(f"         - {issue}")

        report_data['sensors'][sensor_id] = sensor_info
        report_data['issues_count'] += len(sensor_info['issues'])

    # Summary
    log_line(f"\n{'='*80}")
    log_line("📊 SUMMARY")
    log_line("="*80)
    log_line(f"\n   Total sensors: {len(sensors)}")
    log_line(f"   Total rows: {len(df):,}")
    log_line(f"   Total issues: {report_data['issues_count']}")

    if report_data['issues_count'] == 0:
        log_line(f"\n   ✅ No critical issues detected!")
    else:
        log_line(f"\n   ⚠️  Review individual sensor reports above for details")

    log_line("\n" + "="*80)

    # Save report
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log))

    print(f"✓ Report saved: {report_file}")

    return report_data


def generate_all_reports():
    """Generate quality reports for all structures"""
    merged_path = "/Users/shawon/Downloads/Company Project/STR Data - Merged"
    report_dir = "/Users/shawon/Downloads/Company Project/Data Quality Reports"

    os.makedirs(report_dir, exist_ok=True)

    structures = sorted([f.replace('_merged.xlsx', '')
                        for f in os.listdir(merged_path)
                        if f.endswith('_merged.xlsx')])

    print("="*80)
    print("🚀 GENERATING DATA QUALITY REPORTS")
    print("="*80)
    print(f"\nFound {len(structures)} structures")
    print(f"Reports will be saved to: {report_dir}\n")

    all_reports = []

    for idx, structure in enumerate(structures, 1):
        print(f"\n{'#'*80}")
        print(f"# {idx}/{len(structures)}: {structure}")
        print(f"{'#'*80}\n")

        report = analyze_structure_quality(structure, report_dir)
        if report:
            all_reports.append(report)

    # Generate combined summary report
    print(f"\n\n{'='*80}")
    print("📊 GENERATING COMBINED SUMMARY REPORT")
    print("="*80)

    summary_file = os.path.join(report_dir, "00_COMBINED_SUMMARY.txt")
    summary = []

    summary.append("="*80)
    summary.append("COMBINED DATA QUALITY SUMMARY - ALL STRUCTURES")
    summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("="*80)

    # Overall statistics
    total_sensors = sum(len(r['sensors']) for r in all_reports)
    total_rows = sum(r['total_rows'] for r in all_reports)
    total_issues = sum(r['issues_count'] for r in all_reports)

    summary.append(f"\n{'='*80}")
    summary.append("📊 OVERALL STATISTICS")
    summary.append("="*80)
    summary.append(f"\n   Structures analyzed: {len(all_reports)}")
    summary.append(f"   Total sensors: {total_sensors}")
    summary.append(f"   Total data rows: {total_rows:,}")
    summary.append(f"   Total issues detected: {total_issues}")

    # Per-structure summary
    summary.append(f"\n{'='*80}")
    summary.append("📋 PER-STRUCTURE SUMMARY")
    summary.append("="*80)

    for report in all_reports:
        summary.append(f"\n{report['structure']}:")
        summary.append(f"   Sensors: {len(report['sensors'])}")
        summary.append(f"   Rows: {report['total_rows']:,}")
        summary.append(f"   Date range: {report['date_range']['start']} to {report['date_range']['end']}")
        summary.append(f"   Issues: {report['issues_count']}")

        if report['issues_count'] > 0:
            summary.append(f"   Status: ⚠️  Needs attention")
        else:
            summary.append(f"   Status: ✅ Healthy")

    # Structures with most issues
    structures_by_issues = sorted(all_reports, key=lambda x: x['issues_count'], reverse=True)

    summary.append(f"\n{'='*80}")
    summary.append("⚠️  STRUCTURES REQUIRING ATTENTION")
    summary.append("="*80)

    structures_with_issues = [r for r in structures_by_issues if r['issues_count'] > 0]

    if structures_with_issues:
        summary.append(f"\nTop structures with issues:")
        for report in structures_with_issues[:10]:
            summary.append(f"\n   {report['structure']}: {report['issues_count']} issues")
            # List sensor-specific issues
            for sensor_id, sensor_info in report['sensors'].items():
                if sensor_info['issues']:
                    summary.append(f"      {sensor_id}: {', '.join(sensor_info['issues'][:3])}")
    else:
        summary.append(f"\n✅ All structures are healthy!")

    # Sensor type statistics
    summary.append(f"\n{'='*80}")
    summary.append("🎯 SENSOR TYPE STATISTICS")
    summary.append("="*80)

    sensor_types = {}
    for report in all_reports:
        for sensor_id, sensor_info in report['sensors'].items():
            stype = sensor_info['type']
            if stype not in sensor_types:
                sensor_types[stype] = {'count': 0, 'issues': 0}
            sensor_types[stype]['count'] += 1
            sensor_types[stype]['issues'] += len(sensor_info['issues'])

    for stype, stats in sorted(sensor_types.items()):
        summary.append(f"\n   {stype}:")
        summary.append(f"      Count: {stats['count']}")
        summary.append(f"      Issues: {stats['issues']}")

    summary.append(f"\n{'='*80}")
    summary.append("END OF SUMMARY")
    summary.append("="*80)

    # Save combined summary
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))

    print(f"\n✅ Combined summary saved: {summary_file}")

    print("\n" + "="*80)
    print("✅ ALL REPORTS GENERATED!")
    print("="*80)
    print(f"\n📁 Location: {report_dir}")
    print(f"   - Individual reports: {len(all_reports)} files")
    print(f"   - Combined summary: 00_COMBINED_SUMMARY.txt")
    print("="*80)


if __name__ == "__main__":
    generate_all_reports()
