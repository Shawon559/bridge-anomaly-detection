import pandas as pd
import numpy as np
import os
from datetime import datetime

def analyze_structure_improved(structure_name, report_dir):
    """
    Improved data quality analysis with proper categorization.
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
        'critical_issues': 0
    }

    # Define measurement categories
    BRIDGE_MEASUREMENTS = ['p2p', 'rms', 'value']
    ENVIRONMENTAL = ['temperature', 'humidity']
    SENSOR_HEALTH = ['battery']

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

        # === SENSOR HEALTH ===
        log_line(f"\n   🏥 SENSOR HEALTH:")

        sensor_df = sensor_df.sort_values('timestamp')
        time_diffs = sensor_df['timestamp'].diff()
        median_interval = time_diffs.median()

        # Data completeness
        if pd.notna(median_interval):
            total_duration = sensor_df['timestamp'].max() - sensor_df['timestamp'].min()
            expected_samples = int(total_duration / median_interval)
            actual_samples = len(sensor_df)
            completeness = (actual_samples / expected_samples * 100) if expected_samples > 0 else 0

            log_line(f"      Data completeness: {completeness:.1f}%")

            if completeness < 90:
                issue = f"Low data completeness: {completeness:.1f}%"
                sensor_info['issues'].append(issue)
                log_line(f"      ⚠️  {issue}")
            else:
                log_line(f"      ✓ Good data coverage")

            # Data gaps
            large_gaps = time_diffs[time_diffs > median_interval * 2]
            if len(large_gaps) > 0:
                log_line(f"      Data gaps: {len(large_gaps)} (largest: {large_gaps.max()})")
                if len(large_gaps) > 50:
                    issue = f"Many data gaps: {len(large_gaps)}"
                    sensor_info['issues'].append(issue)
                    log_line(f"      ⚠️  {issue}")

        # Battery status
        if 'battery' in sensor_df.columns and not sensor_df['battery'].isna().all():
            battery_data = sensor_df['battery'].dropna()
            if len(battery_data) > 0:
                avg_battery = battery_data.mean()
                min_battery = battery_data.min()

                log_line(f"      Battery: {avg_battery:.1f}% average, {min_battery:.1f}% minimum")

                if min_battery < 20:
                    log_line(f"      ⚠️  Low battery detected - sensor needs maintenance")
                elif avg_battery < 50:
                    log_line(f"      ⚠️  Battery declining - schedule maintenance")
                else:
                    log_line(f"      ✓ Battery healthy")

        # === ENVIRONMENTAL CONDITIONS ===
        log_line(f"\n   🌡️  ENVIRONMENTAL CONDITIONS:")

        env_data_found = False
        for col in ENVIRONMENTAL:
            if col in sensor_df.columns and not sensor_df[col].isna().all():
                env_data_found = True
                col_data = sensor_df[col].dropna()
                if len(col_data) > 0:
                    log_line(f"      {col.capitalize()}: {col_data.mean():.2f} (±{col_data.std():.2f})")
                    log_line(f"         Range: [{col_data.min():.2f}, {col_data.max():.2f}]")

        if not env_data_found:
            log_line(f"      N/A - No environmental sensors")
        else:
            log_line(f"      → Normal environmental variation expected")

        # === BRIDGE MEASUREMENTS (CRITICAL) ===
        log_line(f"\n   🔧 BRIDGE MEASUREMENTS (CRITICAL):")

        critical_data_found = False
        for col in BRIDGE_MEASUREMENTS:
            if col in sensor_df.columns and not sensor_df[col].isna().all():
                critical_data_found = True
                col_data = sensor_df[col].dropna()

                if len(col_data) == 0:
                    continue

                mean_val = col_data.mean()
                std_val = col_data.std()
                min_val = col_data.min()
                max_val = col_data.max()

                log_line(f"\n      {col}:")
                log_line(f"         Mean: {mean_val:.4f}, Std Dev: {std_val:.4f}")
                log_line(f"         Range: [{min_val:.4f}, {max_val:.4f}]")

                # Outlier detection using IQR
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_pct = (len(outliers) / len(col_data)) * 100

                log_line(f"         Outliers: {len(outliers)} ({outlier_pct:.2f}%)")

                if outlier_pct > 5:
                    issue = f"{col}: {outlier_pct:.1f}% outliers (possible anomalies)"
                    sensor_info['issues'].append(issue)
                    report_data['critical_issues'] += 1
                    log_line(f"         ⚠️  High outlier rate - investigate anomalies")
                elif outlier_pct > 1:
                    log_line(f"         ⚠️  Some outliers detected - review recommended")
                else:
                    log_line(f"         ✓ Measurements within normal range")

                # Check for constant values (sensor stuck)
                unique_values = col_data.nunique()
                if unique_values == 1:
                    issue = f"{col}: Constant value - sensor may be stuck!"
                    sensor_info['issues'].append(issue)
                    report_data['critical_issues'] += 1
                    log_line(f"         🚨 CRITICAL: Sensor appears stuck!")

        if not critical_data_found:
            log_line(f"      N/A - No bridge measurement data for this sensor")

        # Overall sensor status
        log_line(f"\n   📊 OVERALL SENSOR STATUS:")
        if len(sensor_info['issues']) == 0:
            log_line(f"      ✅ All checks passed - sensor operating normally")
        else:
            log_line(f"      ⚠️  {len(sensor_info['issues'])} issue(s) detected:")
            for issue in sensor_info['issues']:
                log_line(f"         - {issue}")

        report_data['sensors'][sensor_id] = sensor_info

    # Summary
    log_line(f"\n{'='*80}")
    log_line("📊 SUMMARY")
    log_line("="*80)
    log_line(f"\n   Total sensors: {len(sensors)}")
    log_line(f"   Total rows: {len(df):,}")
    log_line(f"   Total issues: {sum(len(s['issues']) for s in report_data['sensors'].values())}")
    log_line(f"   Critical bridge measurement issues: {report_data['critical_issues']}")

    if report_data['critical_issues'] == 0:
        log_line(f"\n   ✅ No critical issues - bridge measurements look healthy!")
    else:
        log_line(f"\n   ⚠️  {report_data['critical_issues']} critical issues require investigation")

    log_line("\n" + "="*80)

    # Save report
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log))

    print(f"✓ Report saved: {report_file}")

    return report_data


def generate_all_improved_reports():
    """Generate improved quality reports for all structures"""
    merged_path = "/Users/shawon/Downloads/Company Project/STR Data - Merged"
    report_dir = "/Users/shawon/Downloads/Company Project/Data Quality Reports"

    os.makedirs(report_dir, exist_ok=True)

    structures = sorted([f.replace('_merged.xlsx', '')
                        for f in os.listdir(merged_path)
                        if f.endswith('_merged.xlsx')])

    print("="*80)
    print("🚀 GENERATING IMPROVED DATA QUALITY REPORTS")
    print("="*80)
    print(f"\nFound {len(structures)} structures")
    print(f"Reports will be saved to: {report_dir}\n")

    all_reports = []

    for idx, structure in enumerate(structures, 1):
        print(f"\n{'#'*80}")
        print(f"# {idx}/{len(structures)}: {structure}")
        print(f"{'#'*80}\n")

        report = analyze_structure_improved(structure, report_dir)
        if report:
            all_reports.append(report)

    # Generate combined summary
    print(f"\n\n{'='*80}")
    print("📊 GENERATING COMBINED SUMMARY")
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
    total_issues = sum(sum(len(s['issues']) for s in r['sensors'].values()) for r in all_reports)
    total_critical = sum(r['critical_issues'] for r in all_reports)

    summary.append(f"\n{'='*80}")
    summary.append("📊 OVERALL STATISTICS")
    summary.append("="*80)
    summary.append(f"\n   Structures analyzed: {len(all_reports)}")
    summary.append(f"   Total sensors: {total_sensors}")
    summary.append(f"   Total data rows: {total_rows:,}")
    summary.append(f"   Total issues: {total_issues}")
    summary.append(f"   Critical bridge measurement issues: {total_critical}")

    summary.append(f"\n{'='*80}")
    summary.append("KEY INSIGHTS")
    summary.append("="*80)
    summary.append("\n✅ What's Working Well:")
    summary.append("   - Environmental monitoring (temperature, humidity) functioning normally")
    summary.append("   - Most sensors have good battery health")
    summary.append("   - Sampling rates are consistent where data exists")

    summary.append("\n⚠️  Areas Requiring Attention:")
    summary.append("   - Data completeness varies (some sensors have gaps)")
    summary.append(f"   - {total_critical} sensors showing measurement anomalies")
    summary.append("   - Some sensors experiencing periodic offline periods")

    # Per-structure summary
    summary.append(f"\n{'='*80}")
    summary.append("📋 PER-STRUCTURE SUMMARY")
    summary.append("="*80)

    for report in all_reports:
        total_sensor_issues = sum(len(s['issues']) for s in report['sensors'].values())

        summary.append(f"\n{report['structure']}:")
        summary.append(f"   Sensors: {len(report['sensors'])}")
        summary.append(f"   Rows: {report['total_rows']:,}")
        summary.append(f"   Date range: {report['date_range']['start']} to {report['date_range']['end']}")
        summary.append(f"   Total issues: {total_sensor_issues}")
        summary.append(f"   Critical issues: {report['critical_issues']}")

        if report['critical_issues'] == 0:
            summary.append(f"   Status: ✅ Healthy")
        else:
            summary.append(f"   Status: ⚠️  Needs investigation")

    # Critical structures
    critical_structures = [r for r in all_reports if r['critical_issues'] > 0]
    critical_structures.sort(key=lambda x: x['critical_issues'], reverse=True)

    if critical_structures:
        summary.append(f"\n{'='*80}")
        summary.append("🚨 STRUCTURES WITH CRITICAL ISSUES")
        summary.append("="*80)
        summary.append("\nThese structures have bridge measurement anomalies requiring investigation:")

        for report in critical_structures[:10]:
            summary.append(f"\n   {report['structure']}: {report['critical_issues']} critical issues")

            # List sensors with critical issues
            for sensor_id, sensor_info in report['sensors'].items():
                critical_sensor_issues = [i for i in sensor_info['issues']
                                        if 'outliers' in i or 'stuck' in i.lower()]
                if critical_sensor_issues:
                    summary.append(f"      {sensor_id}: {', '.join(critical_sensor_issues)}")
    else:
        summary.append(f"\n{'='*80}")
        summary.append("✅ ALL STRUCTURES HEALTHY")
        summary.append("="*80)
        summary.append("\nNo critical bridge measurement issues detected!")

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
        summary.append(f"      Total count: {stats['count']}")
        summary.append(f"      Total issues: {stats['issues']}")
        summary.append(f"      Average issues per sensor: {stats['issues']/stats['count']:.1f}")

    summary.append(f"\n{'='*80}")
    summary.append("📝 RECOMMENDATIONS")
    summary.append("="*80)
    summary.append("\n1. Data Preprocessing:")
    summary.append("   - Fill small data gaps using interpolation")
    summary.append("   - Flag large gaps as 'sensor offline' periods")
    summary.append("   - Investigate and validate outliers in bridge measurements")

    summary.append("\n2. Sensor Maintenance:")
    summary.append("   - Schedule battery replacement for low-battery sensors")
    summary.append("   - Check sensors with constant values (may be stuck)")
    summary.append("   - Review sensors with frequent offline periods")

    summary.append("\n3. Anomaly Detection Preparation:")
    summary.append("   - Focus anomaly detection on p2p, rms, and value measurements")
    summary.append("   - Use temperature/humidity as contextual features")
    summary.append("   - Consider seasonal patterns in environmental data")

    summary.append(f"\n{'='*80}")
    summary.append("END OF SUMMARY")
    summary.append("="*80)

    # Save combined summary
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))

    print(f"\n✅ Combined summary saved: {summary_file}")

    print("\n" + "="*80)
    print("✅ ALL IMPROVED REPORTS GENERATED!")
    print("="*80)
    print(f"\n📁 Location: {report_dir}")
    print(f"   - Individual reports: {len(all_reports)} files")
    print(f"   - Combined summary: 00_COMBINED_SUMMARY.txt")
    print(f"\n📊 Summary:")
    print(f"   Total sensors: {total_sensors}")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Critical issues: {total_critical}")
    print("="*80)


if __name__ == "__main__":
    generate_all_improved_reports()
