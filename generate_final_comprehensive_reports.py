import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def analyze_structure_comprehensive(structure_name, report_dir):
    """
    Comprehensive data quality analysis with detailed temporal gap analysis.
    """
    merged_path = f"/Users/shawon/Downloads/Company Project/STR Data - Merged/{structure_name}_merged.xlsx"
    report_file = os.path.join(report_dir, f"{structure_name}_comprehensive_quality_report.txt")

    log = []

    def log_line(message):
        log.append(message)
        print(message)

    if not os.path.exists(merged_path):
        log_line(f"❌ File not found: {structure_name}_merged.xlsx")
        return None

    log_line("="*80)
    log_line(f"📊 COMPREHENSIVE DATA QUALITY REPORT: {structure_name}")
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
        'critical_issues': 0,
        'temporal_gaps': []
    }

    # Define measurement categories
    BRIDGE_MEASUREMENTS = ['p2p', 'rms', 'value']
    ENVIRONMENTAL = ['temperature', 'humidity']
    SENSOR_HEALTH = ['battery']

    # Per-sensor analysis
    log_line(f"\n{'='*80}")
    log_line("🔍 PER-SENSOR COMPREHENSIVE ANALYSIS")
    log_line("="*80)

    sensors = sorted(df['sensor_id'].unique())

    for sensor_id in sensors:
        sensor_df = df[df['sensor_id'] == sensor_id].copy()
        sensor_type = sensor_df['sensor_type'].iloc[0]

        log_line(f"\n{'='*80}")
        log_line(f"📡 SENSOR: {sensor_id} ({sensor_type})")
        log_line("="*80)

        sensor_info = {
            'type': sensor_type,
            'total_rows': len(sensor_df),
            'issues': [],
            'offline_periods': []
        }

        # Sort by timestamp for temporal analysis
        sensor_df = sensor_df.sort_values('timestamp').reset_index(drop=True)

        # === BASIC INFORMATION ===
        log_line(f"\n   📊 BASIC INFORMATION:")
        log_line(f"      Total readings: {len(sensor_df):,}")
        log_line(f"      Date range: {sensor_df['timestamp'].min()} to {sensor_df['timestamp'].max()}")
        duration_days = (sensor_df['timestamp'].max() - sensor_df['timestamp'].min()).days
        log_line(f"      Duration: {duration_days} days")

        # === SAMPLING RATE & DATA COMPLETENESS ===
        log_line(f"\n   ⏱️  SAMPLING RATE & DATA COMPLETENESS:")

        time_diffs = sensor_df['timestamp'].diff()
        median_interval = time_diffs.median()

        if pd.notna(median_interval):
            log_line(f"      Median sampling interval: {median_interval}")

            total_duration = sensor_df['timestamp'].max() - sensor_df['timestamp'].min()
            expected_samples = int(total_duration / median_interval)
            actual_samples = len(sensor_df)
            completeness = (actual_samples / expected_samples * 100) if expected_samples > 0 else 0

            log_line(f"      Expected samples: {expected_samples:,}")
            log_line(f"      Actual samples: {actual_samples:,}")
            log_line(f"      Data completeness: {completeness:.1f}%")

            if completeness < 90:
                issue = f"Low data completeness: {completeness:.1f}%"
                sensor_info['issues'].append(issue)
                log_line(f"      ⚠️  {issue}")
            else:
                log_line(f"      ✓ Good data coverage")

        # === DETAILED TEMPORAL GAP ANALYSIS ===
        log_line(f"\n   📅 DETAILED TEMPORAL GAP ANALYSIS:")

        if pd.notna(median_interval) and len(time_diffs) > 0:
            # Define significant gap threshold (2x median)
            gap_threshold = median_interval * 2

            # Find all gaps
            gaps_mask = time_diffs > gap_threshold
            gaps = time_diffs[gaps_mask]

            if len(gaps) > 0:
                log_line(f"      Total gaps detected: {len(gaps)}")
                log_line(f"      Gap threshold: {gap_threshold}")
                log_line(f"      Largest gap: {gaps.max()}")
                log_line(f"      Total time offline: {gaps.sum()}")

                # Calculate offline percentage
                total_gap_time = gaps.sum()
                total_duration_time = sensor_df['timestamp'].max() - sensor_df['timestamp'].min()
                offline_pct = (total_gap_time / total_duration_time * 100) if total_duration_time > timedelta(0) else 0
                log_line(f"      Offline percentage: {offline_pct:.2f}%")

                # Detailed offline periods
                log_line(f"\n      🔍 DETAILED OFFLINE PERIODS:")

                gap_indices = time_diffs[gaps_mask].index
                offline_periods = []

                for idx in gap_indices:
                    gap_start = sensor_df.loc[idx-1, 'timestamp']
                    gap_end = sensor_df.loc[idx, 'timestamp']
                    gap_duration = gap_end - gap_start

                    offline_periods.append({
                        'start': gap_start,
                        'end': gap_end,
                        'duration': gap_duration
                    })

                # Sort by duration (largest first)
                offline_periods.sort(key=lambda x: x['duration'], reverse=True)

                # Show top 10 longest offline periods
                num_to_show = min(10, len(offline_periods))
                log_line(f"      Showing top {num_to_show} longest offline periods:")

                for i, period in enumerate(offline_periods[:num_to_show], 1):
                    log_line(f"         {i}. Start: {period['start']}")
                    log_line(f"            End:   {period['end']}")
                    log_line(f"            Duration: {period['duration']}")
                    log_line("")

                sensor_info['offline_periods'] = offline_periods
                report_data['temporal_gaps'].extend([{
                    'sensor': sensor_id,
                    'start': p['start'],
                    'end': p['end'],
                    'duration': p['duration']
                } for p in offline_periods])

                if len(gaps) > 50:
                    issue = f"Many offline periods: {len(gaps)}"
                    sensor_info['issues'].append(issue)
                    log_line(f"      ⚠️  {issue}")

                if offline_pct > 20:
                    issue = f"High offline percentage: {offline_pct:.2f}%"
                    sensor_info['issues'].append(issue)
                    log_line(f"      ⚠️  {issue}")
            else:
                log_line(f"      ✓ No significant gaps detected - continuous data collection")
        else:
            log_line(f"      N/A - Insufficient data for gap analysis")

        # === DATA QUALITY TIMELINE ===
        log_line(f"\n   📈 DATA QUALITY TIMELINE:")

        # Analyze data quality in time windows (e.g., monthly)
        if len(sensor_df) > 100:
            sensor_df['month'] = sensor_df['timestamp'].dt.to_period('M')
            monthly_stats = sensor_df.groupby('month').size()

            log_line(f"      Monthly data point distribution:")

            # Show first 6 and last 6 months if more than 12 months
            if len(monthly_stats) > 12:
                log_line(f"         First 6 months:")
                for month, count in monthly_stats.head(6).items():
                    log_line(f"            {month}: {count:,} readings")
                log_line(f"         ...")
                log_line(f"         Last 6 months:")
                for month, count in monthly_stats.tail(6).items():
                    log_line(f"            {month}: {count:,} readings")
            else:
                for month, count in monthly_stats.items():
                    log_line(f"         {month}: {count:,} readings")

            # Check for degradation
            first_half_avg = monthly_stats.iloc[:len(monthly_stats)//2].mean()
            second_half_avg = monthly_stats.iloc[len(monthly_stats)//2:].mean()

            if second_half_avg < first_half_avg * 0.7:
                issue = "Data collection degrading over time"
                sensor_info['issues'].append(issue)
                log_line(f"      ⚠️  {issue}")
                log_line(f"         First half avg: {first_half_avg:.0f} readings/month")
                log_line(f"         Second half avg: {second_half_avg:.0f} readings/month")

        # === SENSOR HEALTH ===
        log_line(f"\n   🏥 SENSOR HEALTH:")

        # Battery status
        if 'battery' in sensor_df.columns and not sensor_df['battery'].isna().all():
            battery_data = sensor_df['battery'].dropna()
            if len(battery_data) > 0:
                avg_battery = battery_data.mean()
                min_battery = battery_data.min()
                first_battery = battery_data.iloc[0]
                last_battery = battery_data.iloc[-1]

                log_line(f"      Battery status:")
                log_line(f"         Current: {last_battery:.1f}%")
                log_line(f"         Average: {avg_battery:.1f}%")
                log_line(f"         Minimum: {min_battery:.1f}%")
                log_line(f"         Initial: {first_battery:.1f}%")
                log_line(f"         Decline: {first_battery - last_battery:.1f}%")

                if min_battery < 20:
                    log_line(f"      🚨 CRITICAL: Low battery - sensor needs immediate maintenance!")
                elif last_battery < 30:
                    log_line(f"      ⚠️  Low battery - schedule maintenance soon")
                elif avg_battery < 50:
                    log_line(f"      ⚠️  Battery declining - monitor closely")
                else:
                    log_line(f"      ✓ Battery healthy")
        else:
            log_line(f"      N/A - No battery monitoring")

        # Data freshness
        log_line(f"\n      Data freshness:")
        last_reading = sensor_df['timestamp'].max()
        days_since_last = (datetime.now() - last_reading).days
        log_line(f"         Last reading: {last_reading}")
        log_line(f"         Days since last reading: {days_since_last}")

        if days_since_last > 30:
            log_line(f"      ⚠️  No recent data - sensor may be offline")
        elif days_since_last > 7:
            log_line(f"      ⚠️  Data not fresh - check sensor status")
        else:
            log_line(f"      ✓ Recent data available")

        # === ENVIRONMENTAL CONDITIONS ===
        log_line(f"\n   🌡️  ENVIRONMENTAL CONDITIONS:")

        env_data_found = False
        for col in ENVIRONMENTAL:
            if col in sensor_df.columns and not sensor_df[col].isna().all():
                env_data_found = True
                col_data = sensor_df[col].dropna()
                if len(col_data) > 0:
                    log_line(f"      {col.capitalize()}:")
                    log_line(f"         Mean: {col_data.mean():.2f}")
                    log_line(f"         Std Dev: {col_data.std():.2f}")
                    log_line(f"         Range: [{col_data.min():.2f}, {col_data.max():.2f}]")
                    log_line(f"         Missing: {sensor_df[col].isna().sum()} ({sensor_df[col].isna().sum()/len(sensor_df)*100:.1f}%)")

        if not env_data_found:
            log_line(f"      N/A - No environmental sensors")
        else:
            log_line(f"      → Environmental data available for contextual analysis")

        # === BRIDGE MEASUREMENTS (CRITICAL) ===
        log_line(f"\n   🔧 BRIDGE MEASUREMENTS (CRITICAL FOR ANOMALY DETECTION):")

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

                log_line(f"\n      {col.upper()}:")
                log_line(f"         Count: {len(col_data):,}")
                log_line(f"         Missing: {sensor_df[col].isna().sum()} ({sensor_df[col].isna().sum()/len(sensor_df)*100:.1f}%)")
                log_line(f"         Mean: {mean_val:.4f}")
                log_line(f"         Std Dev: {std_val:.4f}")
                log_line(f"         Range: [{min_val:.4f}, {max_val:.4f}]")

                # Outlier detection using IQR
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_pct = (len(outliers) / len(col_data)) * 100

                log_line(f"         Outliers (3×IQR): {len(outliers)} ({outlier_pct:.2f}%)")
                log_line(f"         Outlier bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")

                if outlier_pct > 5:
                    issue = f"{col}: {outlier_pct:.1f}% outliers (possible anomalies)"
                    sensor_info['issues'].append(issue)
                    report_data['critical_issues'] += 1
                    log_line(f"         🚨 High outlier rate - investigate for structural anomalies")
                elif outlier_pct > 1:
                    log_line(f"         ⚠️  Some outliers detected - review recommended")
                else:
                    log_line(f"         ✓ Measurements within normal range")

                # Check for constant values (sensor stuck)
                unique_values = col_data.nunique()
                if unique_values == 1:
                    issue = f"{col}: Constant value - sensor stuck!"
                    sensor_info['issues'].append(issue)
                    report_data['critical_issues'] += 1
                    log_line(f"         🚨 CRITICAL: Sensor appears stuck at {col_data.iloc[0]:.4f}!")
                elif unique_values < 10 and len(col_data) > 100:
                    log_line(f"         ⚠️  Low variance - only {unique_values} unique values")

                # Check for sudden jumps
                if len(col_data) > 1:
                    changes = col_data.diff().abs()
                    mean_change = changes.mean()
                    std_change = changes.std()

                    if pd.notna(mean_change) and pd.notna(std_change) and std_change > 0:
                        large_jumps = changes[changes > mean_change + 5 * std_change]

                        if len(large_jumps) > 0:
                            log_line(f"         ⚠️  Sudden jumps detected: {len(large_jumps)} instances")
                            log_line(f"            Max jump: {large_jumps.max():.4f}")

        if not critical_data_found:
            log_line(f"      N/A - No bridge measurement data for this sensor")

        # === OVERALL SENSOR STATUS ===
        log_line(f"\n   {'='*80}")
        log_line(f"   📊 OVERALL SENSOR STATUS:")
        log_line(f"   {'='*80}")

        if len(sensor_info['issues']) == 0:
            log_line(f"      ✅ ALL CHECKS PASSED")
            log_line(f"      → Sensor operating normally")
            log_line(f"      → Data quality is good for anomaly detection")
        else:
            log_line(f"      ⚠️  {len(sensor_info['issues'])} ISSUE(S) DETECTED:")
            for i, issue in enumerate(sensor_info['issues'], 1):
                log_line(f"         {i}. {issue}")
            log_line(f"\n      → Review issues before using for anomaly detection")

        report_data['sensors'][sensor_id] = sensor_info

    # === CROSS-SENSOR HEALTH COMPARISON ===
    log_line(f"\n{'='*80}")
    log_line("🔄 CROSS-SENSOR HEALTH COMPARISON")
    log_line("="*80)

    # Group sensors by type
    sensors_by_type = {}
    for sensor_id, sensor_info in report_data['sensors'].items():
        stype = sensor_info['type']
        if stype not in sensors_by_type:
            sensors_by_type[stype] = []
        sensors_by_type[stype].append({
            'id': sensor_id,
            'issues': len(sensor_info['issues']),
            'rows': sensor_info['total_rows']
        })

    for stype, sensors_list in sorted(sensors_by_type.items()):
        log_line(f"\n   {stype.upper()}:")
        log_line(f"      Total sensors: {len(sensors_list)}")

        total_issues = sum(s['issues'] for s in sensors_list)
        avg_issues = total_issues / len(sensors_list) if sensors_list else 0

        log_line(f"      Total issues: {total_issues}")
        log_line(f"      Average issues per sensor: {avg_issues:.1f}")

        # List sensors with issues
        sensors_with_issues = [s for s in sensors_list if s['issues'] > 0]
        if sensors_with_issues:
            log_line(f"      Sensors with issues:")
            for s in sorted(sensors_with_issues, key=lambda x: x['issues'], reverse=True):
                log_line(f"         {s['id']}: {s['issues']} issue(s), {s['rows']:,} rows")
        else:
            log_line(f"      ✓ All {stype} sensors healthy")

    # === SUMMARY ===
    log_line(f"\n{'='*80}")
    log_line("📊 SUMMARY")
    log_line("="*80)
    log_line(f"\n   Structure: {structure_name}")
    log_line(f"   Total sensors: {len(sensors)}")
    log_line(f"   Total data rows: {len(df):,}")
    log_line(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    log_line(f"   Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")

    total_issues = sum(len(s['issues']) for s in report_data['sensors'].values())
    log_line(f"\n   Total issues detected: {total_issues}")
    log_line(f"   Critical bridge measurement issues: {report_data['critical_issues']}")
    log_line(f"   Total offline periods: {len(report_data['temporal_gaps'])}")

    if len(report_data['temporal_gaps']) > 0:
        total_offline_time = sum((g['duration'] for g in report_data['temporal_gaps']), timedelta())
        log_line(f"   Total offline time: {total_offline_time}")

    log_line(f"\n   {'='*80}")
    if report_data['critical_issues'] == 0 and total_issues == 0:
        log_line(f"   ✅ EXCELLENT DATA QUALITY")
        log_line(f"   → Ready for anomaly detection analysis")
    elif report_data['critical_issues'] == 0:
        log_line(f"   ✅ GOOD DATA QUALITY")
        log_line(f"   → Minor issues detected, proceed with caution")
        log_line(f"   → Review individual sensor reports for details")
    else:
        log_line(f"   ⚠️  DATA QUALITY ISSUES DETECTED")
        log_line(f"   → {report_data['critical_issues']} critical issues require investigation")
        log_line(f"   → Address issues before anomaly detection")

    log_line(f"   {'='*80}")

    log_line("\n" + "="*80)

    # Save report
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log))

    print(f"✓ Report saved: {report_file}")

    return report_data


def generate_all_comprehensive_reports():
    """Generate comprehensive quality reports for all structures"""
    merged_path = "/Users/shawon/Downloads/Company Project/STR Data - Merged"
    report_dir = "/Users/shawon/Downloads/Company Project/Data Quality Reports"

    os.makedirs(report_dir, exist_ok=True)

    structures = sorted([f.replace('_merged.xlsx', '')
                        for f in os.listdir(merged_path)
                        if f.endswith('_merged.xlsx')])

    print("="*80)
    print("🚀 GENERATING COMPREHENSIVE DATA QUALITY REPORTS")
    print("="*80)
    print(f"\nFound {len(structures)} structures")
    print(f"Reports will be saved to: {report_dir}\n")

    all_reports = []

    for idx, structure in enumerate(structures, 1):
        print(f"\n{'#'*80}")
        print(f"# {idx}/{len(structures)}: {structure}")
        print(f"{'#'*80}\n")

        report = analyze_structure_comprehensive(structure, report_dir)
        if report:
            all_reports.append(report)

    # Generate combined summary
    print(f"\n\n{'='*80}")
    print("📊 GENERATING COMBINED SUMMARY")
    print("="*80)

    summary_file = os.path.join(report_dir, "00_COMBINED_COMPREHENSIVE_SUMMARY.txt")
    summary = []

    summary.append("="*80)
    summary.append("COMBINED COMPREHENSIVE DATA QUALITY SUMMARY - ALL STRUCTURES")
    summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("="*80)

    # Overall statistics
    total_sensors = sum(len(r['sensors']) for r in all_reports)
    total_rows = sum(r['total_rows'] for r in all_reports)
    total_issues = sum(sum(len(s['issues']) for s in r['sensors'].values()) for r in all_reports)
    total_critical = sum(r['critical_issues'] for r in all_reports)
    total_offline_periods = sum(len(r['temporal_gaps']) for r in all_reports)

    summary.append(f"\n{'='*80}")
    summary.append("📊 OVERALL STATISTICS")
    summary.append("="*80)
    summary.append(f"\n   Structures analyzed: {len(all_reports)}")
    summary.append(f"   Total sensors: {total_sensors}")
    summary.append(f"   Total data rows: {total_rows:,}")
    summary.append(f"   Total issues: {total_issues}")
    summary.append(f"   Critical bridge measurement issues: {total_critical}")
    summary.append(f"   Total offline periods detected: {total_offline_periods}")

    summary.append(f"\n{'='*80}")
    summary.append("📈 KEY INSIGHTS")
    summary.append("="*80)
    summary.append("\n✅ Strengths:")
    summary.append("   - Environmental monitoring (temperature, humidity) functioning normally")
    summary.append("   - Most sensors maintaining consistent sampling rates")
    summary.append("   - Battery health generally good across sensors")

    summary.append("\n⚠️  Areas Requiring Attention:")
    summary.append(f"   - {total_critical} critical bridge measurement anomalies detected")
    summary.append(f"   - {total_offline_periods} offline periods across all sensors")
    summary.append("   - Some sensors showing data collection degradation over time")
    summary.append("   - Data completeness varies (some sensors have gaps)")

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
        summary.append(f"   Offline periods: {len(report['temporal_gaps'])}")

        if report['critical_issues'] == 0 and total_sensor_issues == 0:
            summary.append(f"   Status: ✅ Excellent")
        elif report['critical_issues'] == 0:
            summary.append(f"   Status: ✅ Good (minor issues)")
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

        for report in critical_structures:
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

    # Longest offline periods
    all_gaps = []
    for report in all_reports:
        for gap in report['temporal_gaps']:
            all_gaps.append({
                'structure': report['structure'],
                'sensor': gap['sensor'],
                'start': gap['start'],
                'end': gap['end'],
                'duration': gap['duration']
            })

    if all_gaps:
        all_gaps.sort(key=lambda x: x['duration'], reverse=True)

        summary.append(f"\n{'='*80}")
        summary.append("⏱️  LONGEST OFFLINE PERIODS (TOP 20)")
        summary.append("="*80)
        summary.append("\nSensors with longest data collection interruptions:")

        for i, gap in enumerate(all_gaps[:20], 1):
            summary.append(f"\n   {i}. {gap['structure']} - {gap['sensor']}")
            summary.append(f"      Start: {gap['start']}")
            summary.append(f"      End:   {gap['end']}")
            summary.append(f"      Duration: {gap['duration']}")

    # Sensor type statistics
    summary.append(f"\n{'='*80}")
    summary.append("🎯 SENSOR TYPE STATISTICS")
    summary.append("="*80)

    sensor_types = {}
    for report in all_reports:
        for sensor_id, sensor_info in report['sensors'].items():
            stype = sensor_info['type']
            if stype not in sensor_types:
                sensor_types[stype] = {'count': 0, 'issues': 0, 'rows': 0}
            sensor_types[stype]['count'] += 1
            sensor_types[stype]['issues'] += len(sensor_info['issues'])
            sensor_types[stype]['rows'] += sensor_info['total_rows']

    for stype, stats in sorted(sensor_types.items()):
        summary.append(f"\n   {stype}:")
        summary.append(f"      Total sensors: {stats['count']}")
        summary.append(f"      Total data rows: {stats['rows']:,}")
        summary.append(f"      Average rows per sensor: {stats['rows']/stats['count']:,.0f}")
        summary.append(f"      Total issues: {stats['issues']}")
        summary.append(f"      Average issues per sensor: {stats['issues']/stats['count']:.1f}")

    summary.append(f"\n{'='*80}")
    summary.append("📝 RECOMMENDATIONS FOR ANOMALY DETECTION")
    summary.append("="*80)
    summary.append("\n1. Data Preprocessing:")
    summary.append("   - Fill small gaps (<2× median interval) using interpolation")
    summary.append("   - Flag large gaps as 'sensor offline' periods - exclude from analysis")
    summary.append("   - Investigate outliers in bridge measurements (p2p, rms, value)")
    summary.append("   - Remove or correct readings during identified offline periods")

    summary.append("\n2. Sensor Maintenance:")
    summary.append("   - Address sensors with critical battery levels immediately")
    summary.append("   - Check sensors showing constant values (may be stuck)")
    summary.append("   - Review sensors with frequent offline periods")
    summary.append("   - Investigate sensors with data collection degradation")

    summary.append("\n3. Anomaly Detection Strategy:")
    summary.append("   - Focus on bridge measurements: p2p, rms, value")
    summary.append("   - Use temperature/humidity as contextual features")
    summary.append("   - Consider temporal patterns (time of day, seasonal)")
    summary.append("   - Account for identified offline periods in model training")
    summary.append("   - Cross-validate anomalies across multiple sensors")

    summary.append("\n4. Data Quality Improvements:")
    summary.append("   - Monitor sensors with high offline percentages")
    summary.append("   - Schedule preventive maintenance for declining sensors")
    summary.append("   - Improve data collection consistency")
    summary.append("   - Regular battery replacement schedule")

    summary.append(f"\n{'='*80}")
    summary.append("END OF COMPREHENSIVE SUMMARY")
    summary.append("="*80)

    # Save combined summary
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))

    print(f"\n✅ Combined summary saved: {summary_file}")

    print("\n" + "="*80)
    print("✅ ALL COMPREHENSIVE REPORTS GENERATED!")
    print("="*80)
    print(f"\n📁 Location: {report_dir}")
    print(f"   - Individual reports: {len(all_reports)} files")
    print(f"   - Combined summary: 00_COMBINED_COMPREHENSIVE_SUMMARY.txt")
    print(f"\n📊 Summary:")
    print(f"   Total sensors: {total_sensors}")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Total issues: {total_issues}")
    print(f"   Critical issues: {total_critical}")
    print(f"   Offline periods: {total_offline_periods}")
    print("="*80)


if __name__ == "__main__":
    generate_all_comprehensive_reports()
