import pandas as pd
import numpy as np
import os
from datetime import datetime

# Official Normal Value Ranges (as given by the company)
OFFICIAL_RANGES = {
    'accelerometer': {
        'p2p': {'min': -1.0, 'max': 1.0, 'unit': 'g'},
        'rms': {'min': -1.0, 'max': 1.0, 'unit': 'g'}
    },
    'displacement': {
        'value': {'min': 0.0, 'max': 500.0, 'unit': 'mm'}
    },
    'tilt': {
        'value': {'min': -5.0, 'max': 5.0, 'unit': 'degree'}
    },
    'temperature_probe': {
        'temperature': {'min': -40.0, 'max': 125.0, 'unit': '°C'}
    }
}


def validate_structure(structure_name, report_dir):
    """
    Validate structure data against official normal ranges.
    """
    merged_path = f"/Users/shawon/Downloads/Company Project/STR Data - Merged/{structure_name}_merged.xlsx"
    report_file = os.path.join(report_dir, f"{structure_name}_range_validation_report.txt")

    log = []

    def log_line(message):
        log.append(message)
        print(message)

    if not os.path.exists(merged_path):
        log_line(f"❌ File not found: {structure_name}_merged.xlsx")
        return None

    log_line("="*80)
    log_line(f"📊 RANGE VALIDATION REPORT: {structure_name}")
    log_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_line("="*80)

    # Load data
    print(f"⏳ Loading {structure_name}...")
    df = pd.read_excel(merged_path)
    log_line(f"\n✓ Loaded {len(df):,} rows")

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Overall info
    log_line(f"\n{'='*80}")
    log_line("📋 STRUCTURE INFORMATION")
    log_line("="*80)
    log_line(f"   Total rows: {len(df):,}")
    log_line(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    log_line(f"   Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    log_line(f"   Unique sensors: {df['sensor_id'].nunique()}")

    # Official ranges reference
    log_line(f"\n{'='*80}")
    log_line("📏 OFFICIAL NORMAL VALUE RANGES")
    log_line("="*80)
    log_line("\n   Accelerometer (AC):")
    log_line("      p2p, rms: -1.0 to +1.0 g")
    log_line("\n   Displacement Meter (DI):")
    log_line("      value: 0 to 500 mm")
    log_line("\n   Tiltmeter (TI):")
    log_line("      value: -5° to +5°")
    log_line("\n   Thermometer (TP):")
    log_line("      temperature: -40°C to +125°C")

    validation_results = {
        'structure': structure_name,
        'total_rows': len(df),
        'sensors': {},
        'total_violations': 0,
        'total_in_range': 0,
        'critical_sensors': []
    }

    # Per-sensor validation
    log_line(f"\n{'='*80}")
    log_line("🔍 PER-SENSOR RANGE VALIDATION")
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
            'measurements': {},
            'total_violations': 0,
            'total_checked': 0
        }

        # Get applicable ranges for this sensor type
        if sensor_type not in OFFICIAL_RANGES:
            log_line(f"\n   ⚠️  No official ranges defined for {sensor_type}")
            continue

        ranges = OFFICIAL_RANGES[sensor_type]

        log_line(f"\n   📊 Total readings: {len(sensor_df):,}")
        log_line(f"\n   🔍 RANGE VALIDATION RESULTS:")

        # Check each measurement column
        for column, range_spec in ranges.items():
            if column not in sensor_df.columns or sensor_df[column].isna().all():
                continue

            col_data = sensor_df[column].dropna()

            if len(col_data) == 0:
                continue

            min_range = range_spec['min']
            max_range = range_spec['max']
            unit = range_spec['unit']

            # Find violations
            violations = col_data[(col_data < min_range) | (col_data > max_range)]
            in_range = col_data[(col_data >= min_range) & (col_data <= max_range)]

            violation_count = len(violations)
            in_range_count = len(in_range)
            total_count = len(col_data)

            violation_pct = (violation_count / total_count * 100) if total_count > 0 else 0
            in_range_pct = (in_range_count / total_count * 100) if total_count > 0 else 0

            sensor_info['total_violations'] += violation_count
            sensor_info['total_checked'] += total_count

            # Store measurement info
            sensor_info['measurements'][column] = {
                'total': total_count,
                'in_range': in_range_count,
                'violations': violation_count,
                'violation_pct': violation_pct,
                'min_value': col_data.min(),
                'max_value': col_data.max(),
                'official_min': min_range,
                'official_max': max_range
            }

            log_line(f"\n      {column.upper()} (Official range: {min_range} to {max_range} {unit}):")
            log_line(f"         Total values: {total_count:,}")
            log_line(f"         ✅ In range: {in_range_count:,} ({in_range_pct:.2f}%)")
            log_line(f"         ❌ Out of range: {violation_count:,} ({violation_pct:.2f}%)")
            log_line(f"         Actual range: {col_data.min():.4f} to {col_data.max():.4f} {unit}")

            if violation_count > 0:
                # Severity analysis
                below_min = col_data[col_data < min_range]
                above_max = col_data[col_data > max_range]

                if len(below_min) > 0:
                    log_line(f"         ⬇️  Below minimum: {len(below_min):,}")
                    log_line(f"            Lowest value: {below_min.min():.4f} {unit}")
                    log_line(f"            Deviation: {abs(below_min.min() - min_range):.4f} {unit} below limit")

                if len(above_max) > 0:
                    log_line(f"         ⬆️  Above maximum: {len(above_max):,}")
                    log_line(f"            Highest value: {above_max.max():.4f} {unit}")
                    log_line(f"            Deviation: {abs(above_max.max() - max_range):.4f} {unit} above limit")

                # Temporal analysis of violations
                violation_indices = sensor_df[column][(sensor_df[column] < min_range) |
                                                       (sensor_df[column] > max_range)].index
                violation_timestamps = sensor_df.loc[violation_indices, 'timestamp']

                if len(violation_timestamps) > 0:
                    first_violation = violation_timestamps.min()
                    last_violation = violation_timestamps.max()

                    log_line(f"\n         📅 Violation Timeline:")
                    log_line(f"            First violation: {first_violation}")
                    log_line(f"            Last violation: {last_violation}")
                    log_line(f"            Duration: {(last_violation - first_violation).days} days")

                # Severity classification
                if violation_pct > 20:
                    log_line(f"\n         🚨 CRITICAL: Very high violation rate ({violation_pct:.1f}%)")
                    sensor_info['severity'] = 'CRITICAL'
                elif violation_pct > 10:
                    log_line(f"\n         ⚠️  WARNING: High violation rate ({violation_pct:.1f}%)")
                    sensor_info['severity'] = 'WARNING'
                elif violation_pct > 1:
                    log_line(f"\n         ⚠️  MODERATE: Noticeable violations ({violation_pct:.1f}%)")
                    sensor_info['severity'] = 'MODERATE'
                else:
                    log_line(f"\n         ℹ️  LOW: Minor violations ({violation_pct:.1f}%)")
                    sensor_info['severity'] = 'LOW'

            else:
                log_line(f"\n         ✅ EXCELLENT: All values within range!")

        # Sensor overall status
        log_line(f"\n   {'='*80}")
        log_line(f"   📊 SENSOR OVERALL STATUS:")
        log_line(f"   {'='*80}")

        if sensor_info['total_checked'] > 0:
            overall_violation_pct = (sensor_info['total_violations'] / sensor_info['total_checked'] * 100)
            log_line(f"      Total measurements checked: {sensor_info['total_checked']:,}")
            log_line(f"      Total violations: {sensor_info['total_violations']:,} ({overall_violation_pct:.2f}%)")

            if sensor_info['total_violations'] == 0:
                log_line(f"      ✅ ALL MEASUREMENTS WITHIN OFFICIAL RANGES")
            else:
                if overall_violation_pct > 10:
                    log_line(f"      🚨 REQUIRES IMMEDIATE INVESTIGATION")
                    validation_results['critical_sensors'].append(sensor_id)
                elif overall_violation_pct > 1:
                    log_line(f"      ⚠️  REQUIRES REVIEW")
                else:
                    log_line(f"      ℹ️  Minor issues detected")

        validation_results['sensors'][sensor_id] = sensor_info
        validation_results['total_violations'] += sensor_info['total_violations']
        validation_results['total_in_range'] += (sensor_info['total_checked'] - sensor_info['total_violations'])

    # Summary
    log_line(f"\n{'='*80}")
    log_line("📊 VALIDATION SUMMARY")
    log_line("="*80)

    total_checked = validation_results['total_violations'] + validation_results['total_in_range']

    log_line(f"\n   Structure: {structure_name}")
    log_line(f"   Total sensors: {len(sensors)}")
    log_line(f"   Total measurements checked: {total_checked:,}")
    log_line(f"   ✅ In range: {validation_results['total_in_range']:,} ({validation_results['total_in_range']/total_checked*100:.2f}%)")
    log_line(f"   ❌ Out of range: {validation_results['total_violations']:,} ({validation_results['total_violations']/total_checked*100:.2f}%)")

    if validation_results['critical_sensors']:
        log_line(f"\n   🚨 CRITICAL SENSORS ({len(validation_results['critical_sensors'])}):")
        for sensor_id in validation_results['critical_sensors']:
            sensor_info = validation_results['sensors'][sensor_id]
            overall_viol_pct = (sensor_info['total_violations'] / sensor_info['total_checked'] * 100)
            log_line(f"      {sensor_id}: {overall_viol_pct:.1f}% violations")

    log_line(f"\n   {'='*80}")
    if validation_results['total_violations'] == 0:
        log_line(f"   ✅ EXCELLENT: ALL DATA WITHIN OFFICIAL RANGES")
        log_line(f"   → Data quality is excellent for anomaly detection")
    elif len(validation_results['critical_sensors']) == 0:
        log_line(f"   ✅ GOOD: Minor violations detected")
        log_line(f"   → Review violations but data is usable")
    else:
        log_line(f"   ⚠️  ATTENTION REQUIRED")
        log_line(f"   → {len(validation_results['critical_sensors'])} sensors need investigation")
        log_line(f"   → Consider filtering or flagging out-of-range values")

    log_line(f"   {'='*80}")

    log_line("\n" + "="*80)

    # Save report
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log))

    print(f"✓ Report saved: {report_file}")

    return validation_results


def validate_all_structures(priority_structure='STR128'):
    """
    Validate all structures, with priority structure first.
    """
    merged_path = "/Users/shawon/Downloads/Company Project/STR Data - Merged"
    report_dir = "/Users/shawon/Downloads/Company Project/Range Validation Reports"

    os.makedirs(report_dir, exist_ok=True)

    # Get all structures
    all_structures = sorted([f.replace('_merged.xlsx', '')
                            for f in os.listdir(merged_path)
                            if f.endswith('_merged.xlsx')])

    # Move priority structure to front
    if priority_structure in all_structures:
        all_structures.remove(priority_structure)
        all_structures.insert(0, priority_structure)

    print("="*80)
    print("🚀 VALIDATING ALL STRUCTURES AGAINST OFFICIAL RANGES")
    print("="*80)
    print(f"\nFound {len(all_structures)} structures")
    print(f"Priority: {priority_structure}")
    print(f"Reports will be saved to: {report_dir}\n")

    all_results = []

    for idx, structure in enumerate(all_structures, 1):
        print(f"\n{'#'*80}")
        print(f"# {idx}/{len(all_structures)}: {structure}")
        if structure == priority_structure:
            print(f"# ⭐ PRIORITY STRUCTURE")
        print(f"{'#'*80}\n")

        result = validate_structure(structure, report_dir)
        if result:
            all_results.append(result)

    # Generate combined summary
    print(f"\n\n{'='*80}")
    print("📊 GENERATING COMBINED SUMMARY")
    print("="*80)

    summary_file = os.path.join(report_dir, "00_COMBINED_RANGE_VALIDATION_SUMMARY.txt")
    summary = []

    summary.append("="*80)
    summary.append("COMBINED RANGE VALIDATION SUMMARY - ALL STRUCTURES")
    summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("="*80)

    summary.append(f"\n{'='*80}")
    summary.append("📏 OFFICIAL NORMAL VALUE RANGES")
    summary.append("="*80)
    summary.append("\n   Accelerometer (AC): p2p, rms = -1.0 to +1.0 g")
    summary.append("   Displacement Meter (DI): value = 0 to 500 mm")
    summary.append("   Tiltmeter (TI): value = -5° to +5°")
    summary.append("   Thermometer (TP): temperature = -40°C to +125°C")

    # Overall statistics
    total_structures = len(all_results)
    total_sensors = sum(len(r['sensors']) for r in all_results)
    total_violations = sum(r['total_violations'] for r in all_results)
    total_in_range = sum(r['total_in_range'] for r in all_results)
    total_checked = total_violations + total_in_range

    summary.append(f"\n{'='*80}")
    summary.append("📊 OVERALL STATISTICS")
    summary.append("="*80)
    summary.append(f"\n   Structures validated: {total_structures}")
    summary.append(f"   Total sensors: {total_sensors}")
    summary.append(f"   Total measurements checked: {total_checked:,}")
    summary.append(f"   ✅ In range: {total_in_range:,} ({total_in_range/total_checked*100:.2f}%)")
    summary.append(f"   ❌ Out of range: {total_violations:,} ({total_violations/total_checked*100:.2f}%)")

    # Per-structure summary
    summary.append(f"\n{'='*80}")
    summary.append("📋 PER-STRUCTURE SUMMARY")
    summary.append("="*80)

    for result in all_results:
        total_checked_struct = result['total_violations'] + result['total_in_range']
        violation_pct = (result['total_violations'] / total_checked_struct * 100) if total_checked_struct > 0 else 0

        priority_marker = " ⭐" if result['structure'] == priority_structure else ""
        summary.append(f"\n{result['structure']}{priority_marker}:")
        summary.append(f"   Sensors: {len(result['sensors'])}")
        summary.append(f"   Measurements checked: {total_checked_struct:,}")
        summary.append(f"   ✅ In range: {result['total_in_range']:,} ({result['total_in_range']/total_checked_struct*100:.2f}%)")
        summary.append(f"   ❌ Out of range: {result['total_violations']:,} ({violation_pct:.2f}%)")

        if len(result['critical_sensors']) > 0:
            summary.append(f"   🚨 Critical sensors: {len(result['critical_sensors'])}")
            summary.append(f"      {', '.join(result['critical_sensors'])}")
            summary.append(f"   Status: ⚠️  Needs investigation")
        elif result['total_violations'] == 0:
            summary.append(f"   Status: ✅ Excellent")
        else:
            summary.append(f"   Status: ✅ Good (minor issues)")

    # Critical structures
    critical_structures = [r for r in all_results if len(r['critical_sensors']) > 0]

    if critical_structures:
        summary.append(f"\n{'='*80}")
        summary.append("🚨 STRUCTURES WITH CRITICAL SENSORS")
        summary.append("="*80)
        summary.append(f"\nThese structures have sensors with >10% out-of-range values:")

        for result in sorted(critical_structures, key=lambda x: len(x['critical_sensors']), reverse=True):
            summary.append(f"\n   {result['structure']}: {len(result['critical_sensors'])} critical sensor(s)")
            for sensor_id in result['critical_sensors']:
                sensor_info = result['sensors'][sensor_id]
                viol_pct = (sensor_info['total_violations'] / sensor_info['total_checked'] * 100)
                summary.append(f"      {sensor_id} ({sensor_info['type']}): {viol_pct:.1f}% violations")

    # Recommendations
    summary.append(f"\n{'='*80}")
    summary.append("📝 RECOMMENDATIONS")
    summary.append("="*80)

    violation_rate = (total_violations / total_checked * 100) if total_checked > 0 else 0

    summary.append(f"\n   Overall violation rate: {violation_rate:.2f}%")

    if violation_rate < 1:
        summary.append("\n   ✅ Excellent data quality:")
        summary.append("      - Less than 1% of data exceeds official ranges")
        summary.append("      - Data is suitable for anomaly detection")
        summary.append("      - Consider flagging violations as anomalies")
    elif violation_rate < 5:
        summary.append("\n   ⚠️  Good data quality with some issues:")
        summary.append("      - Minor violations detected")
        summary.append("      - Review critical sensors before training models")
        summary.append("      - Consider filtering extreme outliers")
    else:
        summary.append("\n   🚨 Data quality requires attention:")
        summary.append("      - Significant violations detected")
        summary.append("      - Investigate sensors with high violation rates")
        summary.append("      - May indicate sensor malfunctions or structural issues")
        summary.append("      - Consider preprocessing to remove/flag violations")

    summary.append(f"\n{'='*80}")
    summary.append("END OF VALIDATION SUMMARY")
    summary.append("="*80)

    # Save summary
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))

    print(f"\n✅ Combined summary saved: {summary_file}")

    print("\n" + "="*80)
    print("✅ ALL VALIDATION REPORTS GENERATED!")
    print("="*80)
    print(f"\n📁 Location: {report_dir}")
    print(f"   - Individual reports: {len(all_results)} files")
    print(f"   - Combined summary: 00_COMBINED_RANGE_VALIDATION_SUMMARY.txt")
    print(f"\n📊 Summary:")
    print(f"   Total measurements: {total_checked:,}")
    print(f"   In range: {total_in_range:,} ({total_in_range/total_checked*100:.2f}%)")
    print(f"   Out of range: {total_violations:,} ({total_violations/total_checked*100:.2f}%)")
    print(f"   Critical structures: {len(critical_structures)}")
    print("="*80)


if __name__ == "__main__":
    # Run validation with STR128 as priority
    validate_all_structures(priority_structure='STR128')
