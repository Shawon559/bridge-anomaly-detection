import os
from datetime import datetime

def parse_merge_log(log_file):
    """Parse a merge log file to extract key information."""
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    info = {
        'structure': None,
        'total_files': 0,
        'total_sheets': 0,
        'before_merge': 0,
        'after_merge': 0,
        'duplicates_removed': 0,
        'sensors': [],
        'date_range_start': None,
        'date_range_end': None,
        'duration_days': 0,
        'file_size_mb': 0,
        'success': False
    }

    lines = content.split('\n')

    for i, line in enumerate(lines):
        # Extract structure name
        if 'MERGING STRUCTURE:' in line:
            info['structure'] = line.split('MERGING STRUCTURE:')[1].strip()

        # Extract file count
        if 'Files to process:' in line:
            try:
                info['total_files'] = int(line.split(':')[1].strip())
            except:
                pass

        # Extract sheets from Found X sheets
        if 'Found' in line and 'sheets:' in line:
            info['total_sheets'] += 1

        # Extract sensor IDs (lines like "AC383 (accelerometer):")
        if ('(accelerometer):' in line or '(displacement):' in line or
            '(tilt):' in line or '(temperature_probe):' in line):
            sensor_id = line.split('(')[0].strip()
            if sensor_id and sensor_id not in info['sensors']:
                info['sensors'].append(sensor_id)

        # Extract before/after merge counts
        if 'Original total rows:' in line:
            try:
                info['before_merge'] = int(line.split(':')[1].strip().replace(',', ''))
            except:
                pass

        if 'Final rows:' in line:
            try:
                info['after_merge'] = int(line.split(':')[1].strip().replace(',', ''))
            except:
                pass

        # Extract duplicates - look for the one with emoji or in validation section
        if "Duplicates removed:" in line:
            try:
                # Extract number after colon, remove commas
                dup_str = line.split(":")[-1].strip().replace(",", "")
                info["duplicates_removed"] = int(dup_str)
            except:
                pass
        # Extract duplicates - look for the one with emoji or in validation section
        if "Duplicates removed:" in line:
            try:
                # Extract number after colon, remove commas
                dup_str = line.split(":")[-1].strip().replace(",", "")
                info["duplicates_removed"] = int(dup_str)
            except:
                pass
        # Extract duplicates - look for the one with emoji or in validation section
        if "Duplicates removed:" in line:
            try:
                # Extract number after colon, remove commas
                dup_str = line.split(":")[-1].strip().replace(",", "")
                info["duplicates_removed"] = int(dup_str)
            except:
                pass
        # Extract duplicates - look for the one with emoji or in validation section
        if "Duplicates removed:" in line:
            try:
                # Extract number after colon, remove commas
                dup_str = line.split(":")[-1].strip().replace(",", "")
                info["duplicates_removed"] = int(dup_str)
            except:
                pass
        # Extract duplicates - look for the one with emoji or in validation section
        if "Duplicates removed:" in line:
            try:
                # Extract number after colon, remove commas
                dup_str = line.split(":")[-1].strip().replace(",", "")
                info["duplicates_removed"] = int(dup_str)
            except:
                pass
        # Extract duplicates - look for the one with emoji or in validation section
        if "Duplicates removed:" in line:
            try:
                # Extract number after colon, remove commas
                dup_str = line.split(":")[-1].strip().replace(",", "")
                info["duplicates_removed"] = int(dup_str)
            except:
                pass

        # Extract date range
        if 'From:' in line and i + 1 < len(lines):
            try:
                info['date_range_start'] = line.split('From:')[1].strip()
                # Next line should be "To:"
                if 'To:' in lines[i+1]:
                    info['date_range_end'] = lines[i+1].split('To:')[1].strip()
            except:
                pass

        # Extract duration
        if 'Duration:' in line and 'days' in line:
            try:
                info['duration_days'] = int(line.split(':')[1].split('days')[0].strip())
            except:
                pass

        # Extract file size
        if 'File size:' in line:
            try:
                size_str = line.split(':')[1].strip()
                if 'MB' in size_str:
                    info['file_size_mb'] = float(size_str.replace('MB', '').strip())
            except:
                pass

        # Check for success
        if 'File saved successfully' in line:
            info['success'] = True

    return info


def generate_combined_merge_summary():
    """Generate a combined summary of all merge operations."""
    merge_dir = "/Users/shawon/Downloads/Company Project/STR Data - Merged"
    output_file = os.path.join(merge_dir, "00_COMBINED_MERGE_SUMMARY.txt")

    # Find all merge log files
    log_files = sorted([f for f in os.listdir(merge_dir) if f.endswith('_merge_log.txt')])

    print("="*80)
    print("🔄 GENERATING COMBINED MERGE SUMMARY")
    print("="*80)
    print(f"\nFound {len(log_files)} merge log files\n")

    all_info = []

    for log_file in log_files:
        log_path = os.path.join(merge_dir, log_file)
        info = parse_merge_log(log_path)
        if info['structure']:
            all_info.append(info)
            print(f"✓ Parsed: {info['structure']}")

    # Generate summary
    summary = []

    summary.append("="*80)
    summary.append("COMBINED MERGE SUMMARY - ALL STRUCTURES")
    summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("="*80)

    # Overall statistics
    total_structures = len(all_info)
    successful_merges = sum(1 for info in all_info if info['success'])
    total_files_processed = sum(info['total_files'] for info in all_info)
    total_sensors = sum(len(info['sensors']) for info in all_info)
    total_rows_before = sum(info['before_merge'] for info in all_info)
    total_rows_after = sum(info['after_merge'] for info in all_info)
    total_duplicates = sum(info['duplicates_removed'] for info in all_info)

    summary.append(f"\n{'='*80}")
    summary.append("📊 OVERALL MERGE STATISTICS")
    summary.append("="*80)
    summary.append(f"\n   Total structures processed: {total_structures}")
    summary.append(f"   Successful merges: {successful_merges}")
    summary.append(f"   Total Excel files processed: {total_files_processed}")
    summary.append(f"   Total sensors identified: {total_sensors}")
    summary.append(f"   Total rows before merge: {total_rows_before:,}")
    summary.append(f"   Total rows after merge: {total_rows_after:,}")
    summary.append(f"   Total duplicates removed: {total_duplicates:,}")

    if total_rows_before > 0:
        duplicate_pct = (total_duplicates / total_rows_before) * 100
        summary.append(f"   Duplicate rate: {duplicate_pct:.2f}%")

    summary.append(f"\n{'='*80}")
    summary.append("📋 PER-STRUCTURE MERGE DETAILS")
    summary.append("="*80)

    for info in all_info:
        summary.append(f"\n{info['structure']}:")
        summary.append(f"   Status: {'✅ Success' if info['success'] else '❌ Failed'}")
        summary.append(f"   Files merged: {info['total_files']}")
        summary.append(f"   Sensors found: {len(info['sensors'])}")

        if info['sensors']:
            summary.append(f"   Sensor IDs: {', '.join(info['sensors'])}")

        summary.append(f"   Rows before merge: {info['before_merge']:,}")
        summary.append(f"   Rows after merge: {info['after_merge']:,}")
        summary.append(f"   Duplicates removed: {info['duplicates_removed']:,}")

        if info['before_merge'] > 0:
            dup_pct = (info['duplicates_removed'] / info['before_merge']) * 100
            summary.append(f"   Duplicate rate: {dup_pct:.2f}%")

        if info['date_range_start'] and info['date_range_end']:
            summary.append(f"   Date range: {info['date_range_start']} to {info['date_range_end']}")
            if info['duration_days'] > 0:
                summary.append(f"   Duration: {info['duration_days']} days")

        if info['file_size_mb'] > 0:
            summary.append(f"   File size: {info['file_size_mb']:.2f} MB")

    # Structures with most duplicates
    summary.append(f"\n{'='*80}")
    summary.append("⚠️  STRUCTURES WITH MOST DUPLICATES")
    summary.append("="*80)

    # Sort by duplicate count
    by_duplicates = sorted(all_info, key=lambda x: x['duplicates_removed'], reverse=True)

    summary.append("\nTop 10 structures by duplicate count:")
    for i, info in enumerate(by_duplicates[:10], 1):
        dup_pct = (info['duplicates_removed'] / info['before_merge'] * 100) if info['before_merge'] > 0 else 0
        summary.append(f"   {i}. {info['structure']}: {info['duplicates_removed']:,} duplicates ({dup_pct:.2f}%)")

    # Structures with most files
    summary.append(f"\n{'='*80}")
    summary.append("📁 STRUCTURES WITH MOST FILES")
    summary.append("="*80)

    by_files = sorted(all_info, key=lambda x: x['total_files'], reverse=True)

    summary.append("\nTop 10 structures by file count:")
    for i, info in enumerate(by_files[:10], 1):
        summary.append(f"   {i}. {info['structure']}: {info['total_files']} files, {len(info['sensors'])} sensors")

    # Sensor type distribution
    summary.append(f"\n{'='*80}")
    summary.append("🎯 SENSOR DISTRIBUTION")
    summary.append("="*80)

    sensor_types = {'AC': 0, 'DI': 0, 'TI': 0, 'TP': 0}

    for info in all_info:
        for sensor in info['sensors']:
            if sensor.startswith('AC'):
                sensor_types['AC'] += 1
            elif sensor.startswith('DI'):
                sensor_types['DI'] += 1
            elif sensor.startswith('TI'):
                sensor_types['TI'] += 1
            elif sensor.startswith('TP'):
                sensor_types['TP'] += 1

    summary.append(f"\n   Accelerometers (AC): {sensor_types['AC']}")
    summary.append(f"   Displacement sensors (DI): {sensor_types['DI']}")
    summary.append(f"   Tilt sensors (TI): {sensor_types['TI']}")
    summary.append(f"   Temperature probes (TP): {sensor_types['TP']}")
    summary.append(f"   Total sensors: {sum(sensor_types.values())}")

    # Data quality insights
    summary.append(f"\n{'='*80}")
    summary.append("💡 MERGE QUALITY INSIGHTS")
    summary.append("="*80)

    avg_duplicate_rate = (total_duplicates / total_rows_before * 100) if total_rows_before > 0 else 0
    avg_sensors_per_structure = total_sensors / total_structures if total_structures > 0 else 0
    avg_files_per_structure = total_files_processed / total_structures if total_structures > 0 else 0

    summary.append(f"\n   Average duplicate rate: {avg_duplicate_rate:.2f}%")
    summary.append(f"   Average sensors per structure: {avg_sensors_per_structure:.1f}")
    summary.append(f"   Average files per structure: {avg_files_per_structure:.1f}")

    if avg_duplicate_rate < 5:
        summary.append(f"\n   ✅ Low duplicate rate indicates good data collection overlap")
    elif avg_duplicate_rate < 15:
        summary.append(f"\n   ⚠️  Moderate duplicate rate - check file date ranges")
    else:
        summary.append(f"\n   🚨 High duplicate rate - significant file overlap detected")

    summary.append(f"\n{'='*80}")
    summary.append("📝 MERGE PROCESS NOTES")
    summary.append("="*80)
    summary.append("\n   Merge Strategy:")
    summary.append("   - All sheets from all Excel files merged into single file per structure")
    summary.append("   - Each sheet represents a different sensor")
    summary.append("   - Duplicates removed based on (timestamp, sensor_id) combination")
    summary.append("   - Long format chosen for optimal anomaly detection")

    summary.append("\n   Column Structure:")
    summary.append("   - timestamp: Datetime of measurement")
    summary.append("   - sensor_id: Unique identifier for each sensor")
    summary.append("   - sensor_type: Type of sensor (accelerometer, displacement, tilt, temperature_probe)")
    summary.append("   - Data columns: p2p, rms, value, temperature, humidity, battery")

    summary.append(f"\n{'='*80}")
    summary.append("END OF MERGE SUMMARY")
    summary.append("="*80)

    # Save summary
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))

    print(f"\n✅ Combined merge summary saved: {output_file}")
    print("="*80)

    # Print summary stats to console
    print("\n📊 SUMMARY:")
    print(f"   Structures: {total_structures}")
    print(f"   Total sensors: {total_sensors}")
    print(f"   Total rows after merge: {total_rows_after:,}")
    print(f"   Duplicates removed: {total_duplicates:,} ({avg_duplicate_rate:.2f}%)")
    print("="*80)


if __name__ == "__main__":
    generate_combined_merge_summary()
