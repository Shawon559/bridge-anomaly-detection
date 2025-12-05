import pandas as pd
import os
from datetime import datetime

def determine_sensor_type(sensor_id, columns):
    """
    Determine sensor type based on sensor_id and available columns.
    """
    sensor_id_upper = sensor_id.upper()

    if sensor_id_upper.startswith('AC'):
        return 'accelerometer'
    elif sensor_id_upper.startswith('DI'):
        return 'displacement'
    elif sensor_id_upper.startswith('TI'):
        return 'tilt'
    elif sensor_id_upper.startswith('TP'):
        return 'temperature_probe'
    else:
        # Fallback: check columns
        if 'p2p' in columns and 'rms' in columns:
            return 'accelerometer'
        elif 'value' in columns:
            return 'other'
        else:
            return 'unknown'

def merge_structure_with_all_sensors(structure_name, output_dir="STR Data - Merged"):
    """
    Merge all sheets (sensors) from all files in a structure.
    Creates one file per structure with sensor_id column.
    """
    base_path = f"/Users/shawon/Downloads/Company Project/STR Data/{structure_name}"
    output_base = f"/Users/shawon/Downloads/Company Project/{output_dir}"

    # Create output directory
    os.makedirs(output_base, exist_ok=True)

    # Log file
    log_file = os.path.join(output_base, f"{structure_name}_merge_log.txt")
    log = []

    def write_log(message):
        log.append(message)
        print(message)

    write_log("="*80)
    write_log(f"🔧 MERGING STRUCTURE: {structure_name} (WITH ALL SENSORS)")
    write_log(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_log("="*80)

    if not os.path.exists(base_path):
        write_log(f"❌ ERROR: Structure {structure_name} not found!")
        with open(log_file, 'w') as f:
            f.write('\n'.join(log))
        return None

    files = sorted([f for f in os.listdir(base_path) if f.endswith('.xlsx')])

    if not files:
        write_log(f"⚠️  WARNING: No Excel files found - SKIPPING")
        with open(log_file, 'w') as f:
            f.write('\n'.join(log))
        return None

    write_log(f"\n📁 Files to process: {len(files)}")
    for i, f in enumerate(files, 1):
        write_log(f"   {i}. {f}")

    # Process all files and sheets
    all_sensor_data = []
    sensor_summary = {}
    total_rows_original = 0

    write_log(f"\n{'='*80}")
    write_log("📥 LOADING ALL SHEETS FROM ALL FILES")
    write_log("="*80)

    for file_idx, filename in enumerate(files, 1):
        filepath = os.path.join(base_path, filename)
        write_log(f"\n📄 FILE {file_idx}: {filename}")

        try:
            # Get all sheets
            excel_file = pd.ExcelFile(filepath)
            sheet_names = excel_file.sheet_names
            write_log(f"   Found {len(sheet_names)} sheets: {', '.join(sheet_names)}")

            for sheet_name in sheet_names:
                try:
                    # Read sheet
                    df = pd.read_excel(filepath, sheet_name=sheet_name)

                    # Skip empty sheets
                    if len(df) == 0:
                        write_log(f"      ⚠️  Sheet '{sheet_name}': EMPTY - skipping")
                        continue

                    # Find timestamp column
                    date_col = None
                    for col in df.columns:
                        if 'time' in col.lower() or 'date' in col.lower():
                            date_col = col
                            break

                    if not date_col:
                        write_log(f"      ⚠️  Sheet '{sheet_name}': No timestamp column - skipping")
                        continue

                    # Convert timestamp
                    df[date_col] = pd.to_datetime(df[date_col])

                    # Add sensor_id and sensor_type columns
                    df['sensor_id'] = sheet_name
                    df['sensor_type'] = determine_sensor_type(sheet_name, df.columns.tolist())

                    # Rename timestamp column to standard name
                    df = df.rename(columns={date_col: 'timestamp'})

                    # Track sensor info
                    if sheet_name not in sensor_summary:
                        sensor_summary[sheet_name] = {
                            'type': df['sensor_type'].iloc[0],
                            'rows': 0,
                            'columns': [col for col in df.columns if col not in ['timestamp', 'sensor_id', 'sensor_type']],
                            'date_range': {'min': df['timestamp'].min(), 'max': df['timestamp'].max()}
                        }

                    sensor_summary[sheet_name]['rows'] += len(df)
                    total_rows_original += len(df)

                    all_sensor_data.append(df)

                    write_log(f"      ✓ Sheet '{sheet_name}': {len(df):,} rows, Type: {df['sensor_type'].iloc[0]}")

                except Exception as e:
                    write_log(f"      ❌ Sheet '{sheet_name}': Error - {str(e)}")
                    continue

        except Exception as e:
            write_log(f"   ❌ Error reading file: {str(e)}")
            continue

    if not all_sensor_data:
        write_log(f"\n❌ No data loaded - ABORTING")
        with open(log_file, 'w') as f:
            f.write('\n'.join(log))
        return None

    # Combine all sensor data
    write_log(f"\n{'='*80}")
    write_log("🔗 COMBINING ALL SENSOR DATA")
    write_log("="*80)

    write_log(f"\n⏳ Combining {len(all_sensor_data)} dataframes...")

    # Standardize columns across all dataframes
    # Get all possible columns
    all_columns = set()
    for df in all_sensor_data:
        all_columns.update(df.columns)

    # Standard columns order
    standard_cols = ['timestamp', 'sensor_id', 'sensor_type']
    sensor_cols = sorted([col for col in all_columns if col not in standard_cols])
    final_column_order = standard_cols + sensor_cols

    # Add missing columns to each dataframe
    for i, df in enumerate(all_sensor_data):
        for col in sensor_cols:
            if col not in df.columns:
                df[col] = None
        all_sensor_data[i] = df[final_column_order]

    # Concatenate
    merged_df = pd.concat(all_sensor_data, ignore_index=True)
    write_log(f"   ✓ Combined: {len(merged_df):,} total rows")

    # Remove duplicates
    write_log(f"\n{'='*80}")
    write_log("🔍 CHECKING FOR DUPLICATES")
    write_log("="*80)

    duplicates_before = merged_df.duplicated(subset=['timestamp', 'sensor_id']).sum()
    write_log(f"\n   📊 Duplicate records found: {duplicates_before:,}")

    if duplicates_before > 0:
        write_log(f"   🧹 Removing duplicates...")
        merged_df = merged_df.drop_duplicates(subset=['timestamp', 'sensor_id'], keep='first')
        write_log(f"   ✓ Removed {duplicates_before:,} duplicate rows")
    else:
        write_log(f"   ✓ No duplicates found")

    # Sort by timestamp and sensor_id
    write_log(f"\n{'='*80}")
    write_log("📊 SORTING DATA")
    write_log("="*80)

    write_log(f"\n   ⏳ Sorting by timestamp and sensor_id...")
    merged_df = merged_df.sort_values(by=['timestamp', 'sensor_id']).reset_index(drop=True)
    write_log(f"   ✓ Data sorted")

    # Validation and summary
    write_log(f"\n{'='*80}")
    write_log("✅ VALIDATION & SUMMARY")
    write_log("="*80)

    write_log(f"\n   📊 Original total rows: {total_rows_original:,}")
    write_log(f"   📊 Duplicates removed: {duplicates_before:,}")
    write_log(f"   📊 Final rows: {len(merged_df):,}")

    write_log(f"\n   📅 Overall date range:")
    write_log(f"      From: {merged_df['timestamp'].min()}")
    write_log(f"      To:   {merged_df['timestamp'].max()}")
    write_log(f"      Duration: {(merged_df['timestamp'].max() - merged_df['timestamp'].min()).days} days")

    write_log(f"\n   🎯 Sensors in merged data: {len(sensor_summary)}")
    for sensor_id, info in sorted(sensor_summary.items()):
        write_log(f"\n      {sensor_id} ({info['type']}):")
        write_log(f"         Rows: {info['rows']:,}")
        write_log(f"         Columns: {', '.join(info['columns'])}")
        write_log(f"         Date range: {info['date_range']['min']} to {info['date_range']['max']}")

    # Check for missing values
    write_log(f"\n   🔍 Missing values per column:")
    missing_summary = {}
    for col in merged_df.columns:
        if col not in ['timestamp', 'sensor_id', 'sensor_type']:
            null_count = merged_df[col].isnull().sum()
            if null_count > 0:
                pct = (null_count / len(merged_df)) * 100
                missing_summary[col] = {'count': null_count, 'percent': pct}

    if missing_summary:
        for col, stats in sorted(missing_summary.items()):
            write_log(f"      {col}: {stats['count']:,} ({stats['percent']:.1f}%)")
    else:
        write_log(f"      ✓ No missing values (this is expected for mixed sensor types)")

    # Save merged file
    write_log(f"\n{'='*80}")
    write_log("💾 SAVING MERGED FILE")
    write_log("="*80)

    output_filename = f"{structure_name}_merged.xlsx"
    output_path = os.path.join(output_base, output_filename)

    write_log(f"\n   ⏳ Saving to: {output_filename}")
    try:
        merged_df.to_excel(output_path, index=False, engine='openpyxl')
        file_size = os.path.getsize(output_path) / (1024*1024)
        write_log(f"   ✅ File saved successfully!")
        write_log(f"   📦 File size: {file_size:.2f} MB")
    except Exception as e:
        write_log(f"   ❌ ERROR saving file: {str(e)}")
        with open(log_file, 'w') as f:
            f.write('\n'.join(log))
        return None

    # Save log
    write_log(f"\n{'='*80}")
    write_log("📝 SAVING LOG FILE")
    write_log("="*80)

    with open(log_file, 'w') as f:
        f.write('\n'.join(log))
    write_log(f"   ✅ Log saved!")

    # Final summary
    write_log(f"\n{'='*80}")
    write_log("🎉 MERGE COMPLETE!")
    write_log("="*80)
    write_log(f"\n   ✅ Files processed: {len(files)}")
    write_log(f"   ✅ Sensors merged: {len(sensor_summary)}")
    write_log(f"   ✅ Total rows: {len(merged_df):,}")
    write_log(f"   ✅ Output: {output_dir}/{output_filename}")
    write_log(f"\n   ⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_log("="*80)

    return {
        'structure': structure_name,
        'output_file': output_path,
        'log_file': log_file,
        'sensors': len(sensor_summary),
        'original_rows': total_rows_original,
        'final_rows': len(merged_df),
        'duplicates_removed': duplicates_before
    }


def merge_all_structures():
    """Merge all structures with all sensors"""
    base_path = "/Users/shawon/Downloads/Company Project/STR Data"
    structures = sorted([d for d in os.listdir(base_path)
                        if os.path.isdir(os.path.join(base_path, d)) and d.startswith('STR')])

    print("="*80)
    print(f"🚀 MERGING ALL STRUCTURES (WITH ALL SENSORS)")
    print(f"   Found {len(structures)} structures to process")
    print("="*80)

    results = []

    for idx, structure in enumerate(structures, 1):
        print(f"\n\n{'#'*80}")
        print(f"# PROCESSING {idx}/{len(structures)}: {structure}")
        print(f"{'#'*80}\n")

        result = merge_structure_with_all_sensors(structure)
        results.append(result)

        if result:
            print(f"\n✅ {structure} completed: {result['sensors']} sensors, {result['final_rows']:,} rows")
        else:
            print(f"\n⚠️  {structure} had issues - check log")

    # Final summary
    print("\n\n" + "="*80)
    print("📊 FINAL SUMMARY - ALL STRUCTURES")
    print("="*80)

    successful = [r for r in results if r is not None]
    failed = [r for r in results if r is None]

    print(f"\n✅ Successfully merged: {len(successful)}/{len(structures)} structures")
    if failed:
        print(f"⚠️  Failed/Skipped: {len(failed)} structures")

    if successful:
        total_sensors = sum(r['sensors'] for r in successful)
        total_rows = sum(r['final_rows'] for r in successful)
        print(f"\n📋 Overall Statistics:")
        print(f"   Total sensors across all structures: {total_sensors}")
        print(f"   Total rows in all merged files: {total_rows:,}")

    print("\n" + "="*80)
    print("✅ ALL MERGING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    merge_all_structures()
