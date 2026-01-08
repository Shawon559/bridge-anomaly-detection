"""
Daily Bridge & Sensor Health Check
===================================

Production script for daily operations.
Scores ONE day of data and generates PNG + CSV reports.

Usage:
    python run_daily_health_check.py <STRUCTURE_ID> <DATA_FILE>

Example:
    python run_daily_health_check.py STR128 yesterdays_data.xlsx

Output:
    - daily_reports/STR128_2024-02-15.png (visualization)
    - daily_reports/STR128_2024-02-15_results.csv (detailed results)
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add model directory
sys.path.append('model')

def load_models(structure_id):
    """Load pre-trained models for the structure"""
    bridge_path = f'trained_models/bridge_health/{structure_id}_bridge_model.pkl'
    sensor_path = f'trained_models/sensor_health/{structure_id}_sensor_model.pkl'

    if not os.path.exists(bridge_path):
        raise FileNotFoundError(f"Bridge model not found: {structure_id}")
    if not os.path.exists(sensor_path):
        raise FileNotFoundError(f"Sensor model not found: {structure_id}")

    with open(bridge_path, 'rb') as f:
        bridge_model = pickle.load(f)
    with open(sensor_path, 'rb') as f:
        sensor_model = pickle.load(f)

    return bridge_model, sensor_model


def score_sensors(sensor_model, df):
    """Score individual sensor health"""
    return sensor_model.score_sensor_daily(df)


def score_bridge(bridge_model, df, test_date):
    """Score bridge health"""
    from datetime import timedelta

    context_start = test_date - timedelta(days=30)
    features = bridge_model.generate_features(df)
    test_features = features[features.index.date == test_date.date()]

    if len(test_features) == 0:
        return None, None

    if hasattr(bridge_model, 'trained_feature_cols'):
        feature_cols = bridge_model.trained_feature_cols
    else:
        feature_cols = [c for c in bridge_model.features_used if c in test_features.columns]

    X_test = pd.DataFrame(index=test_features.index)
    for col in feature_cols:
        if col in test_features.columns:
            X_test[col] = test_features[col]
        else:
            X_test[col] = 0

    X_test = X_test.fillna(0)
    X_scaled = bridge_model.scaler.transform(X_test)
    scores_dict = bridge_model.get_ensemble_scores(X_scaled)
    anomaly_scores = bridge_model.calculate_anomaly_score(test_features, scores_dict)

    return anomaly_scores, test_features


def generate_report(structure_id, test_date, sensor_results, bridge_scores, bridge_model, sensor_model, output_dir):
    """Generate PNG visualization and CSV results"""

    if bridge_scores is not None:
        max_score = bridge_scores.max()
        avg_score = bridge_scores.mean()
        if max_score > 7:
            severity = "CRITICAL"
        elif max_score > 4:
            severity = "WARNING"
        else:
            severity = "NORMAL"
    else:
        max_score = 0
        avg_score = 0
        severity = "NO DATA"

    # Create visualization
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, top=0.92, bottom=0.08)

    fig.suptitle(f'{structure_id} Daily Health Report - {test_date.strftime("%b %d, %Y")}',
                 fontsize=14, fontweight='bold')

    # 1. Bridge Health Score
    ax1 = fig.add_subplot(gs[0, :])
    if bridge_scores is not None and len(bridge_scores) > 0:
        hours = np.arange(len(bridge_scores)) / len(bridge_scores) * 24
        ax1.plot(hours, bridge_scores, 'b-', linewidth=2, label='Anomaly Score')
        ax1.fill_between(hours, 0, bridge_scores, alpha=0.3)
        ax1.axhline(y=4, color='orange', linestyle='--', alpha=0.5, label='Warning (4)')
        ax1.axhline(y=7, color='red', linestyle='--', alpha=0.5, label='Critical (7)')
    else:
        ax1.text(0.5, 0.5, 'NO BRIDGE DATA AVAILABLE',
                ha='center', va='center', fontsize=14, color='red',
                transform=ax1.transAxes)
        ax1.axhline(y=4, color='orange', linestyle='--', alpha=0.5, label='Warning (4)')
        ax1.axhline(y=7, color='red', linestyle='--', alpha=0.5, label='Critical (7)')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Bridge Health Score (0-10)')
    ax1.set_title(f'Bridge Health Score - Max: {max_score:.1f}, Avg: {avg_score:.1f}, Status: {severity}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 10)

    ax2 = fig.add_subplot(gs[1, 0])
    critical_sensors = sensor_results[sensor_results['health_score'] > 7]
    warning_sensors = sensor_results[(sensor_results['health_score'] > 4) & (sensor_results['health_score'] <= 7)]
    sensor_counts = {
        'Healthy\n(0-4)': len(sensor_results[sensor_results['health_score'] <= 4]),
        'Warning\n(4-7)': len(warning_sensors),
        'Critical\n(>7)': len(critical_sensors)
    }
    colors = ['green', 'orange', 'red']
    bars = ax2.bar(sensor_counts.keys(), sensor_counts.values(), color=colors, alpha=0.6)
    ax2.set_ylabel('Number of Sensors')
    ax2.set_title('Sensor Health Summary')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    ax3 = fig.add_subplot(gs[1, 1])
    if len(sensor_results) > 0:
        sensor_sorted = sensor_results.sort_values('health_score', ascending=False)
        colors_map = sensor_sorted['health_score'].apply(
            lambda x: 'red' if x > 7 else ('orange' if x > 4 else 'green')
        )
        ax3.barh(range(len(sensor_sorted)), sensor_sorted['health_score'], color=colors_map, alpha=0.6)
        ax3.set_yticks(range(len(sensor_sorted)))
        ax3.set_yticklabels(sensor_sorted['sensor_id'], fontsize=8)
        ax3.set_xlabel('Health Score')
        ax3.set_title('Individual Sensor Scores')
        ax3.axvline(x=4, color='orange', linestyle='--', alpha=0.5)
        ax3.axvline(x=7, color='red', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.set_xlim(0, 10)
    else:
        ax3.text(0.5, 0.5, 'NO SENSOR DATA AVAILABLE',
                ha='center', va='center', fontsize=14, color='red',
                transform=ax3.transAxes)
        ax3.set_title('Individual Sensor Scores')
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 1)

    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    issues_text = "Issues Detected:\n"
    problematic = sensor_results[sensor_results['health_score'] > 4].sort_values('health_score', ascending=False)

    if len(problematic) > 0:
        for idx, (_, row) in enumerate(problematic.iterrows()):
            if idx >= 5:
                issues_text += f"  ... and {len(problematic)-5} more\n"
                break
            tilt_info = f"[{row['tilt_axis']}]" if pd.notna(row.get('tilt_axis')) else ""
            gap_info = ""
            if 'data_gaps' in row and row['data_gaps'] != 'None':
                coverage = row.get('data_coverage_%', 100)
                gap_info = f" | Coverage: {coverage:.0f}% | Gaps: {row['data_gaps']}"
            issues_text += f"  • {row['sensor_id']}{tilt_info}: {row['diagnosis']} (Score: {row['health_score']:.1f}){gap_info}\n"
    else:
        issues_text += "  OK All sensors operating normally\n"

    issues_text += f"\nModel Features: Weighted ensemble, MAD baselines, tilt axis tracking"

    ax4.text(0.05, 0.95, issues_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    os.makedirs(output_dir, exist_ok=True)
    png_path = f'{output_dir}/{structure_id}_{test_date.strftime("%Y-%m-%d")}.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save CSV
    csv_path = f'{output_dir}/{structure_id}_{test_date.strftime("%Y-%m-%d")}_results.csv'
    sensor_results.to_csv(csv_path, index=False)

    return png_path, csv_path


def main():
    if len(sys.argv) != 3:
        print("Usage: python run_daily_health_check.py <STRUCTURE_ID> <DATA_FILE>")
        print("\nExample:")
        print("  python run_daily_health_check.py STR128 yesterdays_data.xlsx")
        sys.exit(1)

    structure_id = sys.argv[1]
    data_file = sys.argv[2]

    print("="*80)
    print(f"DAILY HEALTH CHECK: {structure_id}")
    print("="*80)

    # Load data
    print(f"\n1. Loading data from {data_file}...")
    df = pd.read_excel(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    test_date = df['timestamp'].dt.date.iloc[0]
    print(f"   Date: {test_date}")
    print(f"   Rows: {len(df)}")

    # Load models
    print(f"\n2. Loading pre-trained models...")
    bridge_model, sensor_model = load_models(structure_id)
    print(f"   OK Bridge model loaded")
    print(f"   OK Sensor model loaded")

    # Score sensors
    print(f"\n3. Scoring sensor health...")
    sensor_results = score_sensors(sensor_model, df)
    critical = len(sensor_results[sensor_results['health_score'] > 7])
    warning = len(sensor_results[(sensor_results['health_score'] > 4) & (sensor_results['health_score'] <= 7)])
    healthy = len(sensor_results) - critical - warning
    print(f"   Critical: {critical}, Warning: {warning}, Healthy: {healthy}")

    # Score bridge
    print(f"\n4. Scoring bridge health...")
    bridge_scores, _ = score_bridge(bridge_model, df, pd.to_datetime(test_date))
    if bridge_scores is not None:
        print(f"   Max score: {bridge_scores.max():.1f}")
        print(f"   Avg score: {bridge_scores.mean():.1f}")
    else:
        print(f"   No bridge scores (insufficient data)")

    # Generate report
    print(f"\n5. Generating reports...")
    png_path, csv_path = generate_report(
        structure_id, pd.to_datetime(test_date), sensor_results,
        bridge_scores, bridge_model, sensor_model, 'daily_reports'
    )
    print(f"   OK PNG: {png_path}")
    print(f"   OK CSV: {csv_path}")

    print(f"\n{'='*80}")
    print("DAILY CHECK COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
