"""
Run 10 Random Tests
Tests bridge and sensor health models on 10 random days and generates PNG visualizations.
"""

import pandas as pd
import pickle
import os
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
sys.path.append('model')

structures_with_models = [
    'STR122', 'STR124', 'STR126', 'STR128', 'STR129', 'STR130', 'STR132',
    'STR171', 'STR172', 'STR173', 'STR175', 'STR176', 'STR177',
    'STR179', 'STR180', 'STR181', 'STR182', 'STR184', 'STR199'
]

test_dates = pd.date_range('2024-02-01', '2024-02-15', freq='D')

random.seed(42)
tests = []
for i in range(10):
    structure = random.choice(structures_with_models)
    date = random.choice(test_dates)
    tests.append((i+1, structure, date))

print("="*80)
print("RUNNING 10 RANDOM TESTS WITH ENHANCED MODELS")
print("="*80)

merged_data_dir = '../STR Data - Merged'
results_summary = []

for test_num, structure, test_date in tests:
    print(f"\n{'='*80}")
    print(f"TEST {test_num}/10: {structure} on {test_date.strftime('%Y-%m-%d')}")
    print(f"{'='*80}")

    try:
        file_path = f"{merged_data_dir}/{structure}_merged.xlsx"
        df = pd.read_excel(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        test_df = df[df['timestamp'].dt.date == test_date.date()]

        if len(test_df) == 0:
            print(f"  WARNING  No data for {test_date.strftime('%Y-%m-%d')}")
            continue

        # Load models
        bridge_path = f'trained_models/bridge_health/{structure}_bridge_model.pkl'
        sensor_path = f'trained_models/sensor_health/{structure}_sensor_model.pkl'

        with open(bridge_path, 'rb') as f:
            bridge_model = pickle.load(f)
        with open(sensor_path, 'rb') as f:
            sensor_model = pickle.load(f)

        # ===== SENSOR HEALTH SCORING =====
        print("\n--- Sensor Health Scoring ---")
        sensor_results = sensor_model.score_sensor_daily(test_df)

        critical_sensors = sensor_results[sensor_results['health_score'] > 7]
        warning_sensors = sensor_results[(sensor_results['health_score'] > 4) & (sensor_results['health_score'] <= 7)]

        print(f"  Critical (>7): {len(critical_sensors)}")
        print(f"  Warning (4-7): {len(warning_sensors)}")
        print(f"  Healthy (0-4): {len(sensor_results) - len(critical_sensors) - len(warning_sensors)}")

        # ===== BRIDGE HEALTH SCORING =====
        print("\n--- Bridge Health Scoring ---")

        # Get context data (need more context for better features)
        context_start = test_date - timedelta(days=30)
        context_df = df[(df['timestamp'] >= context_start) & (df['timestamp'] <= test_date)]

        # Generate features for context
        features = bridge_model.generate_features(context_df)

        # Get test day features only
        test_features = features[features.index.date == test_date.date()]

        if len(test_features) == 0:
            print("  WARNING  No features generated for test date")
            continue

        # Ensure we only use features that were in training (exact columns and order)
        if hasattr(bridge_model, 'trained_feature_cols'):
            feature_cols = bridge_model.trained_feature_cols
        else:
            feature_cols = [c for c in bridge_model.features_used if c in test_features.columns]

        # Create feature matrix with exact columns from training
        X_test = pd.DataFrame(index=test_features.index)
        for col in feature_cols:
            if col in test_features.columns:
                X_test[col] = test_features[col]
            else:
                X_test[col] = 0  # Missing features filled with 0

        X_test = X_test.fillna(0)

        # Transform
        X_scaled = bridge_model.scaler.transform(X_test)

        # Get scores
        scores_dict = bridge_model.get_ensemble_scores(X_scaled)
        anomaly_scores = bridge_model.calculate_anomaly_score(test_features, scores_dict)

        max_score = anomaly_scores.max()
        avg_score = anomaly_scores.mean()

        print(f"  Max anomaly score: {max_score:.1f}")
        print(f"  Avg anomaly score: {avg_score:.1f}")

        if max_score > 7:
            severity = "CRITICAL"
        elif max_score > 4:
            severity = "WARNING"
        else:
            severity = "NORMAL"

        print(f"  Bridge status: {severity}")

        # ===== CREATE VISUALIZATION =====
        print("\n--- Generating PNG ---")

        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, top=0.92, bottom=0.08)

        # Title - simpler
        fig.suptitle(f'{structure} Daily Health Report - {test_date.strftime("%b %d, %Y")}',
                     fontsize=14, fontweight='bold')

        # 1. Bridge Health Score
        ax1 = fig.add_subplot(gs[0, :])
        hours = test_features.index.hour + test_features.index.minute/60
        ax1.plot(hours, anomaly_scores, 'b-', linewidth=2, label='Anomaly Score')
        ax1.axhline(y=4, color='orange', linestyle='--', alpha=0.5, label='Warning (4)')
        ax1.axhline(y=7, color='red', linestyle='--', alpha=0.5, label='Critical (7)')
        ax1.fill_between(hours, 0, anomaly_scores, alpha=0.3)
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Bridge Health Score (0-10)')
        ax1.set_title(f'Bridge Health Score - Max: {max_score:.1f}, Avg: {avg_score:.1f}, Status: {severity}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 10)

        ax2 = fig.add_subplot(gs[1, 0])
        sensor_counts = {
            'Healthy\\n(0-4)': len(sensor_results[sensor_results['health_score'] <= 4]),
            'Warning\\n(4-7)': len(warning_sensors),
            'Critical\\n(>7)': len(critical_sensors)
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

        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')

        issues_text = "Issues Detected:\\n"
        problematic = sensor_results[sensor_results['health_score'] > 4].sort_values('health_score', ascending=False)

        if len(problematic) > 0:
            for idx, row in enumerate(problematic.iterrows()):
                if idx >= 5:  # Limit to top 5
                    issues_text += f"  ... and {len(problematic)-5} more\\n"
                    break
                _, row = row
                tilt_info = f"[{row['tilt_axis']}]" if pd.notna(row.get('tilt_axis')) else ""
                issues_text += f"  • {row['sensor_id']}{tilt_info}: {row['diagnosis']} (Score: {row['health_score']:.1f})\\n"
        else:
            issues_text += "  OK All sensors operating normally\\n"

        issues_text += f"\\nModel Features: Weighted ensemble, MAD baselines, tilt axis tracking"

        ax4.text(0.05, 0.95, issues_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # Save
        output_dir = f'test_results_enhanced'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/test{test_num:02d}_{structure}_{test_date.strftime("%Y%m%d")}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  OK Saved: {output_path}")

        results_summary.append({
            'test_num': test_num,
            'structure': structure,
            'date': test_date.strftime('%Y-%m-%d'),
            'bridge_max': max_score,
            'bridge_avg': avg_score,
            'bridge_status': severity,
            'sensor_critical': len(critical_sensors),
            'sensor_warning': len(warning_sensors),
            'sensor_healthy': len(sensor_results) - len(critical_sensors) - len(warning_sensors)
        })

    except Exception as e:
        print(f"  ERROR Error: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("TEST SUMMARY")
print(f"{'='*80}\n")

if results_summary:
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    print(f"\nOK All {len(results_summary)} tests completed successfully!")
    print(f"OK PNG files saved in: test_results_enhanced/")
else:
    print("No tests completed successfully")
