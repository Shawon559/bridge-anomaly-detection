"""
Quick verification script to check if model results make sense.

This will help us understand if the anomaly detection is working correctly.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load results
print("Loading model results...")
df = pd.read_csv(r"c:\Users\pc\Downloads\Project_Anomaly detection IoT-20251203T010920Z-1-001\Project_Anomaly detection IoT\Shawon\Company Project\STR128 Analysis\Model Results\full_results_with_predictions.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Total samples: {len(df):,}")
print(f"Anomalies: {df['is_anomaly'].sum():,}")
print(f"Normal: {(~df['is_anomaly']).sum():,}")

# Check first 100 samples
print("\n" + "="*80)
print("FIRST 100 SAMPLES:")
print("="*80)
first_100 = df.head(100)
print(f"Anomalies in first 100: {first_100['is_anomaly'].sum()}")
print(f"Date range: {first_100['timestamp'].min()} to {first_100['timestamp'].max()}")

print("\nValues in first 100:")
print(f"  Displacement: min={first_100['displacement_combined'].min():.2f}, max={first_100['displacement_combined'].max():.2f}, mean={first_100['displacement_combined'].mean():.2f}")
print(f"  Dynamic tilt: min={first_100['tilt_dynamic_avg'].min():.2f}, max={first_100['tilt_dynamic_avg'].max():.2f}, mean={first_100['tilt_dynamic_avg'].mean():.2f}")
print(f"  Stable tilt: min={first_100['tilt_stable'].min():.2f}, max={first_100['tilt_stable'].max():.2f}, mean={first_100['tilt_stable'].mean():.2f}")

# Check middle 100 samples (around February 2024)
print("\n" + "="*80)
print("MIDDLE 100 SAMPLES (around Feb 2024):")
print("="*80)
middle_start = len(df) // 2
middle_100 = df.iloc[middle_start:middle_start+100]
print(f"Anomalies in middle 100: {middle_100['is_anomaly'].sum()}")
print(f"Date range: {middle_100['timestamp'].min()} to {middle_100['timestamp'].max()}")

print("\nValues in middle 100:")
print(f"  Displacement: min={middle_100['displacement_combined'].min():.2f}, max={middle_100['displacement_combined'].max():.2f}, mean={middle_100['displacement_combined'].mean():.2f}")
print(f"  Dynamic tilt: min={middle_100['tilt_dynamic_avg'].min():.2f}, max={middle_100['tilt_dynamic_avg'].max():.2f}, mean={middle_100['tilt_dynamic_avg'].mean():.2f}")
print(f"  Stable tilt: min={middle_100['tilt_stable'].min():.2f}, max={middle_100['tilt_stable'].max():.2f}, mean={middle_100['tilt_stable'].mean():.2f}")

# Check when anomalies stop being so common
print("\n" + "="*80)
print("TEMPORAL DISTRIBUTION OF ANOMALIES:")
print("="*80)

df['year_month'] = df['timestamp'].dt.to_period('M')
anomaly_by_month = df.groupby('year_month')['is_anomaly'].agg(['sum', 'count'])
anomaly_by_month['percentage'] = (anomaly_by_month['sum'] / anomaly_by_month['count'] * 100).round(2)

print(anomaly_by_month)

# Look at specific anomalous samples
print("\n" + "="*80)
print("SAMPLE OF 10 ANOMALIES vs 10 NORMAL:")
print("="*80)

anomalies_sample = df[df['is_anomaly']].head(10)
normal_sample = df[~df['is_anomaly']].head(10)

print("\nANOMALIES:")
print(anomalies_sample[['timestamp', 'displacement_combined', 'tilt_dynamic_avg', 'tilt_stable']])

print("\nNORMAL:")
print(normal_sample[['timestamp', 'displacement_combined', 'tilt_dynamic_avg', 'tilt_stable']])

# Create a simple timeline visualization
print("\n" + "="*80)
print("Creating simple timeline visualization...")
print("="*80)

fig, ax = plt.subplots(figsize=(16, 6))

# Plot just anomaly flag over time
anomaly_indices = df[df['is_anomaly']].index
normal_indices = df[~df['is_anomaly']].index

ax.scatter(df.loc[normal_indices, 'timestamp'],
           [1]*len(normal_indices),
           s=1, c='blue', alpha=0.3, label='Normal')
ax.scatter(df.loc[anomaly_indices, 'timestamp'],
           [1]*len(anomaly_indices),
           s=5, c='red', marker='|', label='Anomaly')

ax.set_ylim([0.5, 1.5])
ax.set_xlabel('Date', fontsize=14)
ax.set_title('Timeline: Where are anomalies detected?', fontsize=16, fontweight='bold')
ax.legend()
ax.set_yticks([])
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(r"c:\Users\pc\Downloads\Project_Anomaly detection IoT-20251203T010920Z-1-001\Project_Anomaly detection IoT\Shawon\Company Project\verification_timeline.png",
            dpi=150, bbox_inches='tight')
print("Saved: verification_timeline.png")
plt.close()

# Create side-by-side comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

features = ['displacement_combined', 'tilt_dynamic_avg', 'tilt_stable']
titles = ['Displacement (mm)', 'Dynamic Tilt (deg)', 'Stable Tilt (deg)']

for idx, (feat, title) in enumerate(zip(features, titles)):
    ax = axes[idx]

    # Plot normal
    normal_data = df[~df['is_anomaly']][feat]
    anomaly_data = df[df['is_anomaly']][feat]

    ax.hist(normal_data, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
    ax.hist(anomaly_data, bins=30, alpha=0.6, label='Anomaly', color='red', density=True)

    ax.set_xlabel(title, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{title}\nNormal vs Anomaly Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r"c:\Users\pc\Downloads\Project_Anomaly detection IoT-20251203T010920Z-1-001\Project_Anomaly detection IoT\Shawon\Company Project\verification_distributions.png",
            dpi=150, bbox_inches='tight')
print("Saved: verification_distributions.png")
plt.close()

print("\n" + "="*80)
print("VERIFICATION COMPLETE!")
print("="*80)
print("\nCheck the generated PNG files for visual verification.")
print("\nKey question: Does September 2023 data look genuinely different from later data?")
