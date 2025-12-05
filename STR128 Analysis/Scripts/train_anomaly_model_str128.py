"""
STR128 Anomaly Detection Model - Training & Evaluation

This script:
1. Loads preprocessed STR128 data
2. Trains Isolation Forest anomaly detection model
3. Evaluates and visualizes results
4. Generates presentation-ready outputs

Author: Project Team
Date: 2025-12-05
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
INPUT_FILE = Path(r"c:\Users\pc\Downloads\Project_Anomaly detection IoT-20251203T010920Z-1-001\Project_Anomaly detection IoT\Shawon\Company Project\Preprocessed Data\STR128_model_ready.xlsx")
OUTPUT_DIR = Path(r"c:\Users\pc\Downloads\Project_Anomaly detection IoT-20251203T010920Z-1-001\Project_Anomaly detection IoT\Shawon\Company Project\Model Results")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load preprocessed STR128 data."""
    print("Loading STR128 model-ready data...")
    df = pd.read_excel(INPUT_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Loaded {len(df):,} samples")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    return df

def prepare_features(df):
    """Prepare feature matrix for model training."""
    print("\nPreparing features...")

    # Core structural features (most important)
    core_features = [
        'displacement_combined',
        'tilt_dynamic_avg',
        'tilt_stable'
    ]

    # Rate of change features (detect sudden changes)
    rate_features = [
        'displacement_rate',
        'tilt_dynamic_rate',
        'tilt_stable_rate'
    ]

    # Sensor health features
    health_features = [
        'tilt_dynamic_std',
        'displacement_sync_error'
    ]

    # Combine all features
    all_features = core_features + rate_features + health_features

    print(f"Using {len(all_features)} features:")
    for feat in all_features:
        print(f"  - {feat}")

    X = df[all_features].copy()

    # Check for any remaining NaN
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"\nWarning: {nan_count} NaN values found, filling with median...")
        X = X.fillna(X.median())

    return X, all_features

def train_model(X, contamination=0.01):
    """
    Train Isolation Forest model.

    contamination: expected proportion of anomalies (1% default)
    """
    print(f"\nTraining Isolation Forest model...")
    print(f"Contamination (expected anomaly rate): {contamination*100:.1f}%")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        max_samples='auto',
        verbose=0
    )

    predictions = model.fit_predict(X_scaled)
    anomaly_scores = model.score_samples(X_scaled)

    # Convert predictions: -1 = anomaly, 1 = normal
    is_anomaly = predictions == -1

    print(f"Training complete!")
    print(f"Anomalies detected: {is_anomaly.sum():,} ({is_anomaly.sum()/len(X)*100:.2f}%)")

    return model, scaler, predictions, anomaly_scores, is_anomaly

def analyze_anomalies(df, is_anomaly, anomaly_scores, features):
    """Analyze detected anomalies."""
    print("\nAnalyzing anomalies...")

    # Add results to dataframe
    df_results = df.copy()
    df_results['is_anomaly'] = is_anomaly
    df_results['anomaly_score'] = anomaly_scores

    # Get anomalous samples
    df_anomalies = df_results[df_results['is_anomaly']]

    print(f"\nAnomaly Statistics:")
    print(f"  Total anomalies: {len(df_anomalies):,}")
    print(f"  Percentage: {len(df_anomalies)/len(df)*100:.2f}%")
    print(f"  Date range: {df_anomalies['timestamp'].min()} to {df_anomalies['timestamp'].max()}")

    # Analyze by time
    df_anomalies['hour'] = df_anomalies['timestamp'].dt.hour
    df_anomalies['day_of_week'] = df_anomalies['timestamp'].dt.dayofweek
    df_anomalies['month'] = df_anomalies['timestamp'].dt.month

    print(f"\nAnomaly Distribution:")
    print(f"  By hour (top 3): {df_anomalies['hour'].value_counts().head(3).to_dict()}")
    print(f"  By day of week (0=Mon): {df_anomalies['day_of_week'].value_counts().to_dict()}")
    print(f"  By month: {df_anomalies['month'].value_counts().to_dict()}")

    # Feature statistics for anomalies vs normal
    print(f"\nFeature Comparison (Anomalies vs Normal):")
    print(f"{'Feature':<25} {'Normal Mean':<15} {'Anomaly Mean':<15} {'Difference':<10}")
    print("-" * 70)

    for feat in features:
        normal_mean = df_results[~df_results['is_anomaly']][feat].mean()
        anomaly_mean = df_results[df_results['is_anomaly']][feat].mean()
        diff = anomaly_mean - normal_mean
        print(f"{feat:<25} {normal_mean:<15.4f} {anomaly_mean:<15.4f} {diff:<10.4f}")

    return df_results, df_anomalies

def create_visualizations(df_results, features):
    """Create comprehensive visualizations."""
    print("\nCreating visualizations...")

    # Figure 1: Time series with anomalies highlighted
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    fig.suptitle('STR128 Structural Measurements with Detected Anomalies', fontsize=16, fontweight='bold')

    # Plot 1: Displacement
    ax1 = axes[0]
    normal = df_results[~df_results['is_anomaly']]
    anomaly = df_results[df_results['is_anomaly']]

    ax1.plot(normal['timestamp'], normal['displacement_combined'],
             'b.', markersize=1, alpha=0.3, label='Normal')
    ax1.scatter(anomaly['timestamp'], anomaly['displacement_combined'],
                c='red', s=20, marker='x', label='Anomaly', zorder=5)
    ax1.set_ylabel('Displacement (mm)', fontsize=12)
    ax1.set_title('Combined Displacement (DI549 + DI550)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Dynamic Tilt
    ax2 = axes[1]
    ax2.plot(normal['timestamp'], normal['tilt_dynamic_avg'],
             'g.', markersize=1, alpha=0.3, label='Normal')
    ax2.scatter(anomaly['timestamp'], anomaly['tilt_dynamic_avg'],
                c='red', s=20, marker='x', label='Anomaly', zorder=5)
    ax2.set_ylabel('Tilt (degrees)', fontsize=12)
    ax2.set_title('Dynamic Tilt (Average of TI551, TI552, TI553)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Stable Tilt
    ax3 = axes[2]
    ax3.plot(normal['timestamp'], normal['tilt_stable'],
             'm.', markersize=1, alpha=0.3, label='Normal')
    ax3.scatter(anomaly['timestamp'], anomaly['tilt_stable'],
                c='red', s=20, marker='x', label='Anomaly', zorder=5)
    ax3.set_ylabel('Tilt (degrees)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title('Stable Tilt (TI554)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_timeseries_with_anomalies.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: 1_timeseries_with_anomalies.png")
    plt.close()

    # Figure 2: Anomaly score distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Anomaly Score Analysis', fontsize=14, fontweight='bold')

    # Histogram
    ax1 = axes[0]
    ax1.hist(df_results[~df_results['is_anomaly']]['anomaly_score'],
             bins=50, alpha=0.7, label='Normal', color='blue')
    ax1.hist(df_results[df_results['is_anomaly']]['anomaly_score'],
             bins=30, alpha=0.7, label='Anomaly', color='red')
    ax1.set_xlabel('Anomaly Score', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Anomaly Scores', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Time series of anomaly score
    ax2 = axes[1]
    ax2.plot(df_results['timestamp'], df_results['anomaly_score'],
             'b.', markersize=1, alpha=0.3)
    ax2.scatter(anomaly['timestamp'], anomaly['anomaly_score'],
                c='red', s=20, marker='x', label='Detected Anomalies', zorder=5)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Anomaly Score', fontsize=12)
    ax2.set_title('Anomaly Scores Over Time', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_anomaly_scores.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: 2_anomaly_scores.png")
    plt.close()

    # Figure 3: Feature distributions (Normal vs Anomaly)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Feature Distributions: Normal vs Anomaly', fontsize=14, fontweight='bold')

    key_features = ['displacement_combined', 'tilt_dynamic_avg', 'tilt_stable', 'displacement_rate']

    for idx, feat in enumerate(key_features):
        ax = axes[idx // 2, idx % 2]

        normal_data = df_results[~df_results['is_anomaly']][feat]
        anomaly_data = df_results[df_results['is_anomaly']][feat]

        ax.hist(normal_data, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
        ax.hist(anomaly_data, bins=30, alpha=0.6, label='Anomaly', color='red', density=True)
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(feat.replace('_', ' ').title(), fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_feature_distributions.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: 3_feature_distributions.png")
    plt.close()

    # Figure 4: Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    corr_matrix = df_results[features + ['is_anomaly']].corr()

    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Feature Correlation Matrix (Including Anomaly Flag)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: 4_correlation_heatmap.png")
    plt.close()

    # Figure 5: Monthly anomaly trend
    fig, ax = plt.subplots(figsize=(12, 6))

    df_results['year_month'] = df_results['timestamp'].dt.to_period('M')
    monthly_stats = df_results.groupby('year_month').agg({
        'is_anomaly': ['sum', 'count']
    })
    monthly_stats.columns = ['anomaly_count', 'total_count']
    monthly_stats['anomaly_rate'] = (monthly_stats['anomaly_count'] / monthly_stats['total_count']) * 100

    ax.bar(range(len(monthly_stats)), monthly_stats['anomaly_rate'], color='coral', alpha=0.7)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Anomaly Rate (%)', fontsize=12)
    ax.set_title('Monthly Anomaly Rate Trend', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(monthly_stats)))
    ax.set_xticklabels([str(period) for period in monthly_stats.index], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '5_monthly_anomaly_trend.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: 5_monthly_anomaly_trend.png")
    plt.close()

def generate_report(df_results, df_anomalies, features, model_params):
    """Generate comprehensive text report."""
    print("\nGenerating comprehensive report...")

    report_file = OUTPUT_DIR / 'STR128_Anomaly_Detection_Report.txt'

    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STR128 ANOMALY DETECTION MODEL - RESULTS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Structure: STR128\n")
        f.write(f"Model: Isolation Forest\n\n")

        f.write("DATASET INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Total samples: {len(df_results):,}\n")
        f.write(f"Date range: {df_results['timestamp'].min()} to {df_results['timestamp'].max()}\n")
        f.write(f"Duration: {(df_results['timestamp'].max() - df_results['timestamp'].min()).days} days\n")
        f.write(f"Features used: {len(features)}\n\n")

        f.write("MODEL PARAMETERS\n")
        f.write("-"*80 + "\n")
        for key, value in model_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("ANOMALY DETECTION RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"Anomalies detected: {len(df_anomalies):,}\n")
        f.write(f"Anomaly rate: {len(df_anomalies)/len(df_results)*100:.2f}%\n")
        f.write(f"Normal samples: {len(df_results) - len(df_anomalies):,}\n\n")

        f.write("ANOMALY CHARACTERISTICS\n")
        f.write("-"*80 + "\n")

        # Top anomalous periods
        f.write("\nTop 10 Most Anomalous Samples (Lowest Scores):\n")
        top_anomalies = df_results.nsmallest(10, 'anomaly_score')
        for idx, row in top_anomalies.iterrows():
            f.write(f"  {row['timestamp']}: Score = {row['anomaly_score']:.4f}\n")
            f.write(f"    Displacement: {row['displacement_combined']:.2f} mm, ")
            f.write(f"Dynamic Tilt: {row['tilt_dynamic_avg']:.2f}°, ")
            f.write(f"Stable Tilt: {row['tilt_stable']:.2f}°\n")

        f.write("\nFeature Statistics (Anomalies vs Normal):\n")
        f.write(f"{'Feature':<25} {'Normal Mean':<15} {'Anomaly Mean':<15} {'Difference':<15}\n")
        f.write("-"*70 + "\n")

        for feat in features:
            normal_mean = df_results[~df_results['is_anomaly']][feat].mean()
            anomaly_mean = df_results[df_results['is_anomaly']][feat].mean()
            diff = anomaly_mean - normal_mean
            f.write(f"{feat:<25} {normal_mean:<15.4f} {anomaly_mean:<15.4f} {diff:<15.4f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION & RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")

        f.write("Key Findings:\n")
        f.write(f"1. Detected {len(df_anomalies)} anomalous measurements out of {len(df_results):,} total samples\n")
        f.write(f"2. Anomaly rate of {len(df_anomalies)/len(df_results)*100:.2f}% suggests model is working as expected\n")
        f.write(f"3. Most anomalies are driven by deviations in displacement and tilt measurements\n\n")

        f.write("Recommendations:\n")
        f.write("1. Review identified anomalies for potential structural concerns\n")
        f.write("2. Cross-reference with maintenance logs or known events\n")
        f.write("3. Consider adjusting contamination parameter if anomaly rate seems wrong\n")
        f.write("4. Monitor sensor health metrics (tilt_dynamic_std, displacement_sync_error)\n")
        f.write("5. Investigate any clustering of anomalies in specific time periods\n\n")

        f.write("Next Steps:\n")
        f.write("1. Validate detected anomalies with domain experts\n")
        f.write("2. Fine-tune model parameters based on validation feedback\n")
        f.write("3. Consider ensemble approach combining multiple algorithms\n")
        f.write("4. Deploy model for real-time monitoring\n")

    print(f"  Report saved: {report_file.name}")

    # Save anomalies to CSV for detailed inspection
    anomaly_file = OUTPUT_DIR / 'detected_anomalies.csv'
    df_anomalies.to_csv(anomaly_file, index=False)
    print(f"  Anomalies saved: {anomaly_file.name}")

    # Save full results
    results_file = OUTPUT_DIR / 'full_results_with_predictions.csv'
    df_results.to_csv(results_file, index=False)
    print(f"  Full results saved: {results_file.name}")

def main():
    """Main execution function."""
    print("="*80)
    print("STR128 ANOMALY DETECTION MODEL - TRAINING & EVALUATION")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load data
    df = load_data()

    # Prepare features
    X, features = prepare_features(df)

    # Train model
    contamination = 0.01  # Expect 1% anomalies
    model, scaler, predictions, anomaly_scores, is_anomaly = train_model(X, contamination)

    # Analyze results
    df_results, df_anomalies = analyze_anomalies(df, is_anomaly, anomaly_scores, features)

    # Create visualizations
    create_visualizations(df_results, features)

    # Generate report
    model_params = {
        'algorithm': 'Isolation Forest',
        'contamination': contamination,
        'n_estimators': 100,
        'random_state': 42,
        'features_used': len(features)
    }
    generate_report(df_results, df_anomalies, features, model_params)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  1. 1_timeseries_with_anomalies.png - Time series plots with anomalies marked")
    print("  2. 2_anomaly_scores.png - Anomaly score distributions and trends")
    print("  3. 3_feature_distributions.png - Feature comparisons")
    print("  4. 4_correlation_heatmap.png - Feature correlations")
    print("  5. 5_monthly_anomaly_trend.png - Monthly anomaly rates")
    print("  6. STR128_Anomaly_Detection_Report.txt - Comprehensive text report")
    print("  7. detected_anomalies.csv - List of all detected anomalies")
    print("  8. full_results_with_predictions.csv - Complete dataset with predictions")

    print(f"\nSummary:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Anomalies detected: {is_anomaly.sum():,} ({is_anomaly.sum()/len(df)*100:.2f}%)")
    print(f"  Model ready for presentation!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
