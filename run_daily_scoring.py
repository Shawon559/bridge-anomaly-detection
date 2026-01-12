"""
Daily Sensor Anomaly Scoring Script
====================================
Run this script daily to score sensor data for anomalies.

Usage:
    python run_daily_scoring.py STR122 path/to/data.xlsx

Output:
    - Console: Summary of anomalies detected
    - File: {structure_id}_results_{date}.csv with detailed scores
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add model directory to path
MODEL_DIR = Path(__file__).parent / "model"
sys.path.insert(0, str(MODEL_DIR))

from universal_model_v3 import UniversalModelV3

# Pre-trained models directory
PRETRAINED_DIR = Path(__file__).parent / "pretrained_models"


def load_pretrained_model(structure_id: str):
    """Load pre-trained model for a structure."""
    model_path = PRETRAINED_DIR / f"{structure_id}_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No pre-trained model found for {structure_id}.\n"
            f"Available models: {list(PRETRAINED_DIR.glob('*_model.pkl'))}"
        )

    with open(model_path, 'rb') as f:
        params = pickle.load(f)

    model = params.get('full_model')
    original_features = list(params.get('features_used', []))

    print(f"Loaded pre-trained model for {structure_id}")
    print(f"  Trained: {params.get('trained_at', 'unknown')}")
    print(f"  Config: {params.get('config')}")

    return model, original_features, params


def score_data(model, original_features, df: pd.DataFrame) -> pd.DataFrame:
    """
    Score sensor data using pre-trained model.

    Args:
        model: Pre-trained UniversalModelV3 instance
        original_features: List of features the model was trained on
        df: DataFrame with columns: timestamp, sensor_id, sensor_type, value

    Returns:
        DataFrame with anomaly scores and classifications
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Re-detect sensor patterns for this data period
    model.detect_sensor_configuration(df)
    model.detect_di_pairing(df)
    model.detect_tilt_pattern(df)
    model.detect_accel_pairing(df)

    # Generate features
    features = model.generate_features(df)

    # Build feature matrix with exact columns expected by scaler
    features_for_scaling = pd.DataFrame(index=features.index)
    for col in original_features:
        if col in features.columns:
            features_for_scaling[col] = features[col].values
        else:
            features_for_scaling[col] = 0

    # Fill NaN values
    for col in features_for_scaling.columns:
        if col.endswith(('_flag', '_stuck', '_violation')):
            features_for_scaling[col] = features_for_scaling[col].fillna(0)
        elif col.endswith('_std'):
            features_for_scaling[col] = features_for_scaling[col].fillna(0)
        else:
            features_for_scaling[col] = features_for_scaling[col].ffill().bfill().fillna(0)

    features_clean = features_for_scaling.dropna()

    if len(features_clean) == 0:
        print("WARNING: No valid data to score")
        return pd.DataFrame()

    # Scale and score
    X = model.scaler.transform(features_clean.values)
    scores = model.get_ensemble_scores(X)

    # Calculate anomaly scores (0-10)
    features_clean['anomaly_score'] = model.calculate_anomaly_score(features_clean, scores)

    # Add ensemble flags
    features_clean['if_flag'] = (scores['if_score'] < model.threshold_3sigma).astype(int)
    features_clean['lof_flag'] = (scores['lof_score'] < np.percentile(scores['lof_score'], 3)).astype(int)
    features_clean['svm_flag'] = (scores['svm_score'] < np.percentile(scores['svm_score'], 3)).astype(int)

    # Threshold flag
    features_clean['threshold_flag'] = 0
    for col in features_clean.columns:
        if col.endswith('_zscore_flag'):
            features_clean['threshold_flag'] |= features_clean[col]

    # Ensemble voting
    vote_cols = ['if_flag', 'lof_flag', 'svm_flag', 'threshold_flag']
    features_clean['ensemble_votes'] = features_clean[vote_cols].sum(axis=1)
    features_clean['final_anomaly'] = (features_clean['ensemble_votes'] >= model.vote_threshold).astype(int)

    # Range violations
    features_clean['range_anomaly'] = features_clean.get('range_violation', 0)
    features_clean['combined_anomaly'] = (
        (features_clean['final_anomaly'] == 1) |
        (features_clean['range_anomaly'] == 1)
    ).astype(int)

    # Classify faults
    features_clean['fault_type'] = 'normal'
    features_clean['fault_confidence'] = 0.0

    anomaly_mask = features_clean['combined_anomaly'] == 1
    for idx in features_clean[anomaly_mask].index:
        iloc_idx = features_clean.index.get_loc(idx)
        score = features_clean.loc[idx, 'anomaly_score']
        fault_type, confidence, _ = model.classify_fault(
            features_clean.reset_index(), iloc_idx, score
        )
        features_clean.loc[idx, 'fault_type'] = fault_type
        features_clean.loc[idx, 'fault_confidence'] = confidence

    return features_clean


def get_severity_label(score: float) -> str:
    """Get severity label for a score."""
    if score <= 2:
        return 'NORMAL'
    elif score <= 4:
        return 'LOW'
    elif score <= 6:
        return 'MEDIUM'
    elif score <= 8:
        return 'HIGH'
    else:
        return 'SEVERE'


def print_summary(results_df: pd.DataFrame, structure_id: str):
    """Print summary of scoring results."""
    print("\n" + "="*60)
    print(f"ANOMALY SCORING RESULTS: {structure_id}")
    print("="*60)

    print(f"\nData Summary:")
    print(f"  Time range: {results_df.index.min()} to {results_df.index.max()}")
    print(f"  Total samples: {len(results_df):,}")

    print(f"\nAnomaly Score Distribution:")
    print(f"  Min:    {results_df['anomaly_score'].min():.1f}")
    print(f"  Max:    {results_df['anomaly_score'].max():.1f}")
    print(f"  Mean:   {results_df['anomaly_score'].mean():.2f}")
    print(f"  Median: {results_df['anomaly_score'].median():.2f}")

    print(f"\nSeverity Breakdown:")
    print(f"  0-2 (Normal):    {(results_df['anomaly_score'] <= 2).sum():,}")
    print(f"  2-4 (Low):       {((results_df['anomaly_score'] > 2) & (results_df['anomaly_score'] <= 4)).sum():,}")
    print(f"  4-6 (Medium):    {((results_df['anomaly_score'] > 4) & (results_df['anomaly_score'] <= 6)).sum():,}")
    print(f"  6-8 (High):      {((results_df['anomaly_score'] > 6) & (results_df['anomaly_score'] <= 8)).sum():,}")
    print(f"  8-10 (Severe):   {(results_df['anomaly_score'] > 8).sum():,}")

    # High severity events
    high_severity = results_df[results_df['anomaly_score'] > 6].copy()
    if len(high_severity) > 0:
        print(f"\nHIGH SEVERITY EVENTS ({len(high_severity)}):")
        high_severity = high_severity.sort_values('anomaly_score', ascending=False)
        for idx, row in high_severity.head(10).iterrows():
            print(f"  {idx}: Score {row['anomaly_score']:.1f} ({get_severity_label(row['anomaly_score'])}) - {row['fault_type']}")
    else:
        print(f"\nNo high severity events detected.")

    print("="*60)


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python run_daily_scoring.py <STRUCTURE_ID> <DATA_FILE>")
        print("")
        print("Arguments:")
        print("  STRUCTURE_ID  Structure identifier (e.g., STR122)")
        print("  DATA_FILE     Path to Excel file with sensor data")
        print("")
        print("Example:")
        print("  python run_daily_scoring.py STR122 data/STR122_merged.xlsx")
        print("")
        print("Available pre-trained models:")
        for pkl in sorted(PRETRAINED_DIR.glob("*_model.pkl")):
            print(f"  - {pkl.stem.replace('_model', '')}")
        sys.exit(1)

    structure_id = sys.argv[1].upper()
    data_file = sys.argv[2]

    # Validate inputs
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found: {data_file}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"DAILY ANOMALY SCORING")
    print(f"{'='*60}")
    print(f"Structure: {structure_id}")
    print(f"Data file: {data_file}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load model
    try:
        model, original_features, params = load_pretrained_model(structure_id)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Load data
    print(f"\nLoading data...")
    df = pd.read_excel(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Loaded {len(df):,} rows")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Score data
    print(f"\nScoring data...")
    results_df = score_data(model, original_features, df)

    if results_df.empty:
        print("ERROR: No results generated")
        sys.exit(1)

    # Print summary
    print_summary(results_df, structure_id)

    # Save results
    output_file = f"{structure_id}_results_{datetime.now().strftime('%Y%m%d')}.csv"

    # Select key columns for output
    output_cols = ['anomaly_score', 'ensemble_votes', 'combined_anomaly', 'fault_type', 'fault_confidence']
    output_cols = [c for c in output_cols if c in results_df.columns]

    results_df[output_cols].to_csv(output_file)
    print(f"\nResults saved to: {output_file}")

    # Return exit code based on severity
    max_score = results_df['anomaly_score'].max()
    if max_score > 8:
        print("\n*** SEVERE ANOMALIES DETECTED - IMMEDIATE ATTENTION REQUIRED ***")
        sys.exit(2)
    elif max_score > 6:
        print("\n** High severity anomalies detected - review recommended **")
        sys.exit(1)
    else:
        print("\nNo critical anomalies detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
