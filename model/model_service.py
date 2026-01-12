"""
Model Service for Bridge Monitoring Dashboard
Loads PRE-TRAINED models for instant anomaly scoring (no training needed).
"""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add current directory to path for importing the model
MODEL_DIR = Path(__file__).parent
sys.path.insert(0, str(MODEL_DIR))

from universal_model_v3 import UniversalModelV3

# Pre-trained models directory (relative to this file's parent)
MODELS_DIR = Path(__file__).parent.parent / "pretrained_models"


class AnomalyScorer:
    """
    Service class for scoring bridge sensor data for anomalies.
    Loads PRE-TRAINED model parameters for instant predictions.
    """

    # Severity color mapping (0-10 scale)
    SEVERITY_COLORS = {
        'normal': '#2ECC71',      # Green (0-2)
        'low': '#F1C40F',         # Yellow (2.1-4)
        'medium': '#E67E22',      # Orange (4.1-6)
        'high': '#E74C3C',        # Red (6.1-8)
        'severe': '#8B0000'       # Dark Red (8.1-10)
    }

    def __init__(self, structure_id: str):
        """
        Initialize scorer for a specific structure.
        Automatically loads pre-trained model if available.
        """
        self.structure_id = structure_id
        self.model = None
        self.params = None
        self.results_df = None
        self.is_trained = False

        # Try to load pre-trained model
        self._load_pretrained()

    def _load_pretrained(self) -> bool:
        """Load pre-trained model parameters if available."""
        model_path = MODELS_DIR / f"{self.structure_id}_model.pkl"

        if not model_path.exists():
            print(f"No pre-trained model found for {self.structure_id}")
            print(f"Run 'python3 pretrain_models.py' first to pre-train all models")
            return False

        try:
            with open(model_path, 'rb') as f:
                self.params = pickle.load(f)

            # Load the full trained model
            self.model = self.params.get('full_model')

            # Save the ORIGINAL feature list from training (before generate_features modifies it)
            self.original_features_used = list(self.params.get('features_used', []))

            if self.model:
                print(f"Loaded pre-trained model for {self.structure_id}")
                print(f"  Trained: {self.params.get('trained_at', 'unknown')}")
                print(f"  Config: {self.params.get('config')}")
                self.is_trained = True
                return True

        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            return False

        return False

    def score_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score new data using pre-trained model parameters.
        This is FAST - uses learned baselines without retraining.

        Args:
            df: DataFrame with columns: timestamp, sensor_id, sensor_type, value

        Returns:
            DataFrame with anomaly scores
        """
        if not self.is_trained:
            raise RuntimeError(
                f"No pre-trained model available for {self.structure_id}. "
                "Run 'python3 pretrain_models.py' first."
            )

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Need to re-detect patterns for the new data period
        # (sensor pairing needs to be detected from current data)
        self.model.detect_sensor_configuration(df)
        self.model.detect_di_pairing(df)
        self.model.detect_tilt_pattern(df)
        self.model.detect_accel_pairing(df)

        # Generate features using detected patterns
        features = self.model.generate_features(df)

        # Get the exact feature columns the scaler was trained on (use SAVED original list)
        # NOTE: generate_features() modifies model.features_used, so we use the saved copy
        scaler_features = self.original_features_used

        # Build features_for_scaling with ONLY the 28 expected columns
        features_for_scaling = pd.DataFrame(index=features.index)
        for col in scaler_features:
            if col in features.columns:
                features_for_scaling[col] = features[col].values
            else:
                # Use 0 for missing columns
                features_for_scaling[col] = 0

        # Fill NaN values appropriately
        for col in features_for_scaling.columns:
            if col.endswith(('_flag', '_stuck', '_violation')):
                # Binary flags: fill with 0
                features_for_scaling[col] = features_for_scaling[col].fillna(0)
            elif col.endswith('_std'):
                # Standard deviation columns: 0 if can't compute (single sensor)
                features_for_scaling[col] = features_for_scaling[col].fillna(0)
            else:
                # Numeric columns: forward fill, then backward fill, then 0
                features_for_scaling[col] = features_for_scaling[col].ffill().bfill().fillna(0)

        features_clean = features_for_scaling.dropna()

        if len(features_clean) == 0:
            return pd.DataFrame()

        # Scale using pre-trained scaler (convert to numpy array to avoid feature name issues)
        X = self.model.scaler.transform(features_clean.values)

        # Get ensemble scores
        scores = self.model.get_ensemble_scores(X)

        # Calculate anomaly scores (0-10)
        features_clean['anomaly_score'] = self.model.calculate_anomaly_score(features_clean, scores)

        # Add flags
        features_clean['if_flag'] = (scores['if_score'] < self.model.threshold_3sigma).astype(int)
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
        features_clean['final_anomaly'] = (features_clean['ensemble_votes'] >= self.model.vote_threshold).astype(int)

        # Range violations
        features_clean['range_anomaly'] = features_clean.get('range_violation', 0)

        # Combined anomaly
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
            fault_type, confidence, _ = self.model.classify_fault(
                features_clean.reset_index(), iloc_idx, score
            )
            features_clean.loc[idx, 'fault_type'] = fault_type
            features_clean.loc[idx, 'fault_confidence'] = confidence

        self.results_df = features_clean
        return features_clean

    def get_score_at_timestamp(self, timestamp: datetime) -> Dict:
        """Get anomaly score at a specific timestamp."""
        if self.results_df is None or len(self.results_df) == 0:
            return {
                'timestamp': timestamp,
                'score': 0.0,
                'severity': 'normal',
                'color': self.SEVERITY_COLORS['normal'],
                'fault_type': 'unknown',
                'fault_confidence': 0.0,
                'is_anomaly': False,
                'ensemble_votes': 0
            }

        # Convert timestamp for comparison
        ts = pd.Timestamp(timestamp)

        # Find exact or nearest timestamp
        if ts in self.results_df.index:
            row = self.results_df.loc[ts]
        else:
            # Find nearest
            idx = self.results_df.index.get_indexer([ts], method='nearest')[0]
            if idx >= 0 and idx < len(self.results_df):
                row = self.results_df.iloc[idx]
            else:
                return {
                    'timestamp': timestamp,
                    'score': 0.0,
                    'severity': 'normal',
                    'color': self.SEVERITY_COLORS['normal'],
                    'fault_type': 'unknown',
                    'fault_confidence': 0.0,
                    'is_anomaly': False,
                    'ensemble_votes': 0
                }

        score = row['anomaly_score']

        return {
            'timestamp': row.name if hasattr(row, 'name') else timestamp,
            'score': float(score),
            'severity': self._get_severity_label(score),
            'color': self._get_color_for_score(score),
            'fault_type': row.get('fault_type', 'unknown'),
            'fault_confidence': float(row.get('fault_confidence', 0.0)),
            'is_anomaly': bool(row.get('combined_anomaly', 0)),
            'ensemble_votes': int(row.get('ensemble_votes', 0))
        }

    def get_scores_in_range(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Get all anomaly scores in a time range."""
        if self.results_df is None or len(self.results_df) == 0:
            return pd.DataFrame()

        # Convert to pandas Timestamp for comparison
        start_ts = pd.Timestamp(start_time)
        end_ts = pd.Timestamp(end_time)

        mask = (self.results_df.index >= start_ts) & (self.results_df.index <= end_ts)
        df_range = self.results_df[mask].copy()

        if len(df_range) > 0:
            df_range['severity'] = df_range['anomaly_score'].apply(self._get_severity_label)
            df_range['color'] = df_range['anomaly_score'].apply(self._get_color_for_score)

        return df_range

    def get_high_severity_events(self, threshold: float = 6.0) -> pd.DataFrame:
        """Get all events above a severity threshold."""
        if self.results_df is None or len(self.results_df) == 0:
            return pd.DataFrame()

        high_severity = self.results_df[self.results_df['anomaly_score'] > threshold].copy()

        if len(high_severity) > 0:
            high_severity['severity'] = high_severity['anomaly_score'].apply(self._get_severity_label)
            high_severity['color'] = high_severity['anomaly_score'].apply(self._get_color_for_score)

        return high_severity.sort_values('anomaly_score', ascending=False)

    def get_statistics(self) -> Dict:
        """Get overall statistics."""
        if self.results_df is None or len(self.results_df) == 0:
            return {
                'total_samples': 0,
                'total_anomalies': 0,
                'score_min': 0,
                'score_max': 0,
                'score_mean': 0,
                'score_median': 0,
                'count_normal': 0,
                'count_low': 0,
                'count_medium': 0,
                'count_high': 0,
                'count_severe': 0,
                'fault_types': {}
            }

        df = self.results_df

        return {
            'total_samples': len(df),
            'total_anomalies': int(df['combined_anomaly'].sum()),
            'score_min': float(df['anomaly_score'].min()),
            'score_max': float(df['anomaly_score'].max()),
            'score_mean': float(df['anomaly_score'].mean()),
            'score_median': float(df['anomaly_score'].median()),
            'count_normal': int((df['anomaly_score'] <= 2).sum()),
            'count_low': int(((df['anomaly_score'] > 2) & (df['anomaly_score'] <= 4)).sum()),
            'count_medium': int(((df['anomaly_score'] > 4) & (df['anomaly_score'] <= 6)).sum()),
            'count_high': int(((df['anomaly_score'] > 6) & (df['anomaly_score'] <= 8)).sum()),
            'count_severe': int((df['anomaly_score'] > 8).sum()),
            'fault_types': df[df['combined_anomaly'] == 1]['fault_type'].value_counts().to_dict()
        }

    def get_training_info(self) -> Dict:
        """Get info about the pre-trained model."""
        if not self.params:
            return {}

        return {
            'trained_at': self.params.get('trained_at'),
            'config': self.params.get('config'),
            'data_range': self.params.get('data_range'),
            'training_stats': self.params.get('training_stats'),
            'features': len(self.params.get('features_used', [])),
        }

    def _get_severity_label(self, score: float) -> str:
        """Get severity label for a score."""
        if score <= 2:
            return 'normal'
        elif score <= 4:
            return 'low'
        elif score <= 6:
            return 'medium'
        elif score <= 8:
            return 'high'
        else:
            return 'severe'

    def _get_color_for_score(self, score: float) -> str:
        """Get color for a score value."""
        severity = self._get_severity_label(score)
        return self.SEVERITY_COLORS[severity]


def is_pretrained_available(structure_id: str) -> bool:
    """Check if pre-trained model exists for a structure."""
    model_path = MODELS_DIR / f"{structure_id}_model.pkl"
    return model_path.exists()


def get_available_pretrained() -> List[str]:
    """Get list of structures with pre-trained models."""
    if not MODELS_DIR.exists():
        return []

    models = list(MODELS_DIR.glob("*_model.pkl"))
    return [m.stem.replace('_model', '') for m in models]


if __name__ == "__main__":
    # Test the model service
    print("Testing Model Service with Pre-trained Models...")

    available = get_available_pretrained()
    print(f"\nPre-trained models available: {available}")

    if not available:
        print("\nNo pre-trained models found!")
        print("Run 'python3 pretrain_models.py' to create them.")
    else:
        # Test with first available
        test_struct = available[0]
        print(f"\nTesting with {test_struct}...")

        scorer = AnomalyScorer(test_struct)
        info = scorer.get_training_info()
        print(f"Training info: {info}")
