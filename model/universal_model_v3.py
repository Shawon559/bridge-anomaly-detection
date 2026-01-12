"""
Universal Adaptive Anomaly Detection Model v3.0

FEATURES:

1. TILT SENSOR PAIRING (2+2 Pattern):
   - Uses ground truth axis mapping from company data
   - 2 X-axis sensors + 2 Y-axis sensors per structure
   - Hardcoded mapping for all 21 structures (STR122-STR199)

2. DISPLACEMENT SENSOR PAIRING:
   - Temporal stability method to detect paired sensors
   - Criteria: within-period std < 20mm, drift < 50mm, rolling std median < 5mm

3. SEASONAL BASELINE LEARNING:
   - Learns temperature patterns from data itself
   - Calculates monthly baselines for each sensor type
   - Adjusts anomaly detection based on seasonal norms

4. ANOMALY SCORING (0-10):
   - 0 = completely normal
   - 10 = most severe anomaly
   - Based on ensemble confidence + deviation magnitude

5. FAULT CLASSIFICATION:
   - Sensor Fault: Single sensor deviation
   - Structural Fault: Multiple correlated sensors affected
   - Environmental: Seasonal/temperature-related variations

"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


class UniversalModelV3:
    """
    Universal Adaptive Anomaly Detection Model v3.0

    Features:
    - Tilt pairing: 2+2 pattern using ground truth axis mapping
    - Displacement pairing: Temporal stability detection
    - Seasonal baseline learning from data
    - Anomaly scoring (0-10)
    - Fault classification (sensor/structural/environmental)
    """

    # Official sensor ranges from company
    SENSOR_RANGES = {
        'accelerometer': {'p2p': (-1.0, 1.0), 'rms': (-1.0, 1.0)},
        'displacement': {'value': (0.0, 500.0)},
        'tilt': {'value': (-5.0, 5.0)},
        'temperature_probe': {'value': (-40.0, 125.0)}
    }

    # =========================================================================
    # GROUND TRUTH AXIS MAPPING FROM COMPANY DATA
    # All structures have 2+2 pattern: 2 X-axis sensors, 2 Y-axis sensors
    # Position 1 has X and Y, Position 2 has X and Y
    # =========================================================================
    TILT_AXIS_MAPPING = {
        'STR122': {'x_axis': ['TI535', 'TI537'], 'y_axis': ['TI536', 'TI538']},
        'STR124': {'x_axis': ['TI541', 'TI543'], 'y_axis': ['TI542', 'TI544']},
        'STR127': {'x_axis': ['TI545', 'TI547'], 'y_axis': ['TI546', 'TI548']},
        'STR128': {'x_axis': ['TI551', 'TI553'], 'y_axis': ['TI552', 'TI554']},
        'STR129': {'x_axis': ['TI557', 'TI559'], 'y_axis': ['TI558', 'TI560']},
        'STR178': {'x_axis': ['TI608', 'TI610'], 'y_axis': ['TI609', 'TI611']},
        'STR179': {'x_axis': ['TI616', 'TI618'], 'y_axis': ['TI617', 'TI619']},
        'STR180': {'x_axis': ['TI624', 'TI626'], 'y_axis': ['TI625', 'TI627']},
        'STR181': {'x_axis': ['TI632', 'TI634'], 'y_axis': ['TI633', 'TI635']},
        'STR182': {'x_axis': ['TI640', 'TI642'], 'y_axis': ['TI641', 'TI643']},
        'STR183': {'x_axis': ['TI648', 'TI650'], 'y_axis': ['TI649', 'TI651']},
        'STR184': {'x_axis': ['TI656', 'TI658'], 'y_axis': ['TI657', 'TI659']},
        'STR199': {'x_axis': ['TI807', 'TI812'], 'y_axis': ['TI808', 'TI813']},
    }

    def __init__(self, structure_id=None, ensemble_vote_threshold=2):
        """
        Args:
            structure_id: Structure identifier (e.g., 'STR122') for axis mapping lookup
            ensemble_vote_threshold: Minimum votes needed to flag as anomaly (default 2 of 4)
        """
        self.structure_id = structure_id
        self.vote_threshold = ensemble_vote_threshold
        self.config = None
        self.sensor_patterns = {}
        self.features_used = []
        self.scaler = StandardScaler()

        # Ensemble models
        self.models = {
            'isolation_forest': None,
            'lof': None,
            'ocsvm': None
        }

        # Thresholds
        self.threshold_3sigma = None

        # Seasonal baselines (learned from data)
        self.seasonal_baselines = {}

        # Anomaly scores storage
        self.anomaly_scores = None

    # =========================================================================
    # STEP 1: SENSOR CONFIGURATION DETECTION
    # =========================================================================

    def detect_sensor_configuration(self, df):
        """Detect what sensors are available"""
        sensor_types = df['sensor_type'].unique()
        sensors_by_type = {}

        for stype in sensor_types:
            sensors = df[df['sensor_type'] == stype]['sensor_id'].unique().tolist()
            sensors_by_type[stype] = sensors

        has_di = 'displacement' in sensors_by_type and len(sensors_by_type.get('displacement', [])) >= 2
        has_ti = 'tilt' in sensors_by_type and len(sensors_by_type.get('tilt', [])) >= 2
        has_ac = 'accelerometer' in sensors_by_type and len(sensors_by_type.get('accelerometer', [])) >= 1
        has_tp = 'temperature_probe' in sensors_by_type

        if has_di and has_ti and has_ac:
            config = 'A'  # Full suite
        elif has_di and has_ti:
            config = 'B'  # DI + TI
        elif has_ac and not has_di and not has_ti:
            config = 'C'  # AC only
        elif has_ti and not has_di:
            config = 'D'  # TI only
        else:
            config = 'UNKNOWN'

        self.config = config
        self.sensors_by_type = sensors_by_type

        print(f"Configuration detected: {config}")
        print(f"  Sensors: {sensors_by_type}")

        return config, sensors_by_type

    # =========================================================================
    # STEP 2: SEASONAL BASELINE LEARNING
    # =========================================================================

    def learn_seasonal_baselines(self, df):
        """Learn seasonal patterns from the data itself"""
        print("\nLearning seasonal baselines from data...")

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['month'] = df['timestamp'].dt.month
        df['season'] = df['month'].apply(self._get_season)

        baselines = {}

        # Temperature baselines by month
        if 'temperature_probe' in self.sensors_by_type:
            df_temp = df[df['sensor_type'] == 'temperature_probe']
            temp_monthly = df_temp.groupby('month')['value'].agg(['mean', 'std']).to_dict('index')
            baselines['temperature'] = {
                'monthly': temp_monthly,
                'overall_mean': df_temp['value'].mean(),
                'overall_std': df_temp['value'].std()
            }
            print(f"  Temperature: learned 12-month baseline (mean: {baselines['temperature']['overall_mean']:.1f}°C)")

        # Displacement baselines by month (to capture seasonal expansion/contraction)
        if 'displacement' in self.sensors_by_type:
            df_di = df[df['sensor_type'] == 'displacement']
            di_monthly = df_di.groupby('month')['value'].agg(['mean', 'std']).to_dict('index')
            baselines['displacement'] = {
                'monthly': di_monthly,
                'overall_mean': df_di['value'].mean(),
                'overall_std': df_di['value'].std()
            }
            print(f"  Displacement: learned 12-month baseline (mean: {baselines['displacement']['overall_mean']:.1f}mm)")

        # Tilt baselines by month
        if 'tilt' in self.sensors_by_type:
            df_ti = df[df['sensor_type'] == 'tilt']
            ti_monthly = df_ti.groupby('month')['value'].agg(['mean', 'std']).to_dict('index')
            baselines['tilt'] = {
                'monthly': ti_monthly,
                'overall_mean': df_ti['value'].mean(),
                'overall_std': df_ti['value'].std()
            }
            print(f"  Tilt: learned 12-month baseline (mean: {baselines['tilt']['overall_mean']:.2f}°)")

        # Accelerometer baselines
        if 'accelerometer' in self.sensors_by_type:
            df_ac = df[df['sensor_type'] == 'accelerometer']
            p2p_monthly = df_ac.groupby('month')['p2p'].agg(['mean', 'std']).to_dict('index')
            rms_monthly = df_ac.groupby('month')['rms'].agg(['mean', 'std']).to_dict('index')
            baselines['accelerometer'] = {
                'p2p_monthly': p2p_monthly,
                'rms_monthly': rms_monthly,
                'p2p_mean': df_ac['p2p'].mean(),
                'rms_mean': df_ac['rms'].mean()
            }
            print(f"  Accelerometer: learned 12-month baseline")

        # Calculate temperature-displacement correlation
        if 'temperature' in baselines and 'displacement' in baselines:
            df_merged = df[df['sensor_type'].isin(['temperature_probe', 'displacement'])]
            df_pivot = df_merged.pivot_table(
                index='timestamp', columns='sensor_type', values='value', aggfunc='mean'
            )
            if 'temperature_probe' in df_pivot.columns and 'displacement' in df_pivot.columns:
                corr = df_pivot['temperature_probe'].corr(df_pivot['displacement'])
                baselines['temp_displacement_correlation'] = corr
                print(f"  Temperature-Displacement correlation: {corr:.3f}")

        self.seasonal_baselines = baselines
        return baselines

    def _get_season(self, month):
        """Map month to season"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'

    def get_seasonal_deviation(self, value, sensor_type, month, metric='value'):
        """Calculate how much a value deviates from seasonal baseline"""
        if sensor_type not in self.seasonal_baselines:
            return 0

        baseline = self.seasonal_baselines[sensor_type]

        if sensor_type == 'accelerometer':
            monthly_data = baseline.get(f'{metric}_monthly', {})
        else:
            monthly_data = baseline.get('monthly', {})

        if month not in monthly_data:
            return 0

        expected_mean = monthly_data[month]['mean']
        expected_std = monthly_data[month]['std']

        if expected_std == 0 or np.isnan(expected_std):
            return 0

        # Return z-score relative to seasonal baseline
        return (value - expected_mean) / expected_std

    # =========================================================================
    # STEP 3: SENSOR PAIRING DETECTION
    # =========================================================================

    def detect_di_pairing(self, df):
        """
        Detect displacement sensor pairing using TEMPORAL STABILITY METHOD.

        Criteria (all 3 must be met):
        1. Within-period std < 20mm (early, middle, late periods)
        2. Total drift < 50mm
        3. Rolling window std median < 5mm
        """
        if 'displacement' not in self.sensors_by_type:
            return None

        di_sensors = self.sensors_by_type['displacement']
        if len(di_sensors) < 2:
            return None

        df_di = df[df['sensor_type'] == 'displacement'].copy()

        # Filter to Nov 2023 onward (exclude calibration period)
        start_date = pd.Timestamp('2023-11-01')
        if df_di['timestamp'].min() < start_date:
            df_di = df_di[df_di['timestamp'] >= start_date].copy()

        best_pair = None
        best_within_std = float('inf')
        best_metrics = None

        for s1, s2 in combinations(di_sensors, 2):
            df_s1 = df_di[df_di['sensor_id'] == s1][['timestamp', 'value']].set_index('timestamp')
            df_s2 = df_di[df_di['sensor_id'] == s2][['timestamp', 'value']].set_index('timestamp')

            df_merged = df_s1.join(df_s2, lsuffix='_1', rsuffix='_2', how='inner')

            if len(df_merged) < 100:
                continue

            # Filter extreme values (>100mm likely malfunction)
            df_merged = df_merged[(np.abs(df_merged['value_1']) <= 100) &
                                   (np.abs(df_merged['value_2']) <= 100)]

            if len(df_merged) < 100:
                continue

            combined = df_merged['value_1'] + df_merged['value_2']

            # TEMPORAL STABILITY ANALYSIS
            n = len(combined)
            period_size = min(1000, n // 3)

            if period_size < 100:
                continue

            early = combined.iloc[:period_size]
            middle = combined.iloc[n//2 - period_size//2 : n//2 + period_size//2]
            late = combined.iloc[-period_size:]

            early_std = early.std()
            middle_std = middle.std()
            late_std = late.std()

            early_mean = early.mean()
            late_mean = late.mean()
            total_drift = abs(late_mean - early_mean)

            within_period_std_avg = (early_std + middle_std + late_std) / 3

            # Rolling window analysis
            window_size = min(100, n // 10)
            rolling_std = combined.rolling(window=window_size, min_periods=1).std()
            rolling_std_median = rolling_std.median()

            # PAIRING CRITERIA
            criterion1 = within_period_std_avg < 20  # Within-period stability
            criterion2 = total_drift < 50            # Acceptable drift
            criterion3 = rolling_std_median < 5      # Consistent short-term stability

            is_paired = criterion1 and criterion2 and criterion3

            if is_paired and within_period_std_avg < best_within_std:
                best_within_std = within_period_std_avg
                best_pair = (s1, s2)
                best_metrics = {
                    'within_std': within_period_std_avg,
                    'drift': total_drift,
                    'rolling_median': rolling_std_median
                }

        if best_pair:
            self.sensor_patterns['di_pair'] = best_pair
            print(f"  DI Pairing: {best_pair[0]} + {best_pair[1]} "
                  f"(within-std: {best_metrics['within_std']:.2f}mm, "
                  f"drift: {best_metrics['drift']:.2f}mm, "
                  f"rolling-med: {best_metrics['rolling_median']:.2f}mm)")
            return best_pair

        return None

    def detect_tilt_pattern(self, df):
        """
        Detect tilt sensor pattern using GROUND TRUTH AXIS MAPPING from company data.

        All structures have 2+2 pattern: 2 X-axis sensors, 2 Y-axis sensors
        - X-axis sensors measure tilt in one direction
        - Y-axis sensors measure tilt in perpendicular direction

        Falls back to correlation-based detection only if structure not in mapping.
        """
        if 'tilt' not in self.sensors_by_type:
            return None

        ti_sensors = self.sensors_by_type['tilt']
        n_sensors = len(ti_sensors)

        if n_sensors < 2:
            return None

        # =====================================================================
        # PRIMARY: Use hardcoded axis mapping from company data
        # =====================================================================
        if self.structure_id and self.structure_id in self.TILT_AXIS_MAPPING:
            axis_map = self.TILT_AXIS_MAPPING[self.structure_id]

            # Filter to sensors that exist in the data
            x_sensors = [s for s in axis_map['x_axis'] if s in ti_sensors]
            y_sensors = [s for s in axis_map['y_axis'] if s in ti_sensors]

            if len(x_sensors) >= 1 and len(y_sensors) >= 1:
                # Use 2+2 pattern with X as group, Y as different
                pattern = {
                    'type': '2+2',
                    'group': x_sensors,           # X-axis sensors
                    'different': y_sensors,       # Y-axis sensors
                    'group_corr': 1.0,            # Not used (hardcoded)
                    'cross_corr': 0.0,            # Not used (hardcoded)
                    'score': 1.0,
                    'source': 'axis_mapping'
                }

                self.sensor_patterns['tilt'] = pattern
                print(f"  Tilt pattern: 2+2 (from axis mapping)")
                print(f"    X-axis: {x_sensors}")
                print(f"    Y-axis: {y_sensors}")
                print(f"    Total: {len(x_sensors) + len(y_sensors)} sensors (no sensors dropped)")
                return pattern

        # =====================================================================
        # FALLBACK: Correlation-based detection for unknown structures
        # =====================================================================
        print(f"  WARNING: Structure {self.structure_id} not in axis mapping, using correlation fallback")

        df_ti = df[df['sensor_type'] == 'tilt'].copy()
        df_pivot = df_ti.pivot_table(
            index='timestamp', columns='sensor_id', values='value', aggfunc='mean'
        )

        if len(df_pivot) < 100:
            return None

        available_sensors = [s for s in ti_sensors if s in df_pivot.columns]
        if len(available_sensors) < 2:
            return None

        corr_matrix = df_pivot[available_sensors].corr()

        # For 4 sensors, try 2+2 pattern first (since we know all structures use 2+2)
        if n_sensors >= 4:
            best_pattern = None
            best_score = -999

            for combo in combinations(range(len(available_sensors)), 2):
                group1_indices = list(combo)
                group2_indices = [i for i in range(len(available_sensors)) if i not in group1_indices][:2]

                if len(group2_indices) < 2:
                    continue

                group1 = [available_sensors[i] for i in group1_indices]
                group2 = [available_sensors[i] for i in group2_indices]

                corr1 = abs(corr_matrix.loc[group1[0], group1[1]]) if group1[0] in corr_matrix.columns and group1[1] in corr_matrix.columns else 0
                corr2 = abs(corr_matrix.loc[group2[0], group2[1]]) if group2[0] in corr_matrix.columns and group2[1] in corr_matrix.columns else 0

                cross_corrs = []
                for s1 in group1:
                    for s2 in group2:
                        if s1 in corr_matrix.columns and s2 in corr_matrix.columns:
                            cross_corrs.append(abs(corr_matrix.loc[s1, s2]))
                avg_cross = np.mean(cross_corrs) if cross_corrs else 0

                within_avg = (corr1 + corr2) / 2
                score = within_avg - avg_cross

                if score > best_score:
                    best_score = score
                    best_pattern = {
                        'type': '2+2',
                        'group': group1,
                        'different': group2,
                        'group_corr': within_avg,
                        'cross_corr': avg_cross,
                        'score': score,
                        'source': 'correlation_fallback'
                    }

            if best_pattern:
                self.sensor_patterns['tilt'] = best_pattern
                print(f"  Tilt pattern: {best_pattern['type']} (correlation fallback)")
                print(f"    Group 1: {best_pattern['group']} (corr: {best_pattern['group_corr']:.3f})")
                print(f"    Group 2: {best_pattern['different']} (cross-corr: {best_pattern['cross_corr']:.3f})")
                return best_pattern

        return None

    def detect_accel_pairing(self, df):
        """Detect accelerometer pairing"""
        if 'accelerometer' not in self.sensors_by_type:
            return None

        ac_sensors = self.sensors_by_type['accelerometer']
        if len(ac_sensors) < 2:
            return None

        df_ac = df[df['sensor_type'] == 'accelerometer'].copy()

        df_pivot = df_ac.pivot_table(
            index='timestamp', columns='sensor_id', values=['p2p', 'rms'], aggfunc='mean'
        )

        if len(df_pivot) < 100:
            return None

        s1, s2 = ac_sensors[0], ac_sensors[1]

        p2p_corr = 0
        rms_corr = 0

        if ('p2p', s1) in df_pivot.columns and ('p2p', s2) in df_pivot.columns:
            p2p_corr = df_pivot[('p2p', s1)].corr(df_pivot[('p2p', s2)])

        if ('rms', s1) in df_pivot.columns and ('rms', s2) in df_pivot.columns:
            rms_corr = df_pivot[('rms', s1)].corr(df_pivot[('rms', s2)])

        pattern = {
            'sensors': (s1, s2),
            'p2p_corr': p2p_corr,
            'rms_corr': rms_corr,
            'p2p_paired': p2p_corr > 0.7 if not np.isnan(p2p_corr) else False,
            'rms_paired': rms_corr > 0.7 if not np.isnan(rms_corr) else False
        }

        if pattern['p2p_paired'] or pattern['rms_paired']:
            self.sensor_patterns['accel'] = pattern
            print(f"  Accel pairing: {s1} + {s2} (p2p: {p2p_corr:.3f}, rms: {rms_corr:.3f})")
            return pattern

        return None

    # =========================================================================
    # STEP 4: FEATURE GENERATION
    # =========================================================================

    def generate_features(self, df):
        """Generate features based on detected patterns"""
        print("\nGenerating features...")

        features = pd.DataFrame()
        features['timestamp'] = pd.to_datetime(df['timestamp'].unique())
        features = features.set_index('timestamp')
        features['month'] = features.index.month

        # DI Features
        if 'di_pair' in self.sensor_patterns:
            s1, s2 = self.sensor_patterns['di_pair']
            df_di = df[df['sensor_type'] == 'displacement']

            df_s1 = df_di[df_di['sensor_id'] == s1][['timestamp', 'value']].set_index('timestamp')
            df_s2 = df_di[df_di['sensor_id'] == s2][['timestamp', 'value']].set_index('timestamp')

            df_merged = df_s1.join(df_s2, lsuffix='_1', rsuffix='_2', how='inner')
            df_merged['displacement_combined'] = df_merged['value_1'] + df_merged['value_2']
            df_merged['displacement_sync_error'] = abs(df_merged['value_1']) - abs(df_merged['value_2'])

            features = features.join(df_merged[['displacement_combined', 'displacement_sync_error']])
            features['displacement_rate'] = features['displacement_combined'].diff()

            self.features_used.extend(['displacement_combined', 'displacement_rate', 'displacement_sync_error'])

        # TI Features - handles all pattern types
        if 'tilt' in self.sensor_patterns:
            pattern = self.sensor_patterns['tilt']
            df_ti = df[df['sensor_type'] == 'tilt']

            # Group sensors (axis 1)
            group_sensors = pattern['group']
            df_group = df_ti[df_ti['sensor_id'].isin(group_sensors)]
            group_pivot = df_group.pivot_table(
                index='timestamp', columns='sensor_id', values='value', aggfunc='mean'
            )

            features['tilt_group_avg'] = group_pivot.mean(axis=1)
            features['tilt_group_std'] = group_pivot.std(axis=1)

            # Different sensors (axis 2)
            diff_sensors = pattern['different']
            df_diff = df_ti[df_ti['sensor_id'].isin(diff_sensors)]
            diff_pivot = df_diff.pivot_table(
                index='timestamp', columns='sensor_id', values='value', aggfunc='mean'
            )

            features['tilt_different_avg'] = diff_pivot.mean(axis=1)
            if len(diff_sensors) > 1:
                features['tilt_different_std'] = diff_pivot.std(axis=1)
                self.features_used.append('tilt_different_std')

            features['tilt_group_rate'] = features['tilt_group_avg'].diff()
            features['tilt_axis_difference'] = features['tilt_group_avg'] - features['tilt_different_avg']

            self.features_used.extend(['tilt_group_avg', 'tilt_group_std', 'tilt_different_avg',
                                       'tilt_group_rate', 'tilt_axis_difference'])

        # AC Features
        if 'accel' in self.sensor_patterns:
            pattern = self.sensor_patterns['accel']
            s1, s2 = pattern['sensors']
            df_ac = df[df['sensor_type'] == 'accelerometer']

            df_pivot = df_ac.pivot_table(
                index='timestamp', columns='sensor_id', values=['p2p', 'rms'], aggfunc='mean'
            )

            if pattern['p2p_paired']:
                if ('p2p', s1) in df_pivot.columns and ('p2p', s2) in df_pivot.columns:
                    features['p2p_combined'] = (df_pivot[('p2p', s1)] + df_pivot[('p2p', s2)]) / 2
                    features['p2p_rate'] = features['p2p_combined'].diff()
                    self.features_used.extend(['p2p_combined', 'p2p_rate'])

            if pattern['rms_paired']:
                if ('rms', s1) in df_pivot.columns and ('rms', s2) in df_pivot.columns:
                    features['rms_combined'] = (df_pivot[('rms', s1)] + df_pivot[('rms', s2)]) / 2
                    self.features_used.append('rms_combined')

        # Temperature Features
        if 'temperature_probe' in self.sensors_by_type:
            tp_sensor = self.sensors_by_type['temperature_probe'][0]
            df_tp = df[df['sensor_id'] == tp_sensor][['timestamp', 'value']].set_index('timestamp')
            features = features.join(df_tp.rename(columns={'value': 'temperature'}))
            features['temp_rate'] = features['temperature'].diff()
            self.features_used.extend(['temperature', 'temp_rate'])

        # Temporal features
        features['hour'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        self.features_used.extend(['hour', 'day_of_week', 'month'])

        # Add seasonal deviation features
        features = self.add_seasonal_deviation_features(features)

        # Add sensor health features
        features = self.add_sensor_health_features(features, df)
        features = self.add_threshold_flags(features, df)
        features = self.detect_sensor_stuck(features)
        features = self.add_company_range_flags(features, df)

        print(f"  Total features: {len(self.features_used)}")

        return features

    def add_seasonal_deviation_features(self, features):
        """Add features showing deviation from seasonal baseline"""

        if 'temperature' in features.columns and 'temperature' in self.seasonal_baselines:
            # Calculate temperature deviation from monthly baseline
            features['temp_seasonal_deviation'] = features.apply(
                lambda row: self.get_seasonal_deviation(
                    row['temperature'], 'temperature', row['month']
                ) if pd.notna(row['temperature']) else 0,
                axis=1
            )
            self.features_used.append('temp_seasonal_deviation')

        if 'displacement_combined' in features.columns and 'displacement' in self.seasonal_baselines:
            # Calculate displacement deviation from monthly baseline
            features['displacement_seasonal_deviation'] = features.apply(
                lambda row: self.get_seasonal_deviation(
                    row['displacement_combined'], 'displacement', row['month']
                ) if pd.notna(row['displacement_combined']) else 0,
                axis=1
            )
            self.features_used.append('displacement_seasonal_deviation')

        return features

    def add_sensor_health_features(self, features, df):
        """Add sensor health features"""
        if 'battery' in df.columns:
            df_battery = df.groupby('timestamp')['battery'].mean()
            features = features.join(df_battery.rename('battery'))
            features['low_battery'] = (features['battery'] < 20).astype(int)
            self.features_used.append('low_battery')

        return features

    def add_threshold_flags(self, features, df):
        """Add threshold-based flags"""
        for col in features.columns:
            if col in ['displacement_combined', 'tilt_group_avg', 'p2p_combined', 'rms_combined']:
                mean_val = features[col].mean()
                std_val = features[col].std()
                if std_val > 0:
                    features[f'{col}_zscore'] = (features[col] - mean_val) / std_val
                    features[f'{col}_zscore_flag'] = (abs(features[f'{col}_zscore']) > 3).astype(int)
                    self.features_used.append(f'{col}_zscore_flag')

        return features

    def add_company_range_flags(self, features, df):
        """Add company range violation flags"""
        features['range_violation'] = 0

        # Displacement range: ±50mm relative to BASELINE (company spec for movement)
        # Use MEDIAN for baseline - robust to outliers/sensor malfunctions
        if 'displacement_combined' in features.columns:
            disp_baseline = features['displacement_combined'].median()  # Median is robust to outliers
            disp_deviation = (features['displacement_combined'] - disp_baseline).abs()
            displacement_violation = disp_deviation > 50  # More than 50mm from baseline
            features['displacement_range_flag'] = displacement_violation.astype(int)
            features['range_violation'] |= features['displacement_range_flag']

        # Tilt range: ±5° relative to BASELINE (not absolute)
        # Use MEDIAN for baseline - robust to sensor malfunctions giving wrong values
        if 'tilt_group_avg' in features.columns:
            tilt_baseline = features['tilt_group_avg'].median()  # Median is robust to outliers
            tilt_deviation = (features['tilt_group_avg'] - tilt_baseline).abs()
            tilt_violation = tilt_deviation > 5  # More than 5° from baseline
            features['tilt_range_flag'] = tilt_violation.astype(int)
            features['range_violation'] |= features['tilt_range_flag']

        # Accelerometer p2p range (-1 to +1 g)
        if 'p2p_combined' in features.columns:
            p2p_violation = (features['p2p_combined'] < -1) | (features['p2p_combined'] > 1)
            features['p2p_range_flag'] = p2p_violation.astype(int)
            features['range_violation'] |= features['p2p_range_flag']

        # Accelerometer rms range (-1 to +1 g)
        if 'rms_combined' in features.columns:
            rms_violation = (features['rms_combined'] < -1) | (features['rms_combined'] > 1)
            features['rms_range_flag'] = rms_violation.astype(int)
            features['range_violation'] |= features['rms_range_flag']

        # Temperature range (-40°C to +125°C)
        if 'temperature' in features.columns:
            temp_violation = (features['temperature'] < -40) | (features['temperature'] > 125)
            features['temp_range_flag'] = temp_violation.astype(int)
            features['range_violation'] |= features['temp_range_flag']

        # Add range_violation to features_used so it's included in the analysis
        self.features_used.append('range_violation')

        return features

    def detect_sensor_stuck(self, features):
        """Detect stuck sensors"""
        for col in ['displacement_combined', 'tilt_group_avg', 'p2p_combined']:
            if col in features.columns:
                rolling_std = features[col].rolling(window=36, min_periods=10).std()
                features[f'{col}_stuck'] = (rolling_std < 0.001).astype(int)
                self.features_used.append(f'{col}_stuck')

        return features

    # =========================================================================
    # STEP 5: ENSEMBLE TRAINING
    # =========================================================================

    def train_ensemble(self, X_train):
        """Train ensemble of anomaly detectors"""
        self.models['isolation_forest'] = IsolationForest(
            n_estimators=150,
            contamination='auto',
            max_samples='auto',
            random_state=42
        )
        self.models['isolation_forest'].fit(X_train)

        self.models['lof'] = LocalOutlierFactor(
            n_neighbors=20,
            contamination='auto',
            novelty=True
        )
        self.models['lof'].fit(X_train)

        self.models['ocsvm'] = OneClassSVM(
            kernel='rbf',
            nu=0.05,
            gamma='scale'
        )
        self.models['ocsvm'].fit(X_train)

    def get_ensemble_scores(self, X):
        """Get anomaly scores from all models"""
        scores = {}
        scores['if_score'] = self.models['isolation_forest'].decision_function(X)
        scores['lof_score'] = self.models['lof'].decision_function(X)
        scores['svm_score'] = self.models['ocsvm'].decision_function(X)
        return scores

    def calculate_natural_threshold(self, scores):
        """Calculate natural threshold using 3σ method"""
        if_scores = scores['if_score']
        mean_score = np.mean(if_scores)
        std_score = np.std(if_scores)
        threshold = mean_score - 3 * std_score
        return threshold

    # =========================================================================
    # STEP 6: ANOMALY SCORING (0-10)
    # =========================================================================

    def calculate_anomaly_score(self, features_df, scores):
        """
        Calculate anomaly score (0-10) for each data point

        Score components:
        - Ensemble confidence (how many models agree)
        - Deviation magnitude (z-score severity)
        - Range violation severity
        - Seasonal deviation
        """
        n = len(features_df)
        anomaly_scores = np.zeros(n)

        # 1. Ensemble confidence (0-4 points)
        # More models agreeing = higher score
        if_flag = (scores['if_score'] < self.threshold_3sigma).astype(int)
        lof_flag = (scores['lof_score'] < np.percentile(scores['lof_score'], 3)).astype(int)
        svm_flag = (scores['svm_score'] < np.percentile(scores['svm_score'], 3)).astype(int)

        # Threshold flag
        threshold_flag = np.zeros(n)
        for col in features_df.columns:
            if col.endswith('_zscore_flag'):
                threshold_flag = np.maximum(threshold_flag, features_df[col].values)

        # Ensemble votes (0-4)
        ensemble_votes = if_flag + lof_flag + svm_flag + threshold_flag
        ensemble_score = (ensemble_votes / 4) * 4  # Scale to 0-4

        # 2. Deviation magnitude (0-3 points)
        # How severe is the z-score deviation
        deviation_score = np.zeros(n)
        zscore_cols = [c for c in features_df.columns if c.endswith('_zscore') and not c.endswith('_zscore_flag')]

        if zscore_cols:
            max_zscore = features_df[zscore_cols].abs().max(axis=1).fillna(0).values
            # Cap at 6 sigma (extremely rare), scale to 0-3
            deviation_score = np.minimum(max_zscore / 6, 1) * 3

        # 3. Baseline deviation score (0-2 points) - proportional to deviation even within range
        baseline_deviation_score = np.zeros(n)

        # Tilt baseline deviation: score proportional to how far from baseline
        # Use MEDIAN for baseline (robust to outliers)
        # 0° deviation = 0 pts, 2.5° = 1 pt, 5°+ = 2 pts (hits range limit)
        if 'tilt_group_avg' in features_df.columns:
            tilt_baseline = features_df['tilt_group_avg'].median()  # Median is robust
            tilt_deviation = (features_df['tilt_group_avg'] - tilt_baseline).abs().values
            # Score proportional to deviation, capped at 5° (range limit)
            baseline_deviation_score = np.maximum(baseline_deviation_score,
                                                   np.minimum(tilt_deviation / 5, 1) * 2)

        # Displacement baseline deviation (use MEDIAN)
        if 'displacement_combined' in features_df.columns:
            disp_baseline = features_df['displacement_combined'].median()  # Median is robust
            disp_deviation = (features_df['displacement_combined'] - disp_baseline).abs().values
            # Score proportional to deviation, capped at 50mm (range limit)
            baseline_deviation_score = np.maximum(baseline_deviation_score,
                                                   np.minimum(disp_deviation / 50, 1) * 2)

        # 4. Range violation (0-3 points) - additional penalty for exceeding limits
        range_score = np.zeros(n)

        # Tilt range violation: penalty for exceeding ±5° from baseline (use MEDIAN)
        if 'tilt_group_avg' in features_df.columns:
            tilt_baseline = features_df['tilt_group_avg'].median()
            tilt_deviation = (features_df['tilt_group_avg'] - tilt_baseline).abs().values
            # Beyond 5° = violation, score based on exceedance
            tilt_exceedance = np.maximum(tilt_deviation - 5, 0) / 5
            range_score = np.maximum(range_score, np.minimum(tilt_exceedance * 1.5, 3))

        # Displacement range violation (use MEDIAN)
        if 'displacement_combined' in features_df.columns:
            disp_baseline = features_df['displacement_combined'].median()
            disp_deviation = (features_df['displacement_combined'] - disp_baseline).abs().values
            disp_exceedance = np.maximum(disp_deviation - 50, 0) / 50
            range_score = np.maximum(range_score, np.minimum(disp_exceedance * 1.5, 3))

        # P2P vibration range: ±1g
        if 'p2p_combined' in features_df.columns:
            p2p_vals = features_df['p2p_combined'].abs().values
            p2p_exceedance = np.maximum(p2p_vals - 1, 0) / 0.5
            range_score = np.maximum(range_score, np.minimum(p2p_exceedance * 1.5, 3))

        # 5. Seasonal deviation (0-1 point)
        seasonal_score = np.zeros(n)
        if 'temp_seasonal_deviation' in features_df.columns:
            # High seasonal deviation = something unusual
            seasonal_dev = features_df['temp_seasonal_deviation'].abs().fillna(0).values
            seasonal_score = np.minimum(seasonal_dev / 4, 1) * 1

        # Combine scores:
        # - ensemble_score: 0-4 (how many ML methods agree)
        # - deviation_score: 0-3 (z-score magnitude)
        # - baseline_deviation_score: 0-2 (deviation from sensor baseline)
        # - range_score: 0-3 (exceeding range limits)
        # - seasonal_score: 0-1 (seasonal deviation)
        # Total possible: 13, but clipped to 10
        anomaly_scores = (ensemble_score + deviation_score + baseline_deviation_score +
                          range_score + seasonal_score)
        anomaly_scores = np.clip(anomaly_scores, 0, 10)

        return anomaly_scores

    # =========================================================================
    # STEP 7: FAULT CLASSIFICATION
    # =========================================================================

    def classify_fault(self, features_df, row_idx, anomaly_score):
        """
        Classify the type of fault:
        - 'sensor': Single sensor malfunction
        - 'structural': Physical structural issue
        - 'environmental': Weather/seasonal effect
        - 'unknown': Needs investigation

        Returns: (fault_type, confidence, reasoning)
        """
        row = features_df.iloc[row_idx]

        # Check indicators
        is_stuck = any(row.get(f'{col}_stuck', 0) == 1
                      for col in ['displacement_combined', 'tilt_group_avg', 'p2p_combined'])

        is_range_violation = row.get('range_violation', 0) == 1

        # Check seasonal deviation
        temp_seasonal_dev = abs(row.get('temp_seasonal_deviation', 0))
        disp_seasonal_dev = abs(row.get('displacement_seasonal_deviation', 0))
        has_seasonal_deviation = temp_seasonal_dev > 1.5 or disp_seasonal_dev > 1.5

        # Check z-score flags (which sensors are deviating)
        zscore_flags = {}
        for col in features_df.columns:
            if col.endswith('_zscore_flag'):
                zscore_flags[col] = row.get(col, 0)

        n_sensors_deviating = sum(zscore_flags.values())

        # Temperature correlation check
        temp_correlated = False
        if 'temp_displacement_correlation' in self.seasonal_baselines:
            if abs(self.seasonal_baselines['temp_displacement_correlation']) > 0.4:
                temp_correlated = True

        # Get ensemble votes
        ensemble_votes = row.get('ensemble_votes', 0)

        # Classification logic based on evidence

        # SENSOR FAULT: Stuck sensor
        if is_stuck:
            return ('sensor', 0.9, 'Sensor appears stuck (no variation)')

        # STRUCTURAL: Multiple sensors affected simultaneously (strongest indicator)
        if n_sensors_deviating >= 2:
            return ('structural', 0.8, f'Multiple sensors affected ({n_sensors_deviating} sensor types)')

        # ENVIRONMENTAL: Seasonal deviation (temperature-related)
        if has_seasonal_deviation:
            if temp_correlated:
                if disp_seasonal_dev > 1.5:
                    return ('environmental', 0.85, 'Temperature-driven displacement (thermal expansion)')
                else:
                    return ('environmental', 0.75, 'Temperature-correlated seasonal variation')
            else:
                # Seasonal deviation without temp correlation - still environmental
                return ('environmental', 0.6, 'Seasonal pattern detected')

        # Based on ensemble confidence
        if ensemble_votes >= 3:
            # High confidence anomaly - likely sensor issue
            if is_range_violation:
                return ('sensor', 0.8, 'High-confidence anomaly with range violation')
            else:
                return ('sensor', 0.7, 'High-confidence behavioral anomaly')

        # Moderate confidence anomaly (2 votes)
        if ensemble_votes == 2:
            if n_sensors_deviating == 1:
                return ('sensor', 0.6, 'Single sensor behavioral deviation')
            else:
                # 2 votes but no specific zscore flag - likely minor sensor drift
                return ('sensor', 0.5, 'Minor sensor behavioral anomaly')

        # Range violation without other indicators
        if is_range_violation:
            return ('sensor', 0.6, 'Sensor reading outside company limits')

        # Catch remaining cases - classify based on anomaly score
        if anomaly_score > 3:
            return ('sensor', 0.5, 'Elevated anomaly score - likely sensor issue')
        else:
            return ('sensor', 0.4, 'Low-level behavioral deviation')

    def classify_all_anomalies(self, features_df, anomaly_indices, anomaly_scores):
        """Classify all detected anomalies"""
        classifications = []

        for i, idx in enumerate(anomaly_indices):
            score = anomaly_scores[idx] if idx < len(anomaly_scores) else 0
            fault_type, confidence, reasoning = self.classify_fault(features_df, idx, score)
            classifications.append({
                'index': idx,
                'timestamp': features_df.index[idx],
                'fault_type': fault_type,
                'confidence': confidence,
                'reasoning': reasoning,
                'anomaly_score': score
            })

        return classifications

    # =========================================================================
    # MAIN TRAINING FUNCTION
    # =========================================================================

    def train(self, df):
        """Main training function"""
        print("\n" + "="*70)
        print("TRAINING UNIVERSAL MODEL v3.0")
        print("="*70)

        # Reset
        self.features_used = []
        self.sensor_patterns = {}

        # Step 1: Detect configuration
        self.detect_sensor_configuration(df)

        # Step 2: Learn seasonal baselines
        self.learn_seasonal_baselines(df)

        # Step 3: Detect patterns
        print("\nDetecting sensor patterns...")
        self.detect_di_pairing(df)
        self.detect_tilt_pattern(df)
        self.detect_accel_pairing(df)

        # Step 4: Generate features
        features = self.generate_features(df)

        # Clean data
        feature_cols = [c for c in self.features_used if c in features.columns]
        features_clean = features[feature_cols].dropna()

        print(f"\n  Clean samples: {len(features_clean):,}")

        if len(features_clean) < 100:
            raise ValueError("Not enough clean samples for training")

        # Scale features
        X = self.scaler.fit_transform(features_clean)

        # Step 5: Train ensemble
        print("\nTraining ensemble models...")
        self.train_ensemble(X)

        # Get scores
        scores = self.get_ensemble_scores(X)

        # Calculate natural threshold
        self.threshold_3sigma = self.calculate_natural_threshold(scores)
        print(f"\n  Natural threshold (3σ): {self.threshold_3sigma:.4f}")

        # Step 6: Calculate anomaly scores (0-10)
        print("\nCalculating anomaly scores (0-10)...")
        features_clean['anomaly_score'] = self.calculate_anomaly_score(features_clean, scores)

        # Add individual model results
        features_clean['if_score'] = scores['if_score']
        features_clean['lof_score'] = scores['lof_score']
        features_clean['svm_score'] = scores['svm_score']

        features_clean['if_flag'] = (scores['if_score'] < self.threshold_3sigma).astype(int)
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
        features_clean['final_anomaly'] = (features_clean['ensemble_votes'] >= self.vote_threshold).astype(int)

        # Range violations
        features_clean['range_anomaly'] = features_clean['range_violation'] if 'range_violation' in features_clean.columns else 0

        # Combined anomaly
        features_clean['combined_anomaly'] = ((features_clean['final_anomaly'] == 1) |
                                              (features_clean['range_anomaly'] == 1)).astype(int)

        # Step 7: Classify anomalies
        print("\nClassifying detected anomalies...")
        anomaly_indices = features_clean[features_clean['combined_anomaly'] == 1].index
        anomaly_iloc = [features_clean.index.get_loc(idx) for idx in anomaly_indices]

        # Pass anomaly scores for classification
        anomaly_scores_array = features_clean['anomaly_score'].values
        classifications = self.classify_all_anomalies(features_clean.reset_index(), anomaly_iloc, anomaly_scores_array)

        # Add classification to features
        features_clean['fault_type'] = 'normal'
        features_clean['fault_confidence'] = 0.0

        for c in classifications:
            features_clean.loc[c['timestamp'], 'fault_type'] = c['fault_type']
            features_clean.loc[c['timestamp'], 'fault_confidence'] = c['confidence']

        # Statistics
        n_if = features_clean['if_flag'].sum()
        n_lof = features_clean['lof_flag'].sum()
        n_svm = features_clean['svm_flag'].sum()
        n_threshold = features_clean['threshold_flag'].sum()
        n_final = features_clean['final_anomaly'].sum()
        n_range = features_clean['range_anomaly'].sum()
        n_combined = features_clean['combined_anomaly'].sum()

        print(f"\n  Results (Behavioral - Ensemble Voting):")
        print(f"    Isolation Forest: {n_if:,}")
        print(f"    LOF:              {n_lof:,}")
        print(f"    One-Class SVM:    {n_svm:,}")
        print(f"    Threshold (3σ):   {n_threshold:,}")
        print(f"    ─────────────────────────────────────")
        print(f"    FINAL (≥{self.vote_threshold} votes): {n_final:,}")

        print(f"\n  Results (Range Violations): {n_range:,}")
        print(f"  TOTAL ANOMALIES: {n_combined:,}")

        # Fault classification summary
        print(f"\n  Fault Classification:")
        fault_counts = features_clean[features_clean['combined_anomaly'] == 1]['fault_type'].value_counts()
        for fault_type, count in fault_counts.items():
            print(f"    {fault_type.title()}: {count:,}")

        # Score distribution
        print(f"\n  Anomaly Score Distribution:")
        print(f"    Min:    {features_clean['anomaly_score'].min():.1f}")
        print(f"    Max:    {features_clean['anomaly_score'].max():.1f}")
        print(f"    Mean:   {features_clean['anomaly_score'].mean():.1f}")
        print(f"    Median: {features_clean['anomaly_score'].median():.1f}")

        # Score buckets
        print(f"\n  Score Breakdown:")
        print(f"    0-2 (Normal):     {(features_clean['anomaly_score'] <= 2).sum():,}")
        print(f"    2.1-4 (Low):      {((features_clean['anomaly_score'] > 2) & (features_clean['anomaly_score'] <= 4)).sum():,}")
        print(f"    4.1-6 (Medium):   {((features_clean['anomaly_score'] > 4) & (features_clean['anomaly_score'] <= 6)).sum():,}")
        print(f"    6.1-8 (High):     {((features_clean['anomaly_score'] > 6) & (features_clean['anomaly_score'] <= 8)).sum():,}")
        print(f"    8.1-10 (Severe):  {(features_clean['anomaly_score'] > 8).sum():,}")

        return features_clean

    def get_summary(self):
        """Get model configuration summary"""
        return {
            'configuration': self.config,
            'sensor_patterns': self.sensor_patterns,
            'features_used': self.features_used,
            'n_features': len(self.features_used),
            'threshold_3sigma': self.threshold_3sigma,
            'vote_threshold': self.vote_threshold,
            'seasonal_baselines': list(self.seasonal_baselines.keys())
        }


def main(data_dir="."):
    """
    Run the Universal Model v3 on ALL 21 structures.
    This is a test/demo function - not used in production.

    Args:
        data_dir: Directory containing merged Excel files (STR122_merged.xlsx, etc.)
    """
    import os

    # All 21 structures
    test_structures = [
        'STR122', 'STR124', 'STR126', 'STR128', 'STR129', 'STR130', 'STR132',
        'STR171', 'STR172', 'STR173', 'STR175', 'STR176', 'STR177', 'STR178',
        'STR179', 'STR180', 'STR181', 'STR182', 'STR183', 'STR184', 'STR199'
    ]

    all_results = []

    for str_id in test_structures:
        print(f"\n{'='*80}")
        print(f"TESTING: {str_id}")
        print(f"{'='*80}")

        try:
            file_path = f"{data_dir}/{str_id}_merged.xlsx"
            df = pd.read_excel(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Use all data from Nov 2023 onwards (exclude calibration period)
            df = df[df['timestamp'] >= '2023-11-01']

            # Create and train model (pass structure_id for axis mapping lookup)
            model = UniversalModelV3(structure_id=str_id, ensemble_vote_threshold=2)
            results_df = model.train(df)

            # Get summary
            summary = model.get_summary()

            # Get high-severity anomalies
            severe = results_df[results_df['anomaly_score'] > 6]

            all_results.append({
                'structure': str_id,
                'config': summary['configuration'],
                'tilt_pattern': summary['sensor_patterns'].get('tilt', {}).get('type', 'N/A'),
                'samples': len(results_df),
                'total_anomalies': results_df['combined_anomaly'].sum(),
                'severe_anomalies': len(severe),
                'avg_score': results_df['anomaly_score'].mean(),
                'sensor_faults': (results_df['fault_type'] == 'sensor').sum(),
                'structural_faults': (results_df['fault_type'] == 'structural').sum(),
                'environmental': (results_df['fault_type'] == 'environmental').sum(),
            })

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print(f"\n{'='*80}")
    print("UNIVERSAL MODEL v3.0 RESULTS SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Structure':<10} {'Config':<8} {'Tilt':<8} {'Samples':<12} {'Anomalies':<12} {'Severe':<10} {'Score':<8} {'Sensor':<10} {'Struct':<10} {'Environ':<10} {'Unknown'}")
    print("-"*120)

    total_samples = 0
    total_anomalies = 0
    total_severe = 0
    total_sensor = 0
    total_structural = 0
    total_environmental = 0
    total_unknown = 0

    for r in all_results:
        unknown = r['total_anomalies'] - r['sensor_faults'] - r['structural_faults'] - r['environmental']
        print(f"{r['structure']:<10} {r['config']:<8} {r['tilt_pattern']:<8} {r['samples']:<12,} {int(r['total_anomalies']):<12,} {r['severe_anomalies']:<10} {r['avg_score']:<8.1f} {r['sensor_faults']:<10} {r['structural_faults']:<10} {r['environmental']:<10} {unknown}")

        total_samples += r['samples']
        total_anomalies += r['total_anomalies']
        total_severe += r['severe_anomalies']
        total_sensor += r['sensor_faults']
        total_structural += r['structural_faults']
        total_environmental += r['environmental']
        total_unknown += unknown

    print("-"*120)
    print(f"{'TOTAL':<10} {'':<8} {'':<8} {total_samples:<12,} {int(total_anomalies):<12,} {total_severe:<10} {'':<8} {total_sensor:<10} {total_structural:<10} {total_environmental:<10} {total_unknown}")

    # Fault classification breakdown
    print(f"\n{'='*80}")
    print("FAULT CLASSIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"\n  Total Anomalies: {int(total_anomalies):,} ({total_anomalies/total_samples*100:.2f}%)")
    print(f"\n  Fault Type Breakdown:")
    print(f"    Sensor Faults:      {total_sensor:,} ({total_sensor/total_anomalies*100:.1f}% of anomalies)")
    print(f"    Structural Faults:  {total_structural:,} ({total_structural/total_anomalies*100:.1f}% of anomalies)")
    print(f"    Environmental:      {total_environmental:,} ({total_environmental/total_anomalies*100:.1f}% of anomalies)")
    print(f"    Unknown:            {total_unknown:,} ({total_unknown/total_anomalies*100:.1f}% of anomalies)")

    print(f"\n  Severity Breakdown:")
    print(f"    Severe (score >6): {total_severe:,}")

    print(f"\n{'='*80}")
    print("MODEL FEATURES")
    print(f"{'='*80}")
    print(f"  1. Tilt pairing: 2+2 pattern (ground truth axis mapping)")
    print(f"  2. Displacement pairing: Temporal stability detection")
    print(f"  3. Seasonal baseline learning from data")
    print(f"  4. Anomaly scoring (0-10)")
    print(f"  5. Fault classification (sensor/structural/environmental)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python universal_model_v3.py <DATA_DIR>")
        print("  DATA_DIR: Directory containing STR*_merged.xlsx files")
        sys.exit(1)
