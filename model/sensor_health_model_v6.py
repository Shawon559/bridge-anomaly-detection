"""
Sensor Health Monitoring Model V6

Individual sensor health monitoring with gradual scoring:
- Tilt/Displacement: Gradual scoring based on deviation from stable baseline
- Temperature: Monthly baseline (seasonal patterns)
- Stable value: Uses MEDIAN (robust to outliers)
- Stuck detection: Detects frozen sensors (accelerometer/temperature)
- Erratic detection: Flags excessive random variation
- MAD (Median Absolute Deviation): Robust statistics resistant to outliers
- Tilt axis awareness: Tracks X vs Y axis for better diagnosis
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


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


class SensorHealthMonitorV6:
    """Individual sensor health monitoring with gradual scoring."""

    def __init__(self, structure_id=None):
        """
        Args:
            structure_id: Structure identifier (e.g., 'STR122')
        """
        self.structure_id = structure_id

        self.sensor_baselines = {}
        self.temp_monthly_baselines = {}
        self.sensor_types = {}
        self.tilt_axes = {}
        self._global_noise_levels = {}
        self.is_trained = False

    def _determine_sensor_type(self, sensor_id):
        """Determine sensor type from sensor ID"""
        if not isinstance(sensor_id, str):
            return 'unknown'

        sensor_id = sensor_id.upper()

        if sensor_id.startswith('DI'):
            return 'displacement'
        elif sensor_id.startswith('TI'):
            return 'tilt'
        elif sensor_id.startswith('AC'):
            return 'accelerometer'
        elif sensor_id.startswith('TE') or sensor_id.startswith('TP'):
            return 'temperature'
        else:
            return 'unknown'

    def train(self, df):
        """Learn stable value for each sensor from historical data."""
        print(f"Training sensor health monitor v6 for {self.structure_id}...")

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['month'] = df['timestamp'].dt.month

        sensors = df['sensor_id'].unique()

        for sensor_id in sensors:
            sensor_data = df[df['sensor_id'] == sensor_id].copy()
            sensor_type = self._determine_sensor_type(sensor_id)
            self.sensor_types[sensor_id] = sensor_type

            if sensor_type == 'tilt' and self.structure_id in TILT_AXIS_MAPPING:
                mapping = TILT_AXIS_MAPPING[self.structure_id]
                if sensor_id in mapping['x_axis']:
                    self.tilt_axes[sensor_id] = 'x'
                elif sensor_id in mapping['y_axis']:
                    self.tilt_axes[sensor_id] = 'y'

            value_cols = [col for col in sensor_data.columns
                         if col not in ['timestamp', 'sensor_id', 'sensor_type', 'month']]

            if sensor_type == 'temperature':
                sensor_monthly = {}

                for col in value_cols:
                    if pd.api.types.is_numeric_dtype(sensor_data[col]):
                        monthly_baselines = {}

                        for month in range(1, 13):
                            month_data = sensor_data[sensor_data['month'] == month][col].dropna()

                            if len(month_data) > 30:
                                monthly_baselines[month] = {
                                    'mean': month_data.mean(),
                                    'std': month_data.std(),
                                    'median': month_data.median(),
                                    'count': len(month_data)
                                }

                        if len(monthly_baselines) >= 6:
                            sensor_monthly[col] = monthly_baselines

                if sensor_monthly:
                    self.temp_monthly_baselines[sensor_id] = sensor_monthly

            else:
                sensor_baseline = {}

                for col in value_cols:
                    if pd.api.types.is_numeric_dtype(sensor_data[col]):
                        values = sensor_data[col].dropna()

                        if len(values) > 100:
                            stable_value = values.median()

                            std = values.std()
                            mad = np.median(np.abs(values - stable_value))
                            robust_std = mad * 1.4826 if mad > 0 else std

                            sensor_baseline[col] = {
                                'stable_value': stable_value,
                                'std': std,
                                'mad': mad,
                                'robust_std': robust_std,
                                'count': len(values)
                            }

                if sensor_baseline:
                    self.sensor_baselines[sensor_id] = sensor_baseline

        self._calculate_global_noise_levels(df)

        self.is_trained = True
        print(f"Trained on {len(sensors)} sensors:")
        print(f"  - Temperature (monthly baseline): {len(self.temp_monthly_baselines)}")
        print(f"  - Tilt/Displacement/Accel (stable value): {len(self.sensor_baselines)}")

        if self.tilt_axes:
            x_count = sum(1 for axis in self.tilt_axes.values() if axis == 'x')
            y_count = sum(1 for axis in self.tilt_axes.values() if axis == 'y')
            print(f"  - Tilt axes mapped: {x_count} X-axis, {y_count} Y-axis")

        return self

    def _calculate_global_noise_levels(self, df):
        """Calculate typical noise level for each sensor type."""
        for sensor_type in ['tilt', 'displacement', 'accelerometer']:
            stds = []

            for sensor_id, baseline in self.sensor_baselines.items():
                if self.sensor_types.get(sensor_id) == sensor_type:
                    if 'value' in baseline and 'robust_std' in baseline['value']:
                        stds.append(baseline['value']['robust_std'])

            if stds:
                self._global_noise_levels[sensor_type] = np.median(stds)
            else:
                self._global_noise_levels[sensor_type] = 1.0

    def _detect_erratic_behavior(self, values, sensor_id):
        """Check if sensor is showing erratic behavior."""
        sensor_type = self.sensor_types.get(sensor_id)

        if sensor_type not in ['tilt', 'displacement']:
            return False, 0.0, 0.0

        if len(values) < 50:
            return False, 0.0, 0.0

        global_noise = self._global_noise_levels.get(sensor_type, 1.0)
        current_std = values.std()
        noise_ratio = current_std / (global_noise + 1e-6)

        if noise_ratio > 5.0:
            erratic_score = min(9.0, 5.0 + (noise_ratio - 5.0) * 0.5)
            return True, noise_ratio, erratic_score

        return False, noise_ratio, 0.0

    def score_sensor_daily(self, df):
        """Score individual sensor health for daily data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring. Call train() first.")

        if len(df) == 0:
            return pd.DataFrame(columns=['sensor_id', 'sensor_type', 'health_score',
                                        'diagnosis', 'tilt_axis', 'readings_count',
                                        'data_coverage_%', 'data_gaps'])

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        results = []
        sensors = df['sensor_id'].unique()

        for sensor_id in sensors:
            sensor_data = df[df['sensor_id'] == sensor_id].copy()
            sensor_type = self.sensor_types.get(sensor_id, 'unknown')

            # Detect data gaps
            sensor_data = sensor_data.sort_values('timestamp')
            time_gaps = []
            data_coverage = 100.0

            if len(sensor_data) > 1:
                timestamps = sensor_data['timestamp'].values
                time_diffs = pd.Series(timestamps[1:]) - pd.Series(timestamps[:-1])
                time_diffs_minutes = time_diffs.dt.total_seconds() / 60

                expected_interval = time_diffs_minutes.median()
                large_gaps = time_diffs_minutes[time_diffs_minutes > expected_interval * 3]

                if len(large_gaps) > 0:
                    for idx in large_gaps.index:
                        gap_start = timestamps[idx]
                        gap_end = timestamps[idx + 1]
                        gap_duration_mins = time_diffs_minutes.iloc[idx]
                        time_gaps.append({
                            'start': pd.Timestamp(gap_start),
                            'end': pd.Timestamp(gap_end),
                            'duration_minutes': gap_duration_mins
                        })

                    total_day_minutes = 24 * 60
                    missing_minutes = sum(gap['duration_minutes'] for gap in time_gaps)
                    data_coverage = max(0, 100 - (missing_minutes / total_day_minutes * 100))

            value_cols = [col for col in sensor_data.columns
                         if col not in ['timestamp', 'sensor_id', 'sensor_type']]

            max_score = 0.0
            issues = []

            # Score data gap severity
            if data_coverage < 50:
                max_score = max(max_score, 9.0)
                issues.append('DATA_GAP(>50%)')
            elif data_coverage < 75:
                max_score = max(max_score, 6.0)
                issues.append('DATA_GAP(>25%)')
            elif data_coverage < 90:
                max_score = max(max_score, 3.0)
                issues.append('DATA_GAP(>10%)')

            for col in value_cols:
                if pd.api.types.is_numeric_dtype(sensor_data[col]):
                    values = sensor_data[col].dropna()

                    if len(values) == 0:
                        continue

                    latest_value = values.iloc[-1]

                    if sensor_type == 'temperature' and sensor_id in self.temp_monthly_baselines:
                        if col in self.temp_monthly_baselines[sensor_id]:
                            current_month = sensor_data['timestamp'].iloc[-1].month
                            monthly_baseline = self.temp_monthly_baselines[sensor_id][col].get(current_month, {})

                            if monthly_baseline:
                                baseline_mean = monthly_baseline['mean']
                                baseline_std = monthly_baseline['std']

                                if baseline_std > 0.001:
                                    deviation = abs(latest_value - baseline_mean) / baseline_std

                                    if deviation > 5:
                                        if deviation < 6:
                                            score = 3.0
                                        elif deviation < 8:
                                            score = 5.0
                                        elif deviation < 10:
                                            score = 7.0
                                        else:
                                            score = 9.0

                                        max_score = max(max_score, score)
                                        issues.append('DRIFT')

                    elif sensor_type in ['tilt', 'displacement'] and sensor_id in self.sensor_baselines:
                        if col == 'value' and col in self.sensor_baselines[sensor_id]:
                            baseline = self.sensor_baselines[sensor_id][col]
                            stable_value = baseline['stable_value']
                            deviation = abs(latest_value - stable_value)

                            if deviation <= 2:
                                score = 0.0
                            elif deviation <= 3:
                                score = 2.0 + (deviation - 2.0)
                            elif deviation <= 5:
                                score = 3.0 + (deviation - 3.0)
                            elif deviation <= 10:
                                score = 5.0 + (deviation - 5.0) * 0.4
                            else:
                                score = min(9.0, 7.0 + (deviation - 10.0) * 0.1)

                            if score > 0:
                                max_score = max(max_score, score)
                                if score >= 2:
                                    issues.append('DRIFT')

                            is_erratic, noise_ratio, erratic_score = self._detect_erratic_behavior(values, sensor_id)
                            if is_erratic:
                                max_score = max(max_score, erratic_score)
                                issues.append(f'ERRATIC({noise_ratio:.1f}x)')

                    elif sensor_type == 'accelerometer' and sensor_id in self.sensor_baselines:
                        if col in ['peak_to_peak', 'rms'] and col in self.sensor_baselines[sensor_id]:
                            baseline = self.sensor_baselines[sensor_id][col]
                            stable_value = baseline['stable_value']
                            deviation = abs(latest_value - stable_value)

                            if deviation <= 0.01:
                                score = 0.0
                            elif deviation <= 0.05:
                                score = 2.0 + (deviation - 0.01) / 0.04 * 1.0
                            elif deviation <= 0.1:
                                score = 3.0 + (deviation - 0.05) / 0.05 * 2.0
                            elif deviation <= 0.5:
                                score = 5.0 + (deviation - 0.1) / 0.4 * 2.0
                            else:
                                score = min(9.0, 7.0 + (deviation - 0.5) * 0.2)

                            if score > 0:
                                max_score = max(max_score, score)
                                if score >= 2:
                                    issues.append('DRIFT')

                    if sensor_type in ['accelerometer', 'temperature']:
                        primary_cols = ['value', 'peak_to_peak', 'rms']
                        if col in primary_cols and len(values) > 50:
                            std = values.std()
                            value_range = values.max() - values.min()

                            if std < 0.001 and value_range < 0.001:
                                max_score = max(max_score, 8.0)
                                issues.append('STUCK')

            if max_score < 2:
                diagnosis = 'HEALTHY'
            elif max_score < 5:
                diagnosis = ', '.join(issues) if issues else 'DEGRADED'
            else:
                diagnosis = ', '.join(issues) if issues else 'CRITICAL'

            if sensor_type == 'temperature' and sensor_id not in self.temp_monthly_baselines:
                diagnosis = 'NO_BASELINE'
            elif sensor_type in ['tilt', 'displacement', 'accelerometer'] and sensor_id not in self.sensor_baselines:
                diagnosis = 'NO_BASELINE'

            tilt_axis = self.tilt_axes.get(sensor_id, None)

            # Format gap information
            gap_info = ""
            if len(time_gaps) > 0:
                gap_details = []
                for gap in time_gaps[:3]:  # Show up to 3 largest gaps
                    start_str = gap['start'].strftime('%H:%M')
                    end_str = gap['end'].strftime('%H:%M')
                    duration = int(gap['duration_minutes'])
                    gap_details.append(f"{start_str}-{end_str}({duration}min)")
                gap_info = "; ".join(gap_details)
                if len(time_gaps) > 3:
                    gap_info += f" +{len(time_gaps)-3} more"

            results.append({
                'sensor_id': sensor_id,
                'sensor_type': sensor_type,
                'tilt_axis': tilt_axis,
                'health_score': round(max_score, 1),
                'diagnosis': diagnosis,
                'readings_count': len(sensor_data),
                'data_coverage_%': round(data_coverage, 1),
                'data_gaps': gap_info if gap_info else 'None'
            })

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            critical = len(results_df[results_df['health_score'] > 7])
            warning = len(results_df[results_df['health_score'] > 4])

            print(f"\nSensor Health Summary:")
            print(f"  Total sensors: {len(results_df)}")
            print(f"  Critical (>7): {critical}")
            print(f"  Warning (>4):  {warning}")

            tilt_results = results_df[results_df['sensor_type'] == 'tilt']
            if len(tilt_results) > 0 and tilt_results['tilt_axis'].notna().any():
                x_axis = tilt_results[tilt_results['tilt_axis'] == 'x']
                y_axis = tilt_results[tilt_results['tilt_axis'] == 'y']

                if len(x_axis) > 0:
                    x_issues = len(x_axis[x_axis['health_score'] > 4])
                    print(f"  X-axis tilts: {x_issues}/{len(x_axis)} with issues")

                if len(y_axis) > 0:
                    y_issues = len(y_axis[y_axis['health_score'] > 4])
                    print(f"  Y-axis tilts: {y_issues}/{len(y_axis)} with issues")

        return results_df

    def get_overall_status(self, results_df):
        """Get overall sensor health status."""
        if len(results_df) == 0:
            return {
                'status': 'NO_DATA',
                'score': 0,
                'message': 'No sensor data available'
            }

        max_score = results_df['health_score'].max()
        critical_count = len(results_df[results_df['health_score'] > 7])
        warning_count = len(results_df[results_df['health_score'] > 4])

        if critical_count > 0:
            status = 'CRITICAL'
            message = f"{critical_count} sensor(s) require immediate attention"
        elif warning_count > 0:
            status = 'WARNING'
            message = f"{warning_count} sensor(s) showing degraded performance"
        else:
            status = 'HEALTHY'
            message = "All sensors functioning normally"

        return {
            'status': status,
            'score': round(max_score, 1),
            'critical_sensors': critical_count,
            'warning_sensors': warning_count,
            'total_sensors': len(results_df),
            'message': message
        }
