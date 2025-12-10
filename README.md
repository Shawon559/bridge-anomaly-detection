# Bridge Health Monitoring System

**Version**: 3.0 (Bridge) + 6.0 (Sensor)
**Status**: Production Ready

---

## Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib openpyxl

# Run daily health check
python run_daily_health_check.py STR122 yesterdays_data.xlsx
```

**Output**:
- `daily_reports/STR122_2024-02-15.png` - Visualization
- `daily_reports/STR122_2024-02-15_results.csv` - Detailed results

---

## Package Contents

```
├── run_daily_health_check.py      # Main daily script
├── run_10_random_tests.py         # Testing/validation
├── retrain_all_models.py          # Model retraining
│
├── model/
│   ├── universal_model_v3.py      # Bridge health model
│   └── sensor_health_model_v6.py  # Sensor health model
│
├── preprocessing/
│   └── merge_with_all_sensors.py  # Data preprocessing
│
└── trained_models/
    ├── bridge_health/             # 21 structure models
    └── sensor_health/             # 21 structure models
```

---

## Supported Structures (21)

STR122, STR124, STR126, STR128, STR129, STR130, STR132, STR171, STR172, STR173, STR175, STR176, STR177, STR178, STR179, STR180, STR181, STR182, STR183, STR184, STR199

---

## Features

### Bridge Health Detection
- Ensemble ML (Isolation Forest, LOF, One-Class SVM)
- Seasonal baseline learning
- Anomaly scoring (0-10 scale)
- Severity classification (Normal/Warning/Critical)

### Sensor Health Monitoring
- Gradual scoring based on deviation from baseline
- Detects drift, stuck sensors, erratic behavior
- Data gap tracking with timestamps
- Tilt axis mapping (X/Y)

---

## Output Format

### CSV Report
```csv
sensor_id,sensor_type,health_score,diagnosis,data_coverage_%,data_gaps
DI531,displacement,6.0,"DATA_GAP(>25%), DRIFT",83.0,"05:50-09:00(190min)"
TI535,tilt,0.0,HEALTHY,100.0,None
AC383,accelerometer,0.0,HEALTHY,100.0,None
```

**Columns**:
- `sensor_id` - Sensor identifier
- `sensor_type` - tilt/displacement/accelerometer/temperature
- `health_score` - 0-10 (0=healthy, 10=critical)
- `diagnosis` - Issue description (HEALTHY, DRIFT, STUCK, ERRATIC, DATA_GAP)
- `data_coverage_%` - Percentage of day with data
- `data_gaps` - Time ranges of missing data (e.g., "05:50-09:00(190min)")

### PNG Visualization
- Bridge health chart (hourly anomaly scores)
- Sensor summary (healthy/warning/critical counts)
- Individual sensor scores
- Issues section with detailed diagnostics

---

## Score Interpretation

### Bridge Health
| Score | Status | Action |
|-------|--------|--------|
| 0-4 | Normal | Routine monitoring |
| 4-7 | Warning | Investigate trends |
| 7-10 | Critical | Immediate attention |

### Sensor Health
| Score | Status | Typical Cause |
|-------|--------|---------------|
| 0-2 | Healthy | Normal operation |
| 2-4 | Minor | Gradual drift |
| 4-7 | Degraded | Significant drift or data gaps |
| 7-10 | Critical | Stuck, erratic, or major outage |

### Data Gap Severity
| Coverage | Score | Diagnosis |
|----------|-------|-----------|
| > 90% | 0.0 | No penalty |
| 75-90% | 3.0 | DATA_GAP(>10%) |
| 50-75% | 6.0 | DATA_GAP(>25%) |
| < 50% | 9.0 | DATA_GAP(>50%) |

---

## Usage Examples

### Daily Production Use
```bash
python run_daily_health_check.py STR128 yesterdays_data.xlsx
```

### Validation Testing
```bash
python run_10_random_tests.py "../STR Data - Merged"
```

### Model Retraining (if needed)
```bash
python retrain_all_models.py "../STR Data - Merged"
```

---

## Data Requirements

**Input Excel file** must have:
- `timestamp` - DateTime
- `sensor_id` - Sensor identifier (e.g., TI535, DI531)
- `sensor_type` - tilt, displacement, accelerometer, temperature_probe
- `value` - Primary reading
- Optional: `p2p`, `rms` (for accelerometers)

**Sensor naming**:
- `DIxxx` - Displacement
- `TIxxx` - Tilt
- `ACxxx` - Accelerometer
- `TPxxx` - Temperature

---

## System Behavior

### Normal Data
- Generates PNG visualization and CSV report
- Scores all sensors
- Flags issues (drift, stuck, erratic, gaps)

### No Data / Empty File
- Does NOT crash
- Generates report with "NO DATA AVAILABLE" message
- Creates empty but valid CSV file

### Partial Data (some sensors missing)
- Scores available sensors only
- Flags missing sensors in report
- Shows data gap information with timestamps

---

## Technical Details

### Bridge Model
- Features: 20-56 per structure (displacement, tilt, accelerometer, temperature)
- Ensemble weights: IF 30%, LOF 30%, SVM 40%
- Seasonal baselines: Monthly patterns
- Robust statistics: MAD-based

### Sensor Model
- Baseline: Median stable value
- Temperature: Monthly seasonal adjustment
- Gap detection: Adaptive interval-based
- Stuck detection: Variance threshold < 0.001
- Erratic detection: Noise ratio > 5x normal

---

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, openpyxl

---

## Testing

All 21 models tested: **117/117 tests passed (100%)**

Features verified:
- ✓ Bridge anomaly detection
- ✓ Sensor health monitoring
- ✓ Data gap tracking
- ✓ Empty data handling
- ✓ Tilt axis mapping
- ✓ Visualization generation
- ✓ Edge case handling
