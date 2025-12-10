# Bridge & Sensor Health Monitoring System - Usage Guide

## System Overview

This system monitors bridge structural health and individual sensor health using enhanced ML models with:
- **Ensemble weighted voting** (instead of simple majority)
- **MAD (Median Absolute Deviation)** for robust statistics
- **Seasonal baselines** for temperature/displacement
- **Tilt axis tracking** (X-axis vs Y-axis)
- **Erratic sensor detection** (5x noise threshold)

## Daily Operations Workflow

### Step 1: Get Yesterday's Data
Company receives yesterday's sensor data as Excel file (e.g., `2024-02-15_data.xlsx`)

### Step 2: Run Daily Scoring
```bash
python run_daily_health_check.py <STRUCTURE_ID> <DATA_FILE>
```

Example:
```bash
python run_daily_health_check.py STR128 yesterdays_data.xlsx
```

### Step 3: Check Results
System generates:
- PNG visualization: `daily_reports/STR128_2024-02-15.png`
- CSV results: `daily_reports/STR128_2024-02-15_results.csv`

## Directory Structure

```
Delivery Package/
├── model/                          # Model code
│   ├── universal_model_v3.py       # Bridge health model
│   └── sensor_health_model_v6.py   # Sensor health model
│
├── trained_models/                 # Pre-trained models (19 structures)
│   ├── bridge_health/
│   └── sensor_health/
│
├── run_daily_health_check.py      # MAIN SCRIPT - Run this daily
├── run_10_random_tests.py         # Testing script (10 random tests)
└── USAGE_GUIDE.md                 # This file
```

## What Each Script Does

1. **run_daily_health_check.py** - Production daily script
   - Scores ONE day of data
   - Generates PNG + CSV reports
   - Uses pre-trained models from `trained_models/`

2. **run_10_random_tests.py** - Testing/validation script
   - Tests 10 random structure+date combinations
   - Generates PNG visualizations
   - Useful for system validation

## Model Training (Only if needed)

If you need to retrain models with new historical data:

```bash
# This was already done - models are in trained_models/
# Only run if you get new historical data

python train_all_models.py
```

## Understanding the Results

### Bridge Health Score (0-10)
- **0-4**: NORMAL - No structural concerns
- **4-7**: WARNING - Monitor closely
- **7-10**: CRITICAL - Immediate inspection needed

### Sensor Health Score (0-10)
- **0-4**: HEALTHY - Sensor operating normally
- **4-7**: WARNING - Sensor showing drift/issues
- **7-10**: CRITICAL - Sensor malfunction (stuck/erratic/failed)

### Common Sensor Diagnoses
- **STUCK**: Sensor not changing values (frozen)
- **DRIFT**: Sensor drifted from learned baseline
- **ERRATIC**: Excessive random noise (5x normal)
- **FROZEN**: No variation detected
- **OUTLIER**: Values outside expected range

## Enhancements from Mufty's Code

✓ Ensemble weighted voting (35% IF, 30% LOF, 20% SVM, 15% Threshold)
✓ MAD for robust seasonal baselines
✓ MAD for displacement sensor pairing
✓ Tilt axis awareness (X vs Y detection)
✓ Erratic/noisy sensor detection

## Support

For issues or questions, check the PNG visualizations first - they show:
1. Bridge health score timeline (24 hours)
2. Sensor health summary (critical/warning/healthy counts)
3. Individual sensor scores
4. Detected issues with diagnoses
