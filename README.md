# Bridge Sensor Anomaly Detection System

Ensemble-based machine learning system for structural health monitoring of bridge infrastructure.

**Project:** Bridge Structural Health Monitoring  
**Developed at:** AI Lab, Woosong University  
**Client:** MORICON Company

---

## Overview

This repository contains a production-ready anomaly detection algorithm for bridge sensor monitoring. The system uses ensemble machine learning to detect and classify anomalies from displacement, tilt, accelerometer, and temperature sensors.

## Key Features

- **Ensemble Voting System**: Combines 4 detection algorithms (Isolation Forest, LOF, One-Class SVM, 3σ threshold)
- **Intelligent Sensor Pairing**: Ground truth axis mapping for tilt sensors, temporal stability detection for displacement sensors
- **0-10 Severity Scoring**: Easy-to-interpret anomaly scores with 5 severity levels
- **Fault Classification**: Automatic classification into sensor faults, structural issues, or environmental effects
- **Pre-trained Models**: 21 structures (STR122-STR199) ready for deployment
- **Daily Batch Processing**: Designed for automated daily monitoring operations

## Repository Structure

```
├── model/
│   ├── universal_model_v3.py        # Core ensemble anomaly detection algorithm
│   └── model_service.py             # Scoring service wrapper
├── preprocessing/
│   └── merge_with_all_sensors.py    # Data preprocessing pipeline
├── pretrained_models/
│   └── *.pkl                        # Pre-trained models for 21 structures
├── run_daily_scoring.py             # Daily scoring script
└── README.md
```

## Quick Start

### Installation

```bash
pip install pandas numpy scikit-learn openpyxl
```

### Usage

Run daily anomaly scoring:

```bash
python run_daily_scoring.py <STRUCTURE_ID> <DATA_FILE>
```

Example:
```bash
python run_daily_scoring.py STR122 data/STR122_merged.xlsx
```

### Output

- Console summary with severity breakdown
- CSV file with detailed anomaly scores
- Exit codes: 0 (Normal), 1 (High severity), 2 (Severe)

## Algorithm Details

### Ensemble Approach

The system combines four detection methods:
1. **Isolation Forest** - Tree-based outlier detection
2. **Local Outlier Factor (LOF)** - Density-based detection
3. **One-Class SVM** - Boundary-based detection
4. **3σ Threshold** - Statistical deviation

An anomaly is flagged when ≥2 methods agree (voting threshold).

### Severity Scoring (0-10)

Score components:
- Ensemble confidence (0-4 pts): Number of methods in agreement
- Deviation magnitude (0-3 pts): Z-score severity
- Baseline deviation (0-2 pts): Deviation from normal baseline
- Range violation (0-3 pts): Exceeding sensor operational limits
- Seasonal deviation (0-1 pt): Unusual for current season

### Fault Classification

- **Sensor Fault**: Single sensor malfunction
- **Structural Issue**: Multiple correlated sensors affected
- **Environmental Effect**: Temperature/seasonal influences

## Supported Structures

21 bridge structures with pre-trained models:
- STR122, STR124, STR126, STR128, STR129, STR130, STR132
- STR171, STR172, STR173, STR175, STR176, STR177
- STR178, STR179, STR180, STR181, STR182, STR183, STR184
- STR199

## Data Format

Input data must contain:
- `timestamp` - DateTime
- `sensor_id` - Sensor identifier
- `sensor_type` - displacement, tilt, accelerometer, temperature_probe
- `value` - Sensor reading

Use the preprocessing script to convert raw multi-sheet Excel files to the required format.

## Technical Specifications

- **Python**: 3.8+
- **Dependencies**: pandas, numpy, scikit-learn, openpyxl
- **Sensor Types**: 4 (Displacement, Tilt, Accelerometer, Temperature)
- **Configurations**: 4 sensor configurations (A, B, C, D)
- **Training Data**: November 2023 - September 2025

## License

This project was developed for MORICON Company under Woosong University AI Lab.

---

*Version: 3.0 | January 2026*
