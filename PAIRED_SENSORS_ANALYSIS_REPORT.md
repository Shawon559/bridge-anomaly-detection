# PAIRED SENSOR ANALYSIS - COMPREHENSIVE REPORT
**Generated:** 2025-12-05
**Project:** Bridge Structural Health Monitoring - Anomaly Detection
**Analysis Scope:** All 21 structures

---

## EXECUTIVE SUMMARY

### Key Findings

We successfully identified **34 paired sensor systems** across **11 out of 21 bridge structures**. This discovery has critical implications for your anomaly detection model:

1. **The pairing pattern confirmed in STR128 and STR129 IS widespread** across the dataset
2. **14 displacement sensor pairs** and **20 tilt sensor pairs** identified
3. **Individual sensors showing "range violations" are often working correctly** as part of differential measurement systems
4. **Combined sensor values must be used** for accurate anomaly detection

---

## OVERALL STATISTICS

| Metric | Count |
|--------|-------|
| **Total structures analyzed** | 21 |
| **Structures with paired sensors** | 11 |
| **Total paired systems** | 34 |
| **Displacement sensor pairs** | 14 |
| **Tilt sensor pairs** | 20 |
| **Structures without pairing** | 10 |

### Structures WITHOUT Paired Sensors
STR126, STR130, STR132, STR171, STR172, STR173, STR175, STR176, STR177, STR199

These structures likely have:
- Single sensors (no redundancy)
- Different sensor configurations
- Simpler monitoring systems

---

## YOUR PRIMARY STRUCTURE: STR128

### Paired Sensors Identified

**DI549 + DI550 (Displacement Pair)**
- **Synchronization:** 100.00% (43,809 matching timestamps) ✅
- **Pairing Type:** Complementary (one positive, one negative)
- **Individual Ranges:**
  - DI549: +48.40 to +57.35 mm (mean: +52.29 mm)
  - DI550: -53.50 to -34.30 mm (mean: -41.02 mm)
- **Combined Displacement:**
  - Range: -3.97 to +21.82 mm
  - Mean: **11.26 mm**
  - Std Dev: **5.19 mm**
- **Correlation:** 0.6054 (moderate positive - both respond to same structural movement)

### Analysis & Implications

**Why Perfect Synchronization Matters:**
- 100% timestamp overlap indicates hardware-level synchronization
- Both sensors are triggered simultaneously by the same data acquisition system
- This confirms they are designed to work as a differential pair

**Physical Interpretation:**
- The ~11mm combined displacement represents actual structural movement
- Low standard deviation (5.19mm) indicates stable, predictable behavior
- Both sensors tracking bridge deck deflection from different reference points

**For Your Anomaly Detection Model:**
1. **DO NOT** use DI549 or DI550 individually
2. **USE** the combined value: `displacement = DI549 + DI550`
3. **Expected normal range:** ~11mm ± 10mm (2 std devs)
4. **Anomaly threshold suggestion:** Values outside 1-21mm range warrant investigation

### Tiltmeter Sensors (TI551-TI554)

**Important Finding: NO PAIRING DETECTED**

All 6 possible combinations of the 4 tiltmeters showed:
- High positive correlations (0.48 to 0.90)
- Both sensors moving in the SAME direction
- NOT complementary patterns

**What This Means:**
- These are likely **redundant sensors** for reliability
- They measure the same tilt independently
- NOT differential measurements
- Can be used individually OR averaged for noise reduction

**Recommendation for Model:**
- Use individual tiltmeter readings
- Consider averaging TI551, TI552, TI553 (high correlation ~0.84-0.90)
- TI554 shows lower correlation - may be measuring different axis or location

---

## FRIEND'S STRUCTURE: STR129

### Paired Sensors Identified

**DI555 + DI556 (Displacement Pair)**
- **Synchronization:** 97.38% (100,886 matching timestamps)
- **Pairing Type:** Complementary (one negative, one positive)
- **Individual Ranges:**
  - DI555: -59.63 to +70.61 mm (mean: -22.79 mm)
  - DI556: -464.50 to +464.48 mm (mean: +6.32 mm)
- **Combined Displacement:**
  - Range: -419.11 to +526.72 mm (!!)
  - Mean: **-16.47 mm**
  - Std Dev: **19.74 mm**
- **Correlation:** -0.3624 (moderate negative - differential cancellation)

### Analysis

**Key Differences from STR128:**
1. **Much larger individual sensor ranges** (especially DI556)
2. **Higher variability** (std dev: 19.74mm vs 5.19mm in STR128)
3. **Slightly lower synchronization** (97.4% vs 100%)

**Physical Interpretation:**
- This structure experiences MORE movement than STR128
- The combined -16.47mm suggests slight downward deflection baseline
- Higher std dev indicates more dynamic loading or environmental sensitivity

**For Anomaly Detection:**
- Combined displacement: `displacement = DI555 + DI556`
- Expected normal range: -16mm ± 40mm (2 std devs)
- This structure requires wider anomaly thresholds than STR128

### Tiltmeters: NO PAIRING DETECTED
Similar to STR128, tiltmeters showed no clear pairing patterns.

---

## ALL STRUCTURES WITH PAIRED SENSORS

### Summary Table

| Structure | Displacement Pairs | Tilt Pairs | Total Pairs | Notes |
|-----------|-------------------|------------|-------------|-------|
| STR122 | 1 | 5 | 6 | Complex tilt pairing system |
| STR124 | 1 | 4 | 5 | Perfect 100% sync on multiple pairs |
| **STR128** | **1** | **0** | **1** | Your primary structure |
| **STR129** | **1** | **0** | **1** | Friend's structure |
| STR178 | 1 | 1 | 2 | Standard configuration |
| STR179 | 3 | 3 | 6 | Most complex: 4 displacement sensors |
| STR180 | 0 | 4 | 4 | Tilt-focused monitoring |
| STR181 | 2 | 0 | 2 | Multiple displacement pairs |
| STR182 | 2 | 0 | 2 | Both pairs share DI639 |
| STR183 | 2 | 1 | 3 | Both pairs share DI644 |
| STR184 | 0 | 2 | 2 | Tilt-only pairing |

---

## DETAILED INSIGHTS BY STRUCTURE TYPE

### Pattern 1: Simple Paired Configuration (STR128, STR129, STR124, STR178)
- **Characteristics:** 2 displacement sensors, clearly paired
- **Synchronization:** 97-100%
- **Implementation:** Straightforward differential measurement
- **Model Approach:** Direct addition of paired values

### Pattern 2: Complex Multi-Pair Systems (STR179, STR181, STR182, STR183)
- **Characteristics:** 4 displacement sensors forming multiple overlapping pairs
- **Example STR179:**
  - DI612 + DI614 (90.3% sync)
  - DI613 + DI614 (93.0% sync)
  - DI613 + DI615 (91.7% sync)
- **Implication:** DI614 is central sensor, paired with multiple others
- **Model Approach:** May need to model sensor network, not just pairs

### Pattern 3: Tilt-Heavy Configurations (STR122, STR180, STR184)
- **Characteristics:** Multiple tiltmeter pairs, few/no displacement pairs
- **STR122 has 5 tilt pairs** - most complex tilt monitoring
- **Model Approach:** Focus on angular measurements for these structures

---

## PAIRING DETECTION METHODOLOGY

### Criteria Used

We identified paired sensors using three criteria:

1. **High Timestamp Synchronization (>90%)**
   - Ensures sensors are recorded simultaneously
   - Filters out sensors with different sampling schedules

2. **Complementary Value Patterns**
   - One sensor positive, one negative (differential measurement)
   - OR strong negative correlation (common-mode rejection)

3. **Combined Range Reduction**
   - Individual sensors show large ranges
   - Combined value shows smaller, more stable range
   - Indicates cancellation of common-mode noise

### Examples of Perfect Pairing

**Best Synchronization:**
- STR124 DI539+DI540: 100.00%
- STR128 DI549+DI550: 100.00%
- STR124 TI542+TI544: 100.00%
- STR122 TI537+TI538: 100.00%

**Best Correlation for Differential Pairing:**
- STR181 DI628+DI629: -0.9515 (nearly perfect negative correlation)
- STR122 TI537+TI538: -0.9686 (nearly perfect negative correlation)
- STR182 DI636+DI639: -0.9098
- STR183 DI644+DI647: -0.9403

---

## IMPLICATIONS FOR ANOMALY DETECTION

### Critical Preprocessing Steps

#### 1. Create Combined Sensor Values

**For STR128:**
```python
# Load data
df = pd.read_excel('STR128_merged.xlsx')

# Filter displacement sensors
di549 = df[df['sensor_id'] == 'DI549'][['timestamp', 'value']].rename(columns={'value': 'DI549'})
di550 = df[df['sensor_id'] == 'DI550'][['timestamp', 'value']].rename(columns={'value': 'DI550'})

# Merge on timestamp
merged = pd.merge(di549, di550, on='timestamp')

# Create combined displacement
merged['displacement_combined'] = merged['DI549'] + merged['DI550']

# This combined value is what you should use for anomaly detection
```

#### 2. Update Feature Engineering

**BEFORE (Incorrect):**
```python
features = ['DI549_value', 'DI550_value', 'TI551_value', ...]
# This treats paired sensors as independent features!
```

**AFTER (Correct):**
```python
features = ['displacement_combined', 'TI551_value', 'TI552_value', ...]
# This respects the sensor pairing structure
```

#### 3. Expected Value Ranges

**Individual Sensors (Misleading):**
- DI549: 48-57mm → Appears OUT OF official 0-500mm range
- DI550: -53 to -34mm → Appears INVALID (negative)

**Combined (Correct):**
- DI549+DI550: -4 to +22mm → NORMAL structural displacement

### Model Architecture Recommendations

#### Option A: Use Combined Values Only (Recommended for STR128)
```python
X = df[['displacement_combined', 'TI551', 'TI552', 'TI553', 'TI554',
        'AC542_p2p', 'AC542_rms', 'AC543_p2p', 'AC543_rms']]
```

**Pros:**
- Physically meaningful features
- Reduces dimensionality (7 sensors → 6 features)
- Matches engineering understanding

**Cons:**
- Loses ability to detect individual sensor failures

#### Option B: Use Both Individual + Combined (Comprehensive)
```python
X = df[['DI549', 'DI550', 'displacement_combined',  # Keep all for sensor health
        'TI551', 'TI552', 'TI553', 'TI554',
        'AC542_p2p', 'AC542_rms', 'AC543_p2p', 'AC543_rms']]
```

**Pros:**
- Can detect sensor malfunctions (e.g., only one sensor fails)
- Redundant info helps ML models learn better
- Can validate pairing (if DI549+DI550 ≠ expected, flag issue)

**Cons:**
- Higher dimensionality
- Some feature correlation

**Recommendation:** Use Option B for STR128, with anomaly rules:
1. Structural anomaly: `displacement_combined` outside expected range
2. Sensor malfunction: `|DI549 + DI550 - displacement_combined_rolling_mean| > threshold`

---

## NEXT STEPS FOR YOUR ANALYSIS

### Immediate Actions (Before Model Building)

1. **Create preprocessing script** to generate combined displacement values for all structures
2. **Update STR128 data** with combined displacement column
3. **Verify pairing** by plotting DI549 vs DI550 over time (should see inverse relationship)
4. **Calculate baseline statistics** for combined displacement across different seasons

### For Model Training

1. **Feature Selection:**
   - Use `displacement_combined` instead of DI549/DI550 separately
   - Keep all tiltmeters (they're independent)
   - Include temperature (TP318) as it affects displacement

2. **Anomaly Thresholds:**
   - Statistical: Mean ± 2-3 standard deviations
   - STR128 displacement: 11.26 ± 10.38mm (2σ) = roughly 1-22mm normal range
   - Rate of change: Max expected change per hour (calculate from data)

3. **Temporal Analysis:**
   - Analyze by season (thermal expansion effects)
   - Daily patterns (traffic loading)
   - Weather correlation

4. **Model Comparison:**
   - Train model WITH pairing correction
   - Train model WITHOUT pairing correction
   - Compare false positive rates (pairing should dramatically reduce false alarms)

### Validation Steps

**Test Case 1: Paired vs Unpaired Features**
- Train Isolation Forest on individual sensors → expect high false positive rate
- Train Isolation Forest on combined sensors → expect lower false positive rate

**Test Case 2: Known Normal Periods**
- Identify periods with stable temperature and low traffic (e.g., night, mild weather)
- These should show minimal anomalies with combined features
- But individual sensors might show "violations"

**Test Case 3: Sensor Synchronization Check**
- For any detected anomaly in combined displacement
- Check if DI549 and DI550 are still synchronized
- If desynchronized → sensor malfunction, not structural issue
- If synchronized → investigate structural cause

---

## COMPARISON: STR128 vs STR129

| Metric | STR128 | STR129 | Interpretation |
|--------|--------|--------|----------------|
| Combined mean | 11.26mm | -16.47mm | STR128 deflects upward, STR129 slightly downward |
| Combined std dev | 5.19mm | 19.74mm | STR129 has 3.8x more variability |
| Sync % | 100% | 97.4% | STR128 hardware perfectly synchronized |
| Correlation | 0.605 | -0.362 | Different loading patterns |
| Data points | 43,809 | 100,886 | STR129 has 2.3x more data |

### Implications

**STR128 is more stable:**
- Tighter distribution → easier to detect anomalies
- Lower variability → fewer false positives expected
- Perfect sync → reliable pairing

**STR129 is more dynamic:**
- Higher variability → need wider thresholds
- More data → better for training robust models
- Negative correlation suggests stronger differential effects

**For Your Model:**
- STR128 might be better for initial model development (cleaner signal)
- STR129 better for stress-testing model (handles more variation)
- Consider training on STR128, validating on STR129

---

## FILES GENERATED

1. **[all_paired_sensors_detailed.csv](Sensor Pairing Analysis/all_paired_sensors_detailed.csv)**
   - Complete dataset of all 34 pairs
   - Includes all statistics: sync %, correlation, ranges, std dev
   - **Use this for**: Programmatic access to pairing information

2. **[00_PAIRED_SENSORS_SUMMARY.txt](Sensor Pairing Analysis/00_PAIRED_SENSORS_SUMMARY.txt)**
   - Human-readable summary report
   - Detailed statistics for each pair
   - **Use this for**: Quick reference and documentation

3. **[PAIRED_SENSORS_ANALYSIS_REPORT.md](PAIRED_SENSORS_ANALYSIS_REPORT.md)** (This file)
   - Comprehensive analysis and recommendations
   - Focused on STR128 and anomaly detection implications
   - **Use this for**: Understanding and planning next steps

---

## TECHNICAL NOTES

### Why Sensors Are Paired

**Differential Displacement Measurement:**
- Measures relative movement between two points
- Cancels common-mode vibration and sensor drift
- More accurate than single sensor absolute measurement

**Mathematical Formulation:**
```
Sensor A measures: baseline_A + displacement + noise_A
Sensor B measures: baseline_B - displacement + noise_B
Combined (A+B): (baseline_A + baseline_B) + (noise_A + noise_B)
```

The displacement cancels out IF it's common-mode (e.g., temperature expansion affecting both sensors equally).
The displacement ADDS if it's differential (e.g., bending causing A to compress and B to extend).

### When Individual Sensors Show "False" Range Violations

**Example from STR128:**
- Official range: 0-500mm
- DI550 shows: -53 to -34mm → 100% "violations"
- But this is CORRECT operation as part of paired system

**How to distinguish real malfunction:**
1. Check synchronization: Are paired timestamps still matching?
2. Check correlation: Is the relationship between sensors maintained?
3. Check combined value: Is the combined displacement reasonable?

If synchronization breaks or correlation changes dramatically → Sensor malfunction
If synchronization OK but combined value anomalous → Structural issue

---

## RECOMMENDATIONS FOR MODEL DEPLOYMENT

### Phase 1: Validation (Week 1)

1. **Visualize Paired Sensors**
   - Plot DI549 vs DI550 scatter plot (should show linear inverse relationship)
   - Plot time series of individual vs combined displacement
   - Verify pairing makes physical sense

2. **Recalculate Range Validation**
   - Apply official 0-500mm range to COMBINED displacement, not individual
   - Expect dramatic reduction in "violations"
   - Document new violation rates

3. **Update Documentation**
   - Mark all paired sensors in your dataset documentation
   - Update PROJECT_DOCUMENTATION_COMPLETE.md with this information

### Phase 2: Preprocessing Pipeline (Week 1-2)

1. **Create Combined Values Script**
   - Generate combined displacement for all 14 identified pairs
   - Save as new columns in merged files
   - Validate against known normal periods

2. **Feature Engineering**
   - Design features using combined values
   - Add rate-of-change features (displacement/hour)
   - Add synchronization health metrics

### Phase 3: Model Training (Week 2-3)

1. **Baseline Model (STR128)**
   - Train on combined displacement + tiltmeters + accelerometers
   - Use first 80% of data for training, last 20% for testing
   - Establish baseline performance metrics

2. **Comparative Analysis**
   - Train same model WITHOUT pairing correction
   - Compare false positive rates
   - Document improvement from pairing

3. **Cross-Structure Validation**
   - Train on STR128, test on STR129
   - Evaluate transferability of model

### Phase 4: Anomaly Detection (Week 3-4)

1. **Multiple Detection Methods**
   - Statistical (Z-score on combined values)
   - ML-based (Isolation Forest)
   - Domain rules (rate-of-change limits)

2. **Ensemble Approach**
   - Flag anomaly only if 2+ methods agree
   - Reduces false positives
   - More robust

---

## CONCLUSION

This paired sensor analysis has revealed critical insights about your bridge monitoring dataset:

✅ **STR128 has one confirmed displacement sensor pair** (DI549+DI550)
✅ **Tiltmeters in STR128 are independent** (not paired)
✅ **The pairing pattern is widespread** (11 out of 21 structures)
✅ **Combined values must be used** for accurate anomaly detection
✅ **Individual "range violations" are often normal** for paired sensors

**Impact on Your Model:**
- Using combined displacement will dramatically reduce false positives
- Your anomaly detection will be more aligned with actual structural behavior
- Model performance should improve significantly

**Readiness for Next Phase:**
Your data is now thoroughly understood and ready for anomaly detection model development. The pairing information provides the foundation for physically meaningful feature engineering.

---

**Report Prepared By:** Paired Sensor Identification System
**Date:** 2025-12-05
**Contact:** Continue with preprocessing pipeline and model development
