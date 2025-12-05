# STR128 TILTMETER ANALYSIS - Why They're NOT Paired

**Structure:** STR128
**Date:** 2025-12-05
**Question:** Why aren't the 4 tiltmeters (TI551-TI554) showing paired behavior like the displacement sensors?

---

## TL;DR - Quick Answer

**The tiltmeters are NOT differential pairs because:**
1. They all show **POSITIVE correlations** (0.34 to 0.90)
2. They all measure in the **SAME direction** (all averaging around -17° except TI554)
3. TI551, TI552, TI553 form a **redundant sensor group** (highly correlated)
4. TI554 is **different** - likely measures a different axis or location

**For your model:** Use them as separate features OR average TI551-TI553 for noise reduction.

---

## Detailed Statistical Analysis

### Individual Tiltmeter Statistics

| Sensor | Mean | Std Dev | Min | Max | Range | Data Points |
|--------|------|---------|-----|-----|-------|-------------|
| **TI551** | -17.32° | 4.51° | -26.11° | +26.11° | 52.22° | 44,050 |
| **TI552** | -17.16° | 4.11° | -26.12° | +26.13° | 52.25° | 44,418 |
| **TI553** | -17.19° | 4.27° | -26.53° | +26.53° | 53.06° | 44,362 |
| **TI554** | **-0.82°** | 1.74° | -26.49° | +26.50° | 52.99° | 43,545 |

### Key Observations

1. **TI551, TI552, TI553 are very similar:**
   - All average around -17° (negative tilt)
   - Similar standard deviations (~4°)
   - Nearly identical ranges (~52°)
   - **These are measuring the SAME tilt angle**

2. **TI554 is different:**
   - Average close to zero (-0.82°)
   - Much lower std dev (1.74° vs ~4°)
   - **Likely measuring a different axis or location**

---

## Correlation Matrix

### Pairwise Correlations

|        | TI551 | TI552 | TI553 | TI554 |
|--------|-------|-------|-------|-------|
| **TI551** | 1.0000 | **0.7768** | **0.8962** | 0.4803 |
| **TI552** | **0.7768** | 1.0000 | **0.8433** | 0.3394 |
| **TI553** | **0.8962** | **0.8433** | 1.0000 | 0.4774 |
| **TI554** | 0.4803 | 0.3394 | 0.4774 | 1.0000 |

**Average correlation:** 0.636 (moderate-to-strong POSITIVE)

### What This Means

**High Positive Correlation = NOT Differential Pairs**

For comparison:
- **Displacement pair** DI549+DI550: correlation = **+0.60** (but COMPLEMENTARY values: one +52mm, one -41mm)
- **Tiltmeters** TI551+TI552: correlation = **+0.78** AND SIMILAR values (both around -17°)

The key difference:
- Displacement sensors: opposite signs, positive correlation → **Differential pair**
- Tiltmeters: same sign, positive correlation → **Redundant sensors**

---

## Why Are They Configured This Way?

### Hypothesis 1: Redundancy for Reliability ✓ MOST LIKELY

**Configuration:**
- TI551, TI552, TI553 measure the **same tilt axis** from nearly the same location
- Provides redundancy: if one sensor fails, others continue working
- Common in critical infrastructure monitoring

**Evidence:**
- High correlations (0.78-0.90) between TI551-TI553
- Nearly identical mean values (-17.16° to -17.32°)
- Very similar behavior patterns

**Purpose:**
- Fault tolerance
- Cross-validation of readings
- Average readings to reduce sensor noise

### Hypothesis 2: Different Axes or Locations (TI554)

**TI554 shows different behavior:**
- Mean near zero (-0.82°) vs. -17° for others
- Lower correlation with other tiltmeters (0.34-0.48)
- Smaller variability (1.74° vs ~4°)

**Possible explanations:**
- Measures **perpendicular axis** (e.g., if TI551-TI553 measure X-axis, TI554 measures Y-axis)
- Located at **different position** on bridge (e.g., different span or support)
- Different **sensitivity/calibration**

---

## Contrast with Paired Displacement Sensors

### DI549 + DI550 (Displacement - PAIRED)

```
DI549: mean = +52.29 mm (POSITIVE)
DI550: mean = -41.02 mm (NEGATIVE)
Correlation: +0.60
Combined: +11.26 mm (makes physical sense)
```

**This IS a differential pair because:**
- Opposite signs indicate opposite measurement directions
- When combined, they cancel common-mode noise
- Result is actual structural displacement

### TI551 + TI552 (Tilt - NOT PAIRED)

```
TI551: mean = -17.32° (NEGATIVE)
TI552: mean = -17.16° (NEGATIVE)
Correlation: +0.78
Combined: -34.48° (doesn't make physical sense!)
```

**This is NOT a differential pair because:**
- Same sign indicates same measurement direction
- When combined, they just add up (no cancellation)
- They're measuring the same thing, not opposite sides

---

## Physical Interpretation

### What is the -17° Tilt?

The fact that TI551, TI552, TI553 all average around **-17°** suggests:

1. **Permanent Structural Tilt:**
   - The bridge structure has a baseline tilt of ~17° from horizontal
   - This could be:
     - Design feature (bridge on a slope)
     - Long-term settling
     - Reference angle of sensor mounting

2. **±4° Variation:**
   - Standard deviation ~4° indicates dynamic changes
   - Likely due to:
     - Traffic loading (vehicles cause temporary tilt)
     - Temperature (thermal expansion causes tilt changes)
     - Wind loading

3. **Why Three Sensors Measure Same Thing:**
   - **Reliability:** If one fails, you still have two
   - **Validation:** Compare readings to detect sensor malfunction
   - **Noise Reduction:** Average the three for cleaner signal

### What is TI554 Measuring?

TI554's different behavior (-0.82° mean, lower variability) suggests:
- **Different axis:** While TI551-553 measure longitudinal tilt, TI554 might measure lateral tilt
- **Different location:** Could be on a different part of the bridge structure
- **Different purpose:** Might be monitoring a specific structural element

---

## Implications for Your Anomaly Detection Model

### DO NOT Combine Tiltmeters Like Displacement

**Wrong approach:**
```python
# DON'T DO THIS - makes no sense!
tilt_combined = TI551 + TI552  # Would give -34.48°, meaningless!
```

### Recommended Approaches

#### Option 1: Use Individual Sensors (Simplest)

```python
features = [
    'displacement_combined',  # DI549 + DI550
    'TI551',  # Tilt measurement
    'TI552',  # Backup tilt
    'TI553',  # Backup tilt
    'TI554',  # Different axis/location
    'AC542_p2p', 'AC542_rms',  # Accelerometers
    'AC543_p2p', 'AC543_rms'
]
```

**Pros:** Simple, captures all information
**Cons:** High correlation between TI551-553 might confuse some ML models

#### Option 2: Average Redundant Sensors (Recommended)

```python
# Average the three redundant tiltmeters for noise reduction
df['TI_avg'] = (df['TI551'] + df['TI552'] + df['TI553']) / 3

features = [
    'displacement_combined',  # DI549 + DI550
    'TI_avg',                 # Averaged TI551-553
    'TI554',                  # Different measurement
    'AC542_p2p', 'AC542_rms',
    'AC543_p2p', 'AC543_rms'
]
```

**Pros:**
- Reduces sensor noise through averaging
- Reduces feature dimensionality
- More robust to individual sensor failures

**Cons:** Loses ability to detect individual sensor failures in TI551-553

#### Option 3: Keep Separate + Add Averaged (Comprehensive)

```python
df['TI_avg'] = (df['TI551'] + df['TI552'] + df['TI553']) / 3
df['TI_std'] = df[['TI551', 'TI552', 'TI553']].std(axis=1)  # Disagreement metric

features = [
    'displacement_combined',
    'TI_avg',      # Average tilt
    'TI_std',      # Sensor disagreement (high = possible malfunction)
    'TI554',       # Different axis
    'AC542_p2p', 'AC542_rms',
    'AC543_p2p', 'AC543_rms'
]
```

**Pros:**
- Best of both worlds
- Can detect both structural anomalies AND sensor malfunctions
- `TI_std` high → sensors disagree → possible sensor failure

**Cons:** More features (but still manageable)

---

## Sensor Health Monitoring

### Detecting Tiltmeter Malfunctions

Since TI551-553 measure the same thing, you can use their agreement as a health check:

```python
# Calculate standard deviation among the three
tilt_std = df[['TI551', 'TI552', 'TI553']].std(axis=1)

# Normal: std should be small (sensors agree)
# Anomaly: std suddenly increases (one sensor diverges)

threshold = 2.0  # degrees
if tilt_std > threshold:
    print("WARNING: Tiltmeters disagree - possible sensor malfunction")
```

### Example Scenarios

**Scenario 1: Normal Operation**
```
TI551 = -17.2°
TI552 = -17.4°
TI553 = -17.1°
Std Dev = 0.15° ✓ Good agreement
```

**Scenario 2: Sensor Malfunction**
```
TI551 = -17.2°
TI552 = -25.8°  ← SENSOR FAILURE!
TI553 = -17.1°
Std Dev = 5.02° ✗ High disagreement
```

**Scenario 3: True Structural Anomaly**
```
TI551 = -22.4°
TI552 = -22.7°
TI553 = -22.3°
Std Dev = 0.20° ✓ Good agreement, but all shifted from -17° baseline
→ Real structural event, not sensor issue
```

---

## Comparison with Other Structures

### Do Other Structures Show Tilt Pairing?

Let's check from our analysis:

**Structures WITH tilt pairs:**
- STR122: 5 tilt pairs (complex system)
- STR124: 4 tilt pairs
- STR178: 1 tilt pair
- STR179: 3 tilt pairs
- STR180: 4 tilt pairs
- STR183: 1 tilt pair
- STR184: 2 tilt pairs

**STR128: 0 tilt pairs** ← Your structure

### Why the Difference?

**STR128's approach: Redundancy**
- Multiple sensors measuring same thing
- Focus on reliability
- Simpler configuration

**Other structures' approach: Differential**
- Sensors measuring opposite directions
- More complex but provides richer data
- Can measure bending/torsion more accurately

Both approaches are valid engineering choices depending on:
- Bridge design
- Criticality of monitoring
- Budget constraints
- Maintenance philosophy

---

## Recommendations for Model Building

### 1. Feature Engineering

**Recommended feature set for STR128:**
```python
# Primary structural features
'displacement_combined'  # DI549 + DI550 (11.26mm baseline)
'tilt_primary'          # Average of TI551-553 (-17° baseline)
'tilt_secondary'        # TI554 (-0.82° baseline)

# Sensor health features
'tilt_agreement'        # Std dev of TI551-553 (should be low)
'displacement_sync'     # Abs(DI549 + DI550 - expected) (should be low)

# Environmental/loading features
'vibration_p2p'         # Average of AC542/543 p2p
'vibration_rms'         # Average of AC542/543 rms
'temperature'           # From TP sensor
```

### 2. Baseline Values

Set these as your "normal" reference:
- Displacement: 11.26 ± 10mm
- Primary tilt: -17° ± 8°
- Secondary tilt: -0.82° ± 3.5°

### 3. Anomaly Detection Strategy

**Level 1: Sensor Health**
- Check tilt_agreement < 2°
- Check displacement_sync < 5mm
- If violated → Sensor malfunction, not structural issue

**Level 2: Structural Anomalies**
- Only if Level 1 passes
- Check if displacement or tilt outside expected ranges
- Check rate of change (rapid changes more concerning)

### 4. Don't Expect Tilt Pairing in STR128

Your structure simply doesn't have differential tilt measurement. This is OKAY! Many bridges use redundant sensors instead. Your model should:
- Use tiltmeters as absolute angle measurements
- Not look for "combined tilt" like you do for displacement
- Focus on changes from baseline (-17° for primary tilt)

---

## Summary

### Why STR128 Tiltmeters Aren't Paired

1. ✓ **All positive correlations** (0.34-0.90) - paired sensors show negative correlation
2. ✓ **Same-direction measurements** - paired sensors show opposite directions
3. ✓ **Three sensors (TI551-553) are nearly identical** - redundancy, not differentiation
4. ✓ **TI554 is independent** - different axis or location, not part of a pair

### What This Means for You

**Good news:**
- Your displacement sensors ARE properly paired (DI549+DI550) ✓
- You have redundant tilt sensors for reliability ✓
- You can use sensor agreement to detect malfunctions ✓

**For your model:**
- Average TI551-553 for cleaner tilt signal
- Keep TI554 separate (different measurement)
- Don't try to combine tiltmeters like you do displacement
- Focus on detecting changes from baseline values

### Next Steps

1. ✓ Create combined displacement feature (DI549 + DI550)
2. ✓ Create averaged tilt feature (TI551 + TI552 + TI553) / 3
3. ✓ Add sensor health features (tilt disagreement, displacement sync)
4. ✓ Establish baseline values for normal operation
5. → Ready to build anomaly detection model!

---

**Author:** Sensor Pairing Analysis System
**Date:** 2025-12-05
**Status:** Analysis complete, ready for model development
