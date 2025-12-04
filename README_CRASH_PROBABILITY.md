# Crash Probability Prediction Model

This model predicts crash risk and probability using all available variables from motorcycle crash data.

## Overview

The model analyzes **3,426 motorcycle crash records** from 2024 and predicts:
1. **Crash Risk Score** - A composite score (0-13+) indicating overall crash risk
2. **High Risk Probability** - Probability that a scenario falls into the high-risk category (top 30% of risk scores)

## Model Performance

- **Logistic Regression**: 97% accuracy, 0.998 ROC AUC
- **Random Forest**: 99% accuracy, 0.999 ROC AUC

Both models excel at identifying high-risk crash scenarios.

## Features Used

The model uses **13 features** to predict crash risk:

1. **engine_size** - Motorcycle engine size (cc)
2. **helmet_used** - Whether rider wears helmet (1/0)
3. **helmet_type** - Type of helmet (0-3)
4. **has_boots** - Wearing boots (1/0)
5. **has_eyeprot** - Wearing eye protection (1/0)
6. **has_longpants** - Wearing long pants (1/0)
7. **has_longsleeves** - Wearing long sleeves (1/0)
8. **has_bag** - Has bag (1/0)
9. **has_passenger** - Has passenger (1/0)
10. **has_trailer** - Has trailer (1/0)
11. **num_units** - Number of units involved in crash
12. **passenger_helmet** - Passenger wearing helmet (1/0)
13. **safety_score** - Composite safety score (0-5, sum of protective gear)

## Risk Score Calculation

The risk score is calculated based on:

- **No helmet**: +3 points
- **No eye protection**: +2 points
- **No boots/pants/sleeves**: +1 point each
- **Has passenger**: +2 points
- **Has trailer**: +1 point
- **Very large engine (>1500cc)**: +1 point
- **Multiple units involved**: +2 points

**Risk Levels**:
- **Low**: 0-3 points
- **Medium**: 4-6 points
- **High**: 7-9 points
- **Very High**: 10+ points

## Key Insights

### 1. Safety Score is Most Important
The composite **safety_score** (sum of protective gear) has the highest feature importance (22%), showing that overall safety practices are the strongest predictor.

### 2. Helmet Usage Critical
**helmet_used** is the second most important feature (17.8%), confirming that helmet usage significantly impacts crash risk.

### 3. Eye Protection Matters
**has_eyeprot** ranks third (12.5%), showing eye protection is a strong indicator of overall safety behavior.

### 4. Engine Size Patterns
- **Very Large bikes (1500+cc)**: Highest average risk score (5.10)
- **Large bikes (1000-1500cc)**: Lowest average risk score (4.01)
- **Medium bikes (500-1000cc)**: Moderate risk (4.14)
- **Small bikes (0-500cc)**: Moderate risk (4.19)

### 5. Safety Score Impact
Risk scores decrease dramatically with more protective gear:
- **Safety Score 0** (no gear): Avg risk score 8.77
- **Safety Score 5** (full gear): Avg risk score 0.98

This shows a clear inverse relationship between protective gear and crash risk.

## Usage

### Training the Model

```bash
python3 crash_probability_model.py
```

This will:
- Load and preprocess the data
- Train Logistic Regression and Random Forest models
- Generate visualizations
- Save models (`crash_risk_lr_model.pkl`, `crash_risk_rf_model.pkl`)
- Save scaler (`crash_risk_scaler.pkl`)

### Making Predictions

```python
from predict_crash_probability import predict_crash_risk

# Low risk scenario
result = predict_crash_risk(
    engine_size=1000,
    helmet_used=1,
    helmet_type=1,
    has_boots=1,
    has_eyeprot=1,
    has_longpants=1,
    has_longsleeves=1,
    has_bag=0,
    has_passenger=0,
    has_trailer=0,
    num_units=1,
    passenger_helmet=0
)

print(f"Risk Score: {result['risk_score']}")
print(f"Risk Level: {result['risk_level']}")
print(f"High Risk Probability: {result['high_risk_probability_lr']:.2%}")
```

### Command Line Example

```bash
python3 predict_crash_probability.py
```

## Output Files

### Model Files
- `crash_risk_lr_model.pkl` - Logistic Regression model
- `crash_risk_rf_model.pkl` - Random Forest model
- `crash_risk_scaler.pkl` - Feature scaler

### Visualizations
- `crash_risk_feature_importance.png` - Feature importance from Random Forest
- `crash_risk_roc_curves.png` - ROC curves for both models
- `crash_risk_distribution.png` - Distribution of risk scores
- `crash_risk_by_engine_size.png` - Risk by engine size category
- `crash_risk_by_safety_score.png` - Risk by safety score

## Merging with Additional Crash Data

To add more features (weather, time, location, etc.), uncomment and update in `crash_probability_model.py`:

```python
crash_filepath = 'CRASH_2024.csv'  # Your crash data file
df = merge_with_crash_data(cycle_df, crash_filepath)
```

The model will automatically detect and use:
- Weather conditions
- Time of day
- Municipality/County
- Road type
- Rider age
- Crash severity

## Interpretation

### Risk Score
- **0-3**: Low risk - Good safety practices
- **4-6**: Medium risk - Some safety measures missing
- **7-9**: High risk - Multiple safety issues
- **10+**: Very high risk - Critical safety concerns

### High Risk Probability
- **0-30%**: Low probability of high-risk scenario
- **30-70%**: Medium probability
- **70-100%**: High probability of high-risk scenario

## Recommendations

The model provides automatic recommendations based on the scenario:
- ⚠️ Wear a helmet - most critical safety measure
- ⚠️ Wear eye protection - significantly reduces risk
- ⚠️ Wear full protective gear
- ⚠️ Ensure passenger wears helmet
- ⚠️ Extra caution for larger bikes
- ⚠️ Multiple vehicle scenarios increase risk

## Limitations

1. **All records are crashes**: The model predicts risk within crash scenarios, not crash vs. no-crash
2. **Missing contextual data**: Would benefit from weather, time, location data
3. **Correlation vs. Causation**: Model identifies patterns, not necessarily causation
4. **Sample size**: Some categories have smaller sample sizes

## Future Enhancements

1. Merge with non-crash data to predict crash occurrence
2. Add temporal features (time of day, day of week, season)
3. Add geographic features (municipality, road type)
4. Add weather and road condition features
5. Predict crash severity (fatal, injury, property damage)

## Public Safety Applications

This model can be used for:
1. **Risk Assessment**: Identify high-risk riding scenarios
2. **Safety Campaigns**: Target specific risk factors
3. **Policy Development**: Inform helmet laws and safety regulations
4. **Education**: Demonstrate impact of protective gear
5. **Enforcement**: Prioritize high-risk scenarios

## Contact

For questions or issues, please refer to the main project documentation.

