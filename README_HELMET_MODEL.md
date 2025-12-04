# Helmet Usage Prediction Model

This model predicts whether a motorcycle rider will wear a helmet based on various conditions like weather, age, time of day, municipality, road type, engine size, and other protective gear usage.

## Overview

The model uses **logistic regression** to classify helmet usage (binary classification: helmet worn = 1, not worn = 0). It's designed as a public-safety style model to identify patterns in helmet compliance.

## Features

The model currently uses features from `CYCLE_2024.csv`:
- **Engine Size**: Motorcycle engine size (cc)
- **Helmet Type**: Type of helmet (0 = no helmet, 1-3 = different helmet types)
- **Protective Gear Indicators**: Boots, eye protection, long pants, long sleeves, bag
- **Passenger Indicator**: Whether there's a passenger
- **Trailer Indicator**: Whether the motorcycle has a trailer

### Merging with Main Crash Dataset

The model is designed to merge with a main crash dataset (if available) to add additional features:
- **Weather conditions**
- **Time of day** (extracted from crash time)
- **Municipality/County**
- **Road type**
- **Age** of the rider

To merge with crash data, uncomment and update the merge line in `main()`:
```python
crash_filepath = 'CRASH_2024.csv'  # Update with your crash data file path
df = merge_with_crash_data(cycle_df, crash_filepath)
```

The merge uses `CRN` (Crash Report Number) as the key.

## Usage

### Basic Usage

```bash
python3 helmet_usage_model.py
```

### Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Model Performance

The model achieves:
- **Accuracy**: ~89%
- **ROC AUC Score**: ~0.92
- **F1-Score**: ~0.89

### Key Findings

1. **Helmet Type** is the strongest predictor (coefficient: 2.22)
2. **Eye Protection** usage is positively correlated with helmet usage
3. **Engine Size** shows interesting patterns:
   - Medium-sized bikes (500-1000cc) have highest helmet usage (~55%)
   - Very large bikes (1500+cc) have lower helmet usage (~47%)

## Output Files

The script generates:

1. **Model Files**:
   - `helmet_usage_model.pkl` - Trained logistic regression model
   - `helmet_usage_scaler.pkl` - Feature scaler for preprocessing

2. **Visualizations**:
   - `helmet_model_feature_importance.png` - Feature importance plot
   - `helmet_model_roc_curve.png` - ROC curve for model evaluation
   - `helmet_usage_by_engine_size.png` - Helmet usage by engine size category
   - `helmet_usage_by_time.png` - Helmet usage by time of day (if time data available)
   - `helmet_usage_by_location.png` - Helmet usage by location (if location data available)

## Pattern Analysis

The model identifies patterns such as:

- **Certain counties/municipalities** → lower helmet compliance
- **Nighttime** → less helmet usage (if time data available)
- **Larger engine size bikes** → patterns vary by size category
- **Protective gear correlation** → riders wearing other protective gear are more likely to wear helmets

## Using the Trained Model

To use the saved model for predictions:

```python
import joblib
import pandas as pd
import numpy as np

# Load model and scaler
model = joblib.load('helmet_usage_model.pkl')
scaler = joblib.load('helmet_usage_scaler.pkl')

# Prepare your data (same features as training)
# ... preprocess your data ...

# Scale features
X_scaled = scaler.transform(X)

# Make predictions
predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)[:, 1]
```

## Data Preprocessing

The model handles:
- Missing values (filled with median/mode)
- Unknown values (99999 for engine size, 'U' for indicators)
- Categorical encoding (for locations, road types, weather)
- Feature scaling (StandardScaler)

## Notes

- The model uses **class_weight='balanced'** to handle imbalanced classes
- All features are standardized before training
- The model splits data 80/20 for training/testing
- Stratified sampling ensures balanced class distribution in splits


