"""
Crash Probability Prediction Model
Predicts crash probability and risk factors using all available variables.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_data(filepath='CYCLE_2024.csv'):
    """Load the crash data."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded {len(df)} crash records")
    return df


def merge_with_crash_data(cycle_df, crash_filepath=None):
    """
    Merge cycle data with main crash dataset if available.
    Adds: weather, age, time of day, municipality, road type, crash severity, etc.
    """
    if crash_filepath:
        try:
            crash_df = pd.read_csv(crash_filepath, low_memory=False)
            merged_df = cycle_df.merge(crash_df, on='CRN', how='left')
            print(f"Merged with crash data: {len(merged_df)} records")
            return merged_df
        except FileNotFoundError:
            print(f"Warning: Crash data file {crash_filepath} not found. Using cycle data only.")
            return cycle_df
    return cycle_df


def create_risk_score_target(df):
    """
    Create a risk score based on available variables.
    Since all records are crashes, we create a composite risk score.
    Higher risk = more dangerous conditions/variables.
    """
    risk_score = 0
    
    # Helmet usage (no helmet = higher risk)
    helmet_used = df['MC_DVR_HLMTON_IND'].apply(lambda x: 0 if str(x).upper() == 'Y' else 1)
    risk_score += helmet_used * 3  # High weight
    
    # No eye protection (higher risk)
    no_eyeprot = df['MC_DVR_EYEPRT_IND'].apply(lambda x: 0 if str(x).upper() == 'Y' else 1)
    risk_score += no_eyeprot * 2
    
    # No protective gear (boots, pants, sleeves)
    no_boots = df['MC_DVR_BOOTS_IND'].apply(lambda x: 0 if str(x).upper() == 'Y' else 1)
    no_pants = df['MC_DVR_LNGPNTS_IND'].apply(lambda x: 0 if str(x).upper() == 'Y' else 1)
    no_sleeves = df['MC_DVR_LNGSLV_IND'].apply(lambda x: 0 if str(x).upper() == 'Y' else 1)
    risk_score += (no_boots + no_pants + no_sleeves) * 1
    
    # Passenger present (higher risk)
    has_passenger = df['MC_PASSNGR_IND'].apply(lambda x: 1 if str(x).upper() == 'Y' else 0)
    risk_score += has_passenger * 2
    
    # Trailer (higher risk)
    has_trailer = df['MC_TRAIL_IND'].apply(lambda x: 1 if str(x).upper() == 'Y' else 0)
    risk_score += has_trailer * 1
    
    # Engine size (very large = potentially higher risk)
    engine_size = pd.to_numeric(df['MC_ENGINE_SIZE'], errors='coerce')
    engine_size = engine_size.fillna(engine_size.median())
    engine_size = engine_size.replace(99999, engine_size.median())
    # Normalize engine size contribution
    large_engine = (engine_size > 1500).astype(int)
    risk_score += large_engine * 1
    
    # Multiple units involved (from UNIT_NUM)
    multiple_units = (pd.to_numeric(df['UNIT_NUM'], errors='coerce') > 1).astype(int)
    risk_score += multiple_units * 2
    
    return risk_score


def preprocess_features(df):
    """
    Preprocess all features for crash probability prediction.
    """
    print("\nPreprocessing features...")
    data = df.copy()
    
    features = {}
    
    # Engine size
    data['MC_ENGINE_SIZE'] = pd.to_numeric(data['MC_ENGINE_SIZE'], errors='coerce')
    valid_engine_sizes = data[data['MC_ENGINE_SIZE'] < 99999]['MC_ENGINE_SIZE']
    median_engine = valid_engine_sizes.median()
    data['engine_size'] = data['MC_ENGINE_SIZE'].fillna(median_engine)
    data.loc[data['engine_size'] >= 99999, 'engine_size'] = median_engine
    features['engine_size'] = data['engine_size']
    
    # Engine size categories
    data['engine_size_category'] = pd.cut(
        data['engine_size'],
        bins=[0, 500, 1000, 1500, float('inf')],
        labels=['Small', 'Medium', 'Large', 'Very Large']
    )
    
    # Helmet usage and type
    data['helmet_used'] = data['MC_DVR_HLMTON_IND'].apply(
        lambda x: 1 if str(x).upper() == 'Y' else 0
    )
    features['helmet_used'] = data['helmet_used']
    
    data['helmet_type'] = pd.to_numeric(data['MC_DVR_HLMT_TYPE'], errors='coerce')
    data['helmet_type'] = data['helmet_type'].fillna(0)
    data.loc[data['helmet_type'] == 9, 'helmet_type'] = 0
    features['helmet_type'] = data['helmet_type']
    
    # Protective gear indicators
    gear_cols = {
        'MC_DVR_BOOTS_IND': 'has_boots',
        'MC_DVR_EYEPRT_IND': 'has_eyeprot',
        'MC_DVR_LNGPNTS_IND': 'has_longpants',
        'MC_DVR_LNGSLV_IND': 'has_longsleeves',
        'MC_BAG_IND': 'has_bag'
    }
    
    for col, name in gear_cols.items():
        if col in data.columns:
            data[name] = data[col].apply(lambda x: 1 if str(x).upper() == 'Y' else 0)
            features[name] = data[name]
    
    # Passenger indicator
    if 'MC_PASSNGR_IND' in data.columns:
        data['has_passenger'] = data['MC_PASSNGR_IND'].apply(
            lambda x: 1 if str(x).upper() == 'Y' else 0
        )
        features['has_passenger'] = data['has_passenger']
    
    # Trailer indicator
    if 'MC_TRAIL_IND' in data.columns:
        data['has_trailer'] = data['MC_TRAIL_IND'].apply(
            lambda x: 1 if str(x).upper() == 'Y' else 0
        )
        features['has_trailer'] = data['has_trailer']
    
    # Number of units involved
    data['num_units'] = pd.to_numeric(data['UNIT_NUM'], errors='coerce').fillna(1)
    features['num_units'] = data['num_units']
    
    # Passenger protective gear (if passenger present)
    if 'MC_PAS_HLMTON_IND' in data.columns:
        data['passenger_helmet'] = data.apply(
            lambda row: 1 if (row['has_passenger'] == 1 and 
                            str(row['MC_PAS_HLMTON_IND']).upper() == 'Y') else 0,
            axis=1
        )
        features['passenger_helmet'] = data['passenger_helmet']
    
    # Safety score (composite of protective gear)
    data['safety_score'] = (
        data.get('has_boots', 0) + 
        data.get('has_eyeprot', 0) + 
        data.get('has_longpants', 0) + 
        data.get('has_longsleeves', 0) + 
        data['helmet_used']
    )
    features['safety_score'] = data['safety_score']
    
    # If merged with crash data, add those features
    # Time of day
    time_cols = ['CRASH_TIME', 'TIME_OF_DAY', 'HOUR', 'CRASH_HOUR']
    for col in time_cols:
        if col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data['hour'] = pd.to_datetime(data[col], errors='coerce').dt.hour
                except:
                    data['hour'] = pd.to_numeric(data[col].str[:2], errors='coerce')
            else:
                data['hour'] = data[col]
            
            data['hour'] = data['hour'].fillna(12)
            features['hour'] = data['hour']
            
            # Time categories
            data['time_of_day'] = pd.cut(
                data['hour'],
                bins=[0, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening']
            )
            break
    
    # Weather (if available)
    weather_cols = ['WEATHER', 'WEATHER_COND', 'WEATHER_CONDITION']
    for col in weather_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[f'{col}_encoded'] = le.fit_transform(data[col].fillna('Unknown'))
            features[f'{col}_encoded'] = data[f'{col}_encoded']
            data[f'{col}_labels'] = data[col].fillna('Unknown')
            break
    
    # Location (if available)
    location_cols = ['MUNICIPALITY', 'COUNTY', 'COUNTY_NAME', 'MUNI_NAME']
    for col in location_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[f'{col}_encoded'] = le.fit_transform(data[col].fillna('Unknown'))
            features[f'{col}_encoded'] = data[f'{col}_encoded']
            data[f'{col}_labels'] = data[col].fillna('Unknown')
            break
    
    # Road type (if available)
    road_cols = ['ROAD_TYPE', 'ROAD_SURFACE', 'SURFACE_TYPE']
    for col in road_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[f'{col}_encoded'] = le.fit_transform(data[col].fillna('Unknown'))
            features[f'{col}_encoded'] = data[f'{col}_encoded']
            break
    
    # Age (if available)
    age_cols = ['AGE', 'DRIVER_AGE', 'UNIT_AGE']
    for col in age_cols:
        if col in data.columns:
            data['age'] = pd.to_numeric(data[col], errors='coerce')
            data['age'] = data['age'].fillna(data['age'].median())
            features['age'] = data['age']
            
            # Age categories
            data['age_category'] = pd.cut(
                data['age'],
                bins=[0, 25, 35, 50, 100],
                labels=['Young', 'Adult', 'Middle', 'Senior']
            )
            break
    
    # Crash severity (if available)
    severity_cols = ['CRASH_SEVERITY', 'SEVERITY', 'FATALITY', 'INJURY_SEVERITY']
    for col in severity_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[f'{col}_encoded'] = le.fit_transform(data[col].fillna('Unknown'))
            features[f'{col}_encoded'] = data[f'{col}_encoded']
            data[f'{col}_labels'] = data[col].fillna('Unknown')
            break
    
    # Create feature matrix
    feature_df = pd.DataFrame(features)
    
    print(f"\nFeatures created: {list(feature_df.columns)}")
    print(f"Total features: {len(feature_df.columns)}")
    print(f"Valid records: {len(feature_df)}")
    
    # Create metadata for analysis
    metadata_cols = ['CRN', 'engine_size_category', 'safety_score']
    if 'time_of_day' in data.columns:
        metadata_cols.append('time_of_day')
    if any(f'{col}_labels' in data.columns for col in location_cols):
        for col in location_cols:
            if f'{col}_labels' in data.columns:
                metadata_cols.append(f'{col}_labels')
                break
    if 'age_category' in data.columns:
        metadata_cols.append('age_category')
    if any(f'{col}_labels' in data.columns for col in severity_cols):
        for col in severity_cols:
            if f'{col}_labels' in data.columns:
                metadata_cols.append(f'{col}_labels')
                break
    
    metadata = data[[col for col in metadata_cols if col in data.columns]]
    
    return feature_df, metadata, data


def create_target_variables(data):
    """
    Create target variables for prediction.
    Since all records are crashes, we create:
    1. Risk score (continuous)
    2. High risk binary (risk score > threshold)
    """
    print("\nCreating target variables...")
    
    # Create risk score
    risk_score = create_risk_score_target(data)
    
    # High risk threshold (top 30% of risk scores)
    threshold = risk_score.quantile(0.7)
    high_risk = (risk_score > threshold).astype(int)
    
    print(f"Risk score statistics:")
    print(f"  Mean: {risk_score.mean():.2f}")
    print(f"  Median: {risk_score.median():.2f}")
    print(f"  Min: {risk_score.min()}")
    print(f"  Max: {risk_score.max()}")
    print(f"  High risk threshold: {threshold:.2f}")
    print(f"  High risk cases: {high_risk.sum()} ({high_risk.mean():.2%})")
    
    return risk_score, high_risk


def train_models(X, y_risk, y_high_risk, test_size=0.2, random_state=42):
    """Train multiple models for crash probability prediction."""
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    # Split data
    X_train, X_test, y_risk_train, y_risk_test = train_test_split(
        X, y_risk, test_size=test_size, random_state=random_state
    )
    _, _, y_high_risk_train, y_high_risk_test = train_test_split(
        X, y_high_risk, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    results = {}
    
    # 1. Logistic Regression for High Risk Classification
    print("\n1. Training Logistic Regression (High Risk Classification)...")
    lr_model = LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train_scaled, y_high_risk_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nLogistic Regression - Classification Report:")
    print(classification_report(y_high_risk_test, lr_pred))
    
    if len(np.unique(y_high_risk_test)) > 1:
        lr_auc = roc_auc_score(y_high_risk_test, lr_proba)
        print(f"ROC AUC: {lr_auc:.4f}")
    
    models['logistic_regression'] = lr_model
    results['logistic_regression'] = {
        'predictions': lr_pred,
        'probabilities': lr_proba,
        'y_test': y_high_risk_test
    }
    
    # 2. Random Forest for High Risk Classification
    print("\n2. Training Random Forest (High Risk Classification)...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
    rf_model.fit(X_train, y_high_risk_train)
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    
    print("\nRandom Forest - Classification Report:")
    print(classification_report(y_high_risk_test, rf_pred))
    
    if len(np.unique(y_high_risk_test)) > 1:
        rf_auc = roc_auc_score(y_high_risk_test, rf_proba)
        print(f"ROC AUC: {rf_auc:.4f}")
    
    models['random_forest'] = rf_model
    results['random_forest'] = {
        'predictions': rf_pred,
        'probabilities': rf_proba,
        'y_test': y_high_risk_test
    }
    
    # 3. Feature importance from Random Forest
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("="*60)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.to_string(index=False))
    
    return models, scaler, X_train, X_test, y_high_risk_test, results, feature_importance


def analyze_crash_patterns(data, metadata, risk_score):
    """Analyze patterns in crash risk."""
    print("\n" + "="*60)
    print("CRASH RISK PATTERN ANALYSIS")
    print("="*60)
    
    # Create analysis dataframe - reset all indices to align
    analysis_df = metadata.reset_index(drop=True).copy()
    analysis_df['risk_score'] = risk_score.values if hasattr(risk_score, 'values') else risk_score
    
    if 'safety_score' in data.columns:
        analysis_df['safety_score'] = data['safety_score'].reset_index(drop=True).values
    
    # Engine size patterns
    if 'engine_size_category' in analysis_df.columns:
        print("\n1. Average Risk Score by Engine Size:")
        engine_pattern = analysis_df.groupby('engine_size_category')['risk_score'].agg(['mean', 'count', 'std']).reset_index()
        engine_pattern = engine_pattern.rename(columns={
            'engine_size_category': 'Engine Size',
            'mean': 'Avg Risk Score',
            'count': 'Count',
            'std': 'Std Dev'
        })
        print(engine_pattern.to_string(index=False))
    
    # Safety score patterns
    if 'safety_score' in analysis_df.columns:
        print("\n2. Average Risk Score by Safety Score:")
        safety_pattern = analysis_df.groupby('safety_score')['risk_score'].agg(['mean', 'count']).reset_index()
        safety_pattern = safety_pattern.rename(columns={
            'safety_score': 'Safety Score',
            'mean': 'Avg Risk Score',
            'count': 'Count'
        })
        print(safety_pattern.to_string(index=False))
    
    # Time of day patterns
    if 'time_of_day' in analysis_df.columns:
        print("\n3. Average Risk Score by Time of Day:")
        time_pattern = analysis_df.groupby('time_of_day')['risk_score'].agg(['mean', 'count']).reset_index()
        time_pattern = time_pattern.rename(columns={
            'time_of_day': 'Time of Day',
            'mean': 'Avg Risk Score',
            'count': 'Count'
        })
        print(time_pattern.to_string(index=False))
    
    # Location patterns
    location_col = None
    for col in analysis_df.columns:
        if 'labels' in col and ('COUNTY' in col or 'MUNICIPALITY' in col or 'MUNI' in col):
            location_col = col
            break
    
    if location_col:
        print(f"\n4. Average Risk Score by Location ({location_col}):")
        location_pattern = analysis_df.groupby(location_col)['risk_score'].agg(['mean', 'count']).reset_index()
        location_pattern = location_pattern.rename(columns={
            location_col: 'Location',
            'mean': 'Avg Risk Score',
            'count': 'Count'
        })
        location_pattern = location_pattern.sort_values('Avg Risk Score', ascending=False)
        print(location_pattern.head(15).to_string(index=False))
    
    return analysis_df


def create_visualizations(models, X_test, results, feature_importance, analysis_df, risk_score):
    """Create visualizations for crash probability model."""
    print("\nCreating visualizations...")
    
    # 1. Feature importance plot
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Features for Crash Risk Prediction (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('crash_risk_feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: crash_risk_feature_importance.png")
    plt.close()
    
    # 2. ROC Curves for both models
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (model_name, result) in enumerate(results.items()):
        if len(np.unique(result['y_test'])) > 1:
            fpr, tpr, _ = roc_curve(result['y_test'], result['probabilities'])
            auc = roc_auc_score(result['y_test'], result['probabilities'])
            axes[idx].plot(fpr, tpr, label=f'{model_name.title()} (AUC = {auc:.3f})')
            axes[idx].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[idx].set_xlabel('False Positive Rate')
            axes[idx].set_ylabel('True Positive Rate')
            axes[idx].set_title(f'ROC Curve - {model_name.title()}')
            axes[idx].legend()
            axes[idx].grid(True)
    
    plt.tight_layout()
    plt.savefig('crash_risk_roc_curves.png', dpi=300, bbox_inches='tight')
    print("Saved: crash_risk_roc_curves.png")
    plt.close()
    
    # 3. Risk score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(risk_score, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(risk_score.quantile(0.7), color='r', linestyle='--', label=f'High Risk Threshold ({risk_score.quantile(0.7):.1f})')
    plt.xlabel('Risk Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Crash Risk Scores')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('crash_risk_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: crash_risk_distribution.png")
    plt.close()
    
    # 4. Risk by engine size
    if 'engine_size_category' in analysis_df.columns:
        plt.figure(figsize=(10, 6))
        engine_risk = analysis_df.groupby('engine_size_category')['risk_score'].mean().sort_index()
        engine_risk.plot(kind='bar', color='steelblue')
        plt.ylabel('Average Risk Score')
        plt.xlabel('Engine Size Category')
        plt.title('Average Crash Risk Score by Engine Size')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('crash_risk_by_engine_size.png', dpi=300, bbox_inches='tight')
        print("Saved: crash_risk_by_engine_size.png")
        plt.close()
    
    # 5. Risk by safety score
    plt.figure(figsize=(10, 6))
    safety_risk = analysis_df.groupby('safety_score')['risk_score'].mean()
    safety_risk.plot(kind='bar', color='coral')
    plt.ylabel('Average Risk Score')
    plt.xlabel('Safety Score (Number of Protective Items)')
    plt.title('Average Crash Risk Score by Safety Score')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('crash_risk_by_safety_score.png', dpi=300, bbox_inches='tight')
    print("Saved: crash_risk_by_safety_score.png")
    plt.close()
    
    # 6. Time of day risk (if available)
    if 'time_of_day' in analysis_df.columns:
        plt.figure(figsize=(10, 6))
        time_risk = analysis_df.groupby('time_of_day')['risk_score'].mean()
        time_risk.plot(kind='bar', color='teal')
        plt.ylabel('Average Risk Score')
        plt.xlabel('Time of Day')
        plt.title('Average Crash Risk Score by Time of Day')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('crash_risk_by_time.png', dpi=300, bbox_inches='tight')
        print("Saved: crash_risk_by_time.png")
        plt.close()


def main():
    """Main function to run the crash probability prediction model."""
    print("="*60)
    print("CRASH PROBABILITY PREDICTION MODEL")
    print("="*60)
    
    # Load data
    cycle_df = load_data('CYCLE_2024.csv')
    
    # Merge with crash data if available
    # crash_filepath = 'CRASH_2024.csv'  # Uncomment and specify if available
    # df = merge_with_crash_data(cycle_df, crash_filepath)
    df = cycle_df
    
    # Preprocess features
    X, metadata, processed_data = preprocess_features(df)
    
    # Create target variables
    risk_score, high_risk = create_target_variables(df)
    
    # Train models
    models, scaler, X_train, X_test, y_test, results, feature_importance = train_models(
        X, risk_score, high_risk
    )
    
    # Analyze patterns
    analysis_df = analyze_crash_patterns(processed_data, metadata, risk_score)
    
    # Create visualizations
    create_visualizations(models, X_test, results, feature_importance, analysis_df, risk_score)
    
    # Save models
    import joblib
    joblib.dump(models['logistic_regression'], 'crash_risk_lr_model.pkl')
    joblib.dump(models['random_forest'], 'crash_risk_rf_model.pkl')
    joblib.dump(scaler, 'crash_risk_scaler.pkl')
    print("\nSaved models: crash_risk_lr_model.pkl, crash_risk_rf_model.pkl")
    print("Saved scaler: crash_risk_scaler.pkl")
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE!")
    print("="*60)
    
    return models, scaler, X, risk_score, high_risk, analysis_df


if __name__ == "__main__":
    models, scaler, X, risk_score, high_risk, analysis_df = main()

