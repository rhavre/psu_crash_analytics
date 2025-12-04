"""
Helmet Usage Prediction Model
Predicts whether a motorcycle rider will wear a helmet based on various conditions.
This is a classification task using logistic regression.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_cycle_data(filepath='CYCLE_2024.csv'):
    """Load the motorcycle cycle data."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded {len(df)} records")
    return df


def merge_with_crash_data(cycle_df, crash_filepath=None):
    """
    Merge cycle data with main crash dataset if available.
    The main crash dataset should contain: weather, age, time of day, municipality, road type, etc.
    """
    if crash_filepath:
        try:
            crash_df = pd.read_csv(crash_filepath, low_memory=False)
            # Merge on CRN (Crash Report Number)
            merged_df = cycle_df.merge(crash_df, on='CRN', how='left')
            print(f"Merged with crash data: {len(merged_df)} records")
            return merged_df
        except FileNotFoundError:
            print(f"Warning: Crash data file {crash_filepath} not found. Using cycle data only.")
            return cycle_df
    return cycle_df


def preprocess_data(df):
    """
    Preprocess the data for modeling.
    Creates target variable and feature engineering.
    """
    print("\nPreprocessing data...")
    
    # Create a copy to avoid modifying original
    data = df.copy()
    
    # Create target variable: Helmet Usage (1 = Yes, 0 = No/Unknown)
    # MC_DVR_HLMTON_IND: Y = Yes, N = No, U = Unknown, empty = No
    data['helmet_used'] = data['MC_DVR_HLMTON_IND'].apply(
        lambda x: 1 if str(x).upper() == 'Y' else 0
    )
    
    print(f"Helmet usage distribution:")
    print(data['helmet_used'].value_counts())
    print(f"Helmet usage rate: {data['helmet_used'].mean():.2%}")
    
    # Feature Engineering
    features = {}
    
    # Engine size (convert to numeric, handle missing values)
    data['MC_ENGINE_SIZE'] = pd.to_numeric(data['MC_ENGINE_SIZE'], errors='coerce')
    # Replace 99999 (unknown) and other invalid values with median
    valid_engine_sizes = data[data['MC_ENGINE_SIZE'] < 99999]['MC_ENGINE_SIZE']
    median_engine = valid_engine_sizes.median()
    data['engine_size'] = data['MC_ENGINE_SIZE'].fillna(median_engine)
    data.loc[data['engine_size'] >= 99999, 'engine_size'] = median_engine
    features['engine_size'] = data['engine_size']
    
    # Create engine size categories
    data['engine_size_category'] = pd.cut(
        data['engine_size'],
        bins=[0, 500, 1000, 1500, float('inf')],
        labels=['Small (0-500)', 'Medium (500-1000)', 'Large (1000-1500)', 'Very Large (1500+)']
    )
    
    # Helmet type (0 = No helmet, 1-3 = Different helmet types, 9 = Unknown)
    data['helmet_type'] = pd.to_numeric(data['MC_DVR_HLMT_TYPE'], errors='coerce')
    data['helmet_type'] = data['helmet_type'].fillna(0)
    data.loc[data['helmet_type'] == 9, 'helmet_type'] = 0  # Treat unknown as no helmet
    features['helmet_type'] = data['helmet_type']
    
    # Other protective gear indicators (convert Y/N/U to binary)
    gear_cols = [
        'MC_DVR_BOOTS_IND', 'MC_DVR_EYEPRT_IND', 'MC_DVR_LNGPNTS_IND', 
        'MC_DVR_LNGSLV_IND', 'MC_BAG_IND'
    ]
    
    for col in gear_cols:
        if col in data.columns:
            data[f'{col}_binary'] = data[col].apply(
                lambda x: 1 if str(x).upper() == 'Y' else 0
            )
            features[f'{col}_binary'] = data[f'{col}_binary']
    
    # Passenger indicator
    if 'MC_PASSNGR_IND' in data.columns:
        data['has_passenger'] = data['MC_PASSNGR_IND'].apply(
            lambda x: 1 if str(x).upper() == 'Y' else 0
        )
        features['has_passenger'] = data['has_passenger']
    
    # Trail indicator
    if 'MC_TRAIL_IND' in data.columns:
        data['is_trailer'] = data['MC_TRAIL_IND'].apply(
            lambda x: 1 if str(x).upper() == 'Y' else 0
        )
        features['is_trailer'] = data['is_trailer']
    
    # If merged with crash data, add those features
    # Time of day (if available)
    time_cols = ['CRASH_TIME', 'TIME_OF_DAY', 'HOUR']
    for col in time_cols:
        if col in data.columns:
            # Extract hour if it's a datetime or time string
            if data[col].dtype == 'object':
                try:
                    data['hour'] = pd.to_datetime(data[col], errors='coerce').dt.hour
                except:
                    # Try to extract hour from string format
                    data['hour'] = pd.to_numeric(data[col].str[:2], errors='coerce')
            else:
                data['hour'] = data[col]
            
            # Create time categories
            data['time_of_day'] = pd.cut(
                data['hour'].fillna(12),
                bins=[0, 6, 12, 18, 24],
                labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)']
            )
            features['hour'] = data['hour'].fillna(12)
            break
    
    # Municipality/County (if available)
    location_cols = ['MUNICIPALITY', 'COUNTY', 'COUNTY_NAME', 'MUNI_NAME']
    for col in location_cols:
        if col in data.columns:
            # Encode location
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
    
    # Weather (if available)
    weather_cols = ['WEATHER', 'WEATHER_COND', 'WEATHER_CONDITION']
    for col in weather_cols:
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
            # Create age categories
            data['age_category'] = pd.cut(
                data['age'],
                bins=[0, 25, 35, 50, 100],
                labels=['Young (0-25)', 'Adult (25-35)', 'Middle (35-50)', 'Senior (50+)']
            )
            features['age'] = data['age']
            break
    
    # Create feature matrix
    feature_df = pd.DataFrame(features)
    
    # Remove rows with missing target
    valid_idx = ~data['helmet_used'].isna()
    feature_df = feature_df[valid_idx]
    target = data.loc[valid_idx, 'helmet_used']
    
    # Add metadata for analysis
    metadata_cols = ['CRN', 'engine_size_category', 'helmet_used']
    if 'time_of_day' in data.columns:
        metadata_cols.append('time_of_day')
    if any(f'{col}_labels' in data.columns for col in location_cols):
        for col in location_cols:
            if f'{col}_labels' in data.columns:
                metadata_cols.append(f'{col}_labels')
                break
    if 'age_category' in data.columns:
        metadata_cols.append('age_category')
    
    metadata = data.loc[valid_idx, [col for col in metadata_cols if col in data.columns]]
    
    print(f"\nFeatures created: {list(feature_df.columns)}")
    print(f"Total features: {len(feature_df.columns)}")
    print(f"Valid records: {len(feature_df)}")
    
    return feature_df, target, metadata, data


def train_model(X, y, test_size=0.2, random_state=42):
    """Train logistic regression model."""
    print("\nTraining model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    model = LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # ROC AUC
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC AUC Score: {auc:.4f}")
    
    # Feature importance
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE (Coefficients)")
    print("="*50)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_[0],
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    return model, scaler, X_train, X_test, y_train, y_test, y_pred, y_pred_proba


def analyze_patterns(data, metadata):
    """Analyze patterns in helmet usage."""
    print("\n" + "="*50)
    print("PATTERN ANALYSIS")
    print("="*50)
    
    analysis_df = pd.concat([data[['helmet_used']], metadata], axis=1)
    
    # Engine size patterns
    if 'engine_size_category' in analysis_df.columns:
        print("\n1. Helmet Usage by Engine Size:")
        engine_pattern = analysis_df.groupby('engine_size_category')['helmet_used'].agg(['mean', 'count']).reset_index()
        engine_pattern = engine_pattern.rename(columns={
            'engine_size_category': 'Engine Size Category',
            'mean': 'Helmet Usage Rate',
            'count': 'Count'
        })
        print(engine_pattern.to_string(index=False))
    
    # Time of day patterns
    if 'time_of_day' in analysis_df.columns:
        print("\n2. Helmet Usage by Time of Day:")
        time_pattern = analysis_df.groupby('time_of_day')['helmet_used'].agg(['mean', 'count']).reset_index()
        time_pattern = time_pattern.rename(columns={
            'time_of_day': 'Time of Day',
            'mean': 'Helmet Usage Rate',
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
        print(f"\n3. Helmet Usage by Location ({location_col}):")
        location_pattern = analysis_df.groupby(location_col)['helmet_used'].agg(['mean', 'count']).reset_index()
        location_pattern = location_pattern.rename(columns={
            location_col: 'Location',
            'mean': 'Helmet Usage Rate',
            'count': 'Count'
        })
        location_pattern = location_pattern.sort_values('Helmet Usage Rate')
        print(location_pattern.head(20).to_string(index=False))  # Top 20 locations
    
    # Age patterns
    if 'age_category' in analysis_df.columns:
        print("\n4. Helmet Usage by Age Category:")
        age_pattern = analysis_df.groupby('age_category')['helmet_used'].agg(['mean', 'count']).reset_index()
        age_pattern = age_pattern.rename(columns={
            'age_category': 'Age Category',
            'mean': 'Helmet Usage Rate',
            'count': 'Count'
        })
        print(age_pattern.to_string(index=False))
    
    return analysis_df


def create_visualizations(model, X_train, y_test, y_pred_proba, analysis_df):
    """Create visualizations for the model."""
    print("\nCreating visualizations...")
    
    # 1. Feature importance plot
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['coefficient'])
    plt.xlabel('Coefficient Value')
    plt.title('Feature Importance (Logistic Regression Coefficients)')
    plt.tight_layout()
    plt.savefig('helmet_model_feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: helmet_model_feature_importance.png")
    plt.close()
    
    # 2. ROC Curve
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Helmet Usage Prediction')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('helmet_model_roc_curve.png', dpi=300, bbox_inches='tight')
        print("Saved: helmet_model_roc_curve.png")
        plt.close()
    
    # 3. Engine size vs helmet usage
    if 'engine_size_category' in analysis_df.columns:
        plt.figure(figsize=(10, 6))
        engine_usage = analysis_df.groupby('engine_size_category')['helmet_used'].mean().sort_index()
        engine_usage.plot(kind='bar', color='steelblue')
        plt.ylabel('Helmet Usage Rate')
        plt.xlabel('Engine Size Category')
        plt.title('Helmet Usage by Engine Size')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('helmet_usage_by_engine_size.png', dpi=300, bbox_inches='tight')
        print("Saved: helmet_usage_by_engine_size.png")
        plt.close()
    
    # 4. Time of day vs helmet usage
    if 'time_of_day' in analysis_df.columns:
        plt.figure(figsize=(10, 6))
        time_usage = analysis_df.groupby('time_of_day')['helmet_used'].mean()
        time_usage.plot(kind='bar', color='coral')
        plt.ylabel('Helmet Usage Rate')
        plt.xlabel('Time of Day')
        plt.title('Helmet Usage by Time of Day')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('helmet_usage_by_time.png', dpi=300, bbox_inches='tight')
        print("Saved: helmet_usage_by_time.png")
        plt.close()
    
    # 5. Location patterns (top 15)
    location_col = None
    for col in analysis_df.columns:
        if 'labels' in col and ('COUNTY' in col or 'MUNICIPALITY' in col or 'MUNI' in col):
            location_col = col
            break
    
    if location_col:
        location_usage = analysis_df.groupby(location_col)['helmet_used'].agg(['mean', 'count'])
        location_usage = location_usage[location_usage['count'] >= 10]  # Filter by minimum count
        location_usage = location_usage.sort_values('mean').head(15)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(location_usage)), location_usage['mean'], color='teal')
        plt.yticks(range(len(location_usage)), location_usage.index)
        plt.xlabel('Helmet Usage Rate')
        plt.title('Top 15 Locations with Lowest Helmet Usage (min 10 crashes)')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('helmet_usage_by_location.png', dpi=300, bbox_inches='tight')
        print("Saved: helmet_usage_by_location.png")
        plt.close()


def main():
    """Main function to run the helmet usage prediction model."""
    print("="*60)
    print("HELMET USAGE PREDICTION MODEL")
    print("="*60)
    
    # Load data
    cycle_df = load_cycle_data('CYCLE_2024.csv')
    
    # Merge with crash data if available (uncomment and specify path if you have crash data)
    # crash_filepath = 'CRASH_2024.csv'  # Update with your crash data file path
    # df = merge_with_crash_data(cycle_df, crash_filepath)
    df = cycle_df  # Using cycle data only for now
    
    # Preprocess
    X, y, metadata, processed_data = preprocess_data(df)
    
    # Train model
    model, scaler, X_train, X_test, y_train, y_test, y_pred, y_pred_proba = train_model(X, y)
    
    # Analyze patterns
    analysis_df = analyze_patterns(processed_data, metadata)
    
    # Create visualizations
    create_visualizations(model, X_train, y_test, y_pred_proba, analysis_df)
    
    # Save model
    import joblib
    joblib.dump(model, 'helmet_usage_model.pkl')
    joblib.dump(scaler, 'helmet_usage_scaler.pkl')
    print("\nSaved model: helmet_usage_model.pkl")
    print("Saved scaler: helmet_usage_scaler.pkl")
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE!")
    print("="*60)
    
    return model, scaler, X, y, analysis_df


if __name__ == "__main__":
    model, scaler, X, y, analysis_df = main()

