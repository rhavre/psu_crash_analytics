"""
Predict Crash Probability for New Scenarios
Uses the trained models to predict crash risk/probability.
"""

import joblib
import pandas as pd
import numpy as np


def predict_crash_risk(engine_size, helmet_used=1, helmet_type=1, has_boots=1, 
                      has_eyeprot=1, has_longpants=1, has_longsleeves=1, 
                      has_bag=0, has_passenger=0, has_trailer=0, num_units=1,
                      passenger_helmet=0):
    """
    Predict crash risk/probability for a given scenario.
    
    Parameters:
    -----------
    engine_size : float
        Engine size in cc
    helmet_used : int
        1 if wearing helmet, 0 otherwise
    helmet_type : int
        Helmet type (0 = no helmet, 1-3 = different helmet types)
    has_boots : int
        1 if wearing boots, 0 otherwise
    has_eyeprot : int
        1 if wearing eye protection, 0 otherwise
    has_longpants : int
        1 if wearing long pants, 0 otherwise
    has_longsleeves : int
        1 if wearing long sleeves, 0 otherwise
    has_bag : int
        1 if has bag, 0 otherwise
    has_passenger : int
        1 if has passenger, 0 otherwise
    has_trailer : int
        1 if has trailer, 0 otherwise
    num_units : int
        Number of units involved (typically 1)
    passenger_helmet : int
        1 if passenger wearing helmet, 0 otherwise
    hour : int
        Hour of day (0-23)
    
    Returns:
    --------
    dict : Prediction results with risk score, probability, and recommendations
    """
    try:
        lr_model = joblib.load('crash_risk_lr_model.pkl')
        rf_model = joblib.load('crash_risk_rf_model.pkl')
        scaler = joblib.load('crash_risk_scaler.pkl')
    except FileNotFoundError:
        print("Error: Model files not found. Please run crash_probability_model.py first.")
        return None
    
    # Calculate safety score
    safety_score = has_boots + has_eyeprot + has_longpants + has_longsleeves + helmet_used
    
    # Create feature array in the same order as training
    # Feature order: ['engine_size', 'helmet_used', 'helmet_type', 'has_boots', 
    #                'has_eyeprot', 'has_longpants', 'has_longsleeves', 'has_bag',
    #                'has_passenger', 'has_trailer', 'num_units', 'passenger_helmet', 'safety_score']
    features = np.array([[
        engine_size,
        helmet_used,
        helmet_type,
        has_boots,
        has_eyeprot,
        has_longpants,
        has_longsleeves,
        has_bag,
        has_passenger,
        has_trailer,
        num_units,
        passenger_helmet,
        safety_score
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make predictions
    lr_prediction = lr_model.predict(features_scaled)[0]
    lr_probability = lr_model.predict_proba(features_scaled)[0][1]
    
    rf_prediction = rf_model.predict(features)[0]
    rf_probability = rf_model.predict_proba(features)[0][1]
    
    # Calculate manual risk score
    risk_score = 0
    if helmet_used == 0:
        risk_score += 3
    if has_eyeprot == 0:
        risk_score += 2
    if has_boots == 0:
        risk_score += 1
    if has_longpants == 0:
        risk_score += 1
    if has_longsleeves == 0:
        risk_score += 1
    if has_passenger == 1:
        risk_score += 2
    if has_trailer == 1:
        risk_score += 1
    if engine_size > 1500:
        risk_score += 1
    if num_units > 1:
        risk_score += 2
    
    # Determine risk level
    if risk_score <= 3:
        risk_level = "Low"
    elif risk_score <= 6:
        risk_level = "Medium"
    elif risk_score <= 9:
        risk_level = "High"
    else:
        risk_level = "Very High"
    
    # Generate recommendations
    recommendations = []
    if helmet_used == 0:
        recommendations.append("⚠️ Wear a helmet - this is the most critical safety measure")
    if has_eyeprot == 0:
        recommendations.append("⚠️ Wear eye protection - significantly reduces risk")
    if has_boots == 0 or has_longpants == 0 or has_longsleeves == 0:
        recommendations.append("⚠️ Wear full protective gear (boots, long pants, long sleeves)")
    if has_passenger == 1 and passenger_helmet == 0:
        recommendations.append("⚠️ Ensure passenger wears a helmet")
    if engine_size > 1500:
        recommendations.append("⚠️ Extra caution needed - larger bikes show higher risk patterns")
    if num_units > 1:
        recommendations.append("⚠️ Multiple vehicle scenarios increase risk")
    
    if not recommendations:
        recommendations.append("✅ Good safety practices observed")
    
    return {
        'risk_score': int(risk_score),
        'risk_level': risk_level,
        'high_risk_probability_lr': float(lr_probability),
        'high_risk_probability_rf': float(rf_probability),
        'predicted_high_risk_lr': bool(lr_prediction),
        'predicted_high_risk_rf': bool(rf_prediction),
        'safety_score': int(safety_score),
        'recommendations': recommendations
    }


def predict_from_dataframe(df):
    """
    Predict crash risk for multiple scenarios from a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns matching the feature names
    
    Returns:
    --------
    pandas.DataFrame : Original dataframe with added prediction columns
    """
    try:
        lr_model = joblib.load('crash_risk_lr_model.pkl')
        rf_model = joblib.load('crash_risk_rf_model.pkl')
        scaler = joblib.load('crash_risk_scaler.pkl')
    except FileNotFoundError:
        print("Error: Model files not found. Please run crash_probability_model.py first.")
        return None
    
    # Map column names - feature order from training
    feature_order = [
        'engine_size', 'helmet_used', 'helmet_type', 'has_boots',
        'has_eyeprot', 'has_longpants', 'has_longsleeves', 'has_bag',
        'has_passenger', 'has_trailer', 'num_units', 'passenger_helmet',
        'safety_score'
    ]
    
    # Prepare feature matrix
    X = df[feature_order].values
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    lr_predictions = lr_model.predict(X_scaled)
    lr_probabilities = lr_model.predict_proba(X_scaled)[:, 1]
    
    rf_predictions = rf_model.predict(X)
    rf_probabilities = rf_model.predict_proba(X)[:, 1]
    
    # Add predictions to dataframe
    df_result = df.copy()
    df_result['predicted_high_risk_lr'] = lr_predictions
    df_result['high_risk_probability_lr'] = lr_probabilities
    df_result['predicted_high_risk_rf'] = rf_predictions
    df_result['high_risk_probability_rf'] = rf_probabilities
    
    return df_result


if __name__ == "__main__":
    # Example 1: Low risk scenario
    print("Example 1: Low Risk Scenario (Full Protective Gear)")
    print("=" * 60)
    result1 = predict_crash_risk(
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
    if result1:
        print(f"Risk Score: {result1['risk_score']}")
        print(f"Risk Level: {result1['risk_level']}")
        print(f"High Risk Probability (LR): {result1['high_risk_probability_lr']:.2%}")
        print(f"High Risk Probability (RF): {result1['high_risk_probability_rf']:.2%}")
        print(f"Safety Score: {result1['safety_score']}/5")
        print("\nRecommendations:")
        for rec in result1['recommendations']:
            print(f"  {rec}")
    
    print("\n" + "=" * 60)
    print("Example 2: High Risk Scenario (No Protective Gear)")
    print("=" * 60)
    result2 = predict_crash_risk(
        engine_size=1800,
        helmet_used=0,
        helmet_type=0,
        has_boots=0,
        has_eyeprot=0,
        has_longpants=0,
        has_longsleeves=0,
        has_bag=1,
        has_passenger=1,
        has_trailer=0,
        num_units=2,
        passenger_helmet=0
    )
    if result2:
        print(f"Risk Score: {result2['risk_score']}")
        print(f"Risk Level: {result2['risk_level']}")
        print(f"High Risk Probability (LR): {result2['high_risk_probability_lr']:.2%}")
        print(f"High Risk Probability (RF): {result2['high_risk_probability_rf']:.2%}")
        print(f"Safety Score: {result2['safety_score']}/5")
        print("\nRecommendations:")
        for rec in result2['recommendations']:
            print(f"  {rec}")

