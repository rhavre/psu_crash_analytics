"""
Example script for making helmet usage predictions using the trained model.
"""

import joblib
import pandas as pd
import numpy as np

def predict_helmet_usage(engine_size, helmet_type=0, has_boots=0, has_eyeprot=0, 
                         has_longpants=0, has_longsleeves=0, has_bag=0, 
                         has_passenger=0, is_trailer=0):
    """
    Predict helmet usage for a single rider.
    
    Parameters:
    -----------
    engine_size : float
        Engine size in cc
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
    is_trailer : int
        1 if has trailer, 0 otherwise
    
    Returns:
    --------
    dict : Prediction results with probability and binary prediction
    """
    # Load model and scaler
    try:
        model = joblib.load('helmet_usage_model.pkl')
        scaler = joblib.load('helmet_usage_scaler.pkl')
    except FileNotFoundError:
        print("Error: Model files not found. Please run helmet_usage_model.py first.")
        return None
    
    # Create feature array in the same order as training
    features = np.array([[
        engine_size,
        helmet_type,
        has_boots,
        has_eyeprot,
        has_longpants,
        has_longsleeves,
        has_bag,
        has_passenger,
        is_trailer
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    return {
        'will_wear_helmet': bool(prediction),
        'probability': float(probability),
        'confidence': 'High' if probability > 0.7 or probability < 0.3 else 'Medium'
    }


def predict_from_dataframe(df):
    """
    Predict helmet usage for multiple riders from a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns matching the feature names
    
    Returns:
    --------
    pandas.DataFrame : Original dataframe with added prediction columns
    """
    # Load model and scaler
    try:
        model = joblib.load('helmet_usage_model.pkl')
        scaler = joblib.load('helmet_usage_scaler.pkl')
    except FileNotFoundError:
        print("Error: Model files not found. Please run helmet_usage_model.py first.")
        return None
    
    # Map column names (handle different naming conventions)
    feature_mapping = {
        'engine_size': 'engine_size',
        'MC_ENGINE_SIZE': 'engine_size',
        'helmet_type': 'helmet_type',
        'MC_DVR_HLMT_TYPE': 'helmet_type',
        'has_boots': 'MC_DVR_BOOTS_IND_binary',
        'MC_DVR_BOOTS_IND_binary': 'MC_DVR_BOOTS_IND_binary',
        'has_eyeprot': 'MC_DVR_EYEPRT_IND_binary',
        'MC_DVR_EYEPRT_IND_binary': 'MC_DVR_EYEPRT_IND_binary',
        'has_longpants': 'MC_DVR_LNGPNTS_IND_binary',
        'MC_DVR_LNGPNTS_IND_binary': 'MC_DVR_LNGPNTS_IND_binary',
        'has_longsleeves': 'MC_DVR_LNGSLV_IND_binary',
        'MC_DVR_LNGSLV_IND_binary': 'MC_DVR_LNGSLV_IND_binary',
        'has_bag': 'MC_BAG_IND_binary',
        'MC_BAG_IND_binary': 'MC_BAG_IND_binary',
        'has_passenger': 'has_passenger',
        'is_trailer': 'is_trailer',
        'MC_TRAIL_IND': 'is_trailer'
    }
    
    # Extract features in correct order
    feature_order = [
        'engine_size', 'helmet_type', 'MC_DVR_BOOTS_IND_binary',
        'MC_DVR_EYEPRT_IND_binary', 'MC_DVR_LNGPNTS_IND_binary',
        'MC_DVR_LNGSLV_IND_binary', 'MC_BAG_IND_binary',
        'has_passenger', 'is_trailer'
    ]
    
    # Prepare feature matrix
    X = df[feature_order].values
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Add predictions to dataframe
    df_result = df.copy()
    df_result['predicted_helmet_usage'] = predictions
    df_result['helmet_usage_probability'] = probabilities
    
    return df_result


if __name__ == "__main__":
    # Example 1: Single prediction
    print("Example 1: Single Prediction")
    print("=" * 50)
    result = predict_helmet_usage(
        engine_size=1000,
        helmet_type=1,
        has_boots=1,
        has_eyeprot=1,
        has_longpants=1,
        has_longsleeves=0,
        has_bag=0,
        has_passenger=0,
        is_trailer=0
    )
    if result:
        print(f"Will wear helmet: {result['will_wear_helmet']}")
        print(f"Probability: {result['probability']:.2%}")
        print(f"Confidence: {result['confidence']}")
    
    print("\n" + "=" * 50)
    print("Example 2: Prediction for rider with no protective gear")
    print("=" * 50)
    result2 = predict_helmet_usage(
        engine_size=500,
        helmet_type=0,
        has_boots=0,
        has_eyeprot=0,
        has_longpants=0,
        has_longsleeves=0,
        has_bag=0,
        has_passenger=0,
        is_trailer=0
    )
    if result2:
        print(f"Will wear helmet: {result2['will_wear_helmet']}")
        print(f"Probability: {result2['probability']:.2%}")
        print(f"Confidence: {result2['confidence']}")


