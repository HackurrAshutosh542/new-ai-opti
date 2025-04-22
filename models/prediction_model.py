import joblib
import pandas as pd

# Load model and column list
model = joblib.load("xgboost_best_model.pkl")
columns = joblib.load("xgb_model_columns.pkl")

def predict_campaign_success(df):
    # Encode features
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

    # Predict
    predictions = model.predict_proba(df_encoded)[:, 1] * 100

    # Return DataFrame with predictions
    df_result = df[['campaign_id', 'channel']].copy()
    df_result['success_probability'] = predictions
    return df_result
