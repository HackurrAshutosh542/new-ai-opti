import pandas as pd

def validate_data(df):
    required_columns = ["campaign_id", "channel", "impressions", "clicks", "conversions", "spend"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    if df.isnull().sum().sum() > 0:
        return True, "Data loaded with some missing values. Please review."
    return True, "✅ Data validated successfully."

def process_campaign_data(df):
    # Add derived metrics if they don't exist
    if 'ctr' not in df.columns:
        df['ctr'] = df['clicks'] / df['impressions']
    if 'cvr' not in df.columns:
        df['cvr'] = df['conversions'] / df['clicks']
    if 'cpc' not in df.columns:
        df['cpc'] = df['spend'] / df['clicks']
    if 'cpa' not in df.columns:
        df['cpa'] = df['spend'] / df['conversions']
    if 'roi' not in df.columns and 'revenue' in df.columns:
        df['roi'] = (df['revenue'] - df['spend']) / df['spend']
    return df
