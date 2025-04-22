import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def segment_customers(df):
    # Check for necessary columns
    required_cols = ['age_group', 'device', 'location']
    available_cols = [col for col in required_cols if col in df.columns]
    if not available_cols:
        return pd.DataFrame(columns=['segment', 'device', 'count'])

    # Encode categorical features
    encoded_df = pd.get_dummies(df[available_cols], drop_first=True)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    df['segment'] = kmeans.fit_predict(encoded_df)

    # Create summary DataFrame for chart
    grouped = df.groupby(['segment', 'device']).size().reset_index(name='count')
    return grouped
