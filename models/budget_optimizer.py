import pandas as pd

def optimize_budget(df, total_budget=50000):
    if 'channel' not in df.columns or 'roi' not in df.columns:
        return pd.DataFrame(columns=['channel', 'optimized_budget'])

    channel_roi = df.groupby('channel')['roi'].mean()
    roi_sum = channel_roi.sum()

    allocation = (channel_roi / roi_sum) * total_budget
    result = pd.DataFrame({
        'channel': allocation.index,
        'optimized_budget': allocation.values
    })

    # Optional: simulate expected performance
    result['expected_impressions'] = result['optimized_budget'] * 10
    result['expected_clicks'] = result['optimized_budget'] * 0.5
    result['expected_conversions'] = result['optimized_budget'] * 0.05
    result['expected_revenue'] = result['optimized_budget'] * 1.8

    return result
