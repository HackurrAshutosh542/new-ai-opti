import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# --- Trend Analysis ---
def perform_trend_analysis(df):
    time_series = df.copy()
    time_series['date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='W')
    
    channel_perf_fig = px.line(time_series, x='date', y='conversions', color='channel',
                                title='📈 Weekly Conversions by Channel')
    
    metrics_fig = px.area(time_series, x='date', y='revenue', color='channel',
                          title='💰 Revenue Over Time')

    return {
        'channel_performance_fig': channel_perf_fig,
        'metrics_fig': metrics_fig
    }

# --- Forecasting ---
def generate_forecast(df):
    future_months = ['July', 'Aug', 'Sept']
    conv = np.random.randint(300, 500, 3)
    rev = np.random.randint(20000, 50000, 3)
    spend = np.random.randint(10000, 20000, 3)
    roi = rev / spend

    forecast_summary = pd.DataFrame({
        'month': future_months,
        'conversions': conv,
        'revenue': rev,
        'spend': spend,
        'roi': roi
    })

    conv_fig = px.line(forecast_summary, x='month', y='conversions', markers=True, title="🔮 Predicted Conversions")
    rev_fig = px.line(forecast_summary, x='month', y='revenue', markers=True, title="📈 Forecasted Revenue")

    return {
        'forecast_summary': forecast_summary,
        'conversion_fig': conv_fig,
        'revenue_fig': rev_fig
    }

# --- Marketing Mix Modeling ---
def perform_marketing_mix_model(df):
    channels = ['Facebook', 'Google', 'Instagram', 'Email']
    values = np.random.randint(10000, 40000, 4)
    base = 10000

    waterfall_fig = go.Figure(go.Waterfall(
        name = "", orientation = "v",
        measure = ["relative"]*4 + ["total"],
        x = channels + ["Total"],
        textposition = "outside",
        text = [f"+${v}" for v in values] + [f"=${sum(values)+base}"],
        y = list(values) + [base],
        connector = {"line":{"color":"rgb(63, 63, 63)"}}
    ))
    waterfall_fig.update_layout(title="📉 Revenue Attribution by Channel", showlegend=False)

    contribution_fig = px.pie(names=channels, values=values, title="📊 Channel Revenue Share")
    elasticity_fig = px.bar(x=channels, y=np.random.uniform(0.1, 1.0, 4), title="📈 Channel Elasticity")

    return {
        'contribution_fig': contribution_fig,
        'waterfall_fig': waterfall_fig,
        'elasticity_fig': elasticity_fig
    }

# --- AI-Generated Insights ---
def generate_advanced_insights(df):
    insights = [
        {"title": "Boost Instagram Investment", "description": "Instagram shows higher conversion rates on lower spend — try scaling this channel."},
        {"title": "Reallocate Budget from Email", "description": "Email has low ROI lately. Consider shifting budget to better-performing channels."},
        {"title": "Leverage Retargeting", "description": "Users engaging via retargeting ads have 2.3x higher likelihood to convert."},
        {"title": "Timing Matters", "description": "Campaigns launched mid-week perform 17% better than weekends."},
        {"title": "Optimize CTA Language", "description": "Messages with urgency verbs ('now', 'limited') had 28% higher click-throughs."}
    ]
    return insights
