import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_kpi_cards(df):
    impressions = int(df['impressions'].sum())
    clicks = int(df['clicks'].sum())
    conversions = int(df['conversions'].sum())
    spend = int(df['spend'].sum())
    ctr = df['ctr'].mean() * 100
    cvr = df['cvr'].mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("📢 Impressions", f"{impressions:,}")
    col2.metric("🖱️ Clicks", f"{clicks:,}")
    col3.metric("✅ Conversions", f"{conversions:,}")

    col4, col5, col6 = st.columns(3)
    col4.metric("💸 Spend", f"${spend:,}")
    col5.metric("🎯 Avg. CTR", f"{ctr:.2f}%")
    col6.metric("⚡ Avg. CVR", f"{cvr:.2f}%")

def create_prediction_chart(predictions_df):
    fig = px.bar(predictions_df, x='campaign_id', y='success_probability', color='channel',
                 title='📊 Campaign Success Probability', labels={'success_probability': 'Success %'})
    fig.update_layout(xaxis_title='Campaign ID', yaxis_title='Predicted Success %')
    st.plotly_chart(fig, use_container_width=True)

def create_sentiment_chart(sentiment_df):
    sentiment_counts = sentiment_df['sentiment'].value_counts(normalize=True).reset_index()
    sentiment_counts.columns = ['Sentiment', 'Percentage']
    sentiment_counts['Percentage'] *= 100
    fig = px.pie(sentiment_counts, names='Sentiment', values='Percentage',
                 title='😊 Sentiment Breakdown')
    st.plotly_chart(fig, use_container_width=True)

def create_segment_chart(segment_df):
    fig = px.sunburst(segment_df, path=['segment', 'device'], values='count',
                      title='👥 Customer Segment Breakdown')
    st.plotly_chart(fig, use_container_width=True)

def create_budget_chart(budget_df):
    fig = px.bar(budget_df, x='channel', y='optimized_budget', color='channel',
                 title='💰 Optimized Budget Allocation by Channel', text_auto=True)
    fig.update_layout(xaxis_title='Channel', yaxis_title='Budget ($)')
    st.plotly_chart(fig, use_container_width=True)
