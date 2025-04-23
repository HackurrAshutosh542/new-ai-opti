import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# 1. Insight Cards

def create_insight_cards(insights):
    st.subheader("📌 Key Insights & Recommendations")
    for insight in insights:
        st.info(f"**{insight['title']}**\n\n{insight['description']}")

# 2. Comparative Benchmarking Charts

def create_comparative_charts(df):
    if 'channel' in df.columns and 'roi' in df.columns:
        fig = px.box(df, x='channel', y='roi', color='channel',
                     title="ROI Distribution Across Channels", points="all")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# 3. Attribution Funnel Chart

def create_attribution_analysis(df):
    stages = ["Awareness", "Consideration", "Intent", "Conversion"]
    values = [100, 65, 40, 22]  # Simulated

    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textinfo="value+percent previous+percent total",
        marker={"color": ["#74b9ff", "#a29bfe", "#ffeaa7", "#55efc4"]}
    ))
    fig.update_layout(title="Campaign Funnel Attribution")
    st.plotly_chart(fig, use_container_width=True)

# 4. Customer Journey Sankey

def create_customer_journey_visualization():
    st.subheader("📍 Customer Journey Sankey Flow")

    labels = ["Ad Impression", "Site Visit", "Email Signup", "Product View", "Add to Cart", "Purchase"]
    source = [0, 1, 2, 3, 4]
    target = [1, 2, 3, 4, 5]
    values = [1000, 800, 600, 400, 300]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="blue"
        ),
        link=dict(
            source=source,
            target=target,
            value=values
        ))])

    fig.update_layout(title_text="Customer Journey Flow", font_size=13)
    st.plotly_chart(fig, use_container_width=True)

# 5. Competitive Radar

def create_competitor_analysis():
    st.subheader("📊 Competitor Performance Radar")

    metrics = ["Impressions", "Clicks", "Conversions", "Spend", "Revenue", "ROI"]
    df = pd.DataFrame({
        "Metric": metrics,
        "Your Brand": [90, 70, 60, 80, 85, 75],
        "Competitor A": [85, 60, 50, 70, 75, 70],
        "Competitor B": [70, 50, 45, 65, 60, 55]
    })

    fig = go.Figure()
    for col in df.columns[1:]:
        fig.add_trace(go.Scatterpolar(
            r=df[col],
            theta=df['Metric'],
            fill='toself',
            name=col
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Competitive Radar Chart"
    )
    st.plotly_chart(fig, use_container_width=True)
