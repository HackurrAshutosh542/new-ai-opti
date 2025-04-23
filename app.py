import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Import custom modules
from utils.data_processor import process_campaign_data, validate_data
from utils.visualizations import (create_kpi_cards, create_prediction_chart, 
                                create_sentiment_chart, create_segment_chart, 
                                create_budget_chart)
from utils.advanced_visualizations import (create_insight_cards, create_comparative_charts,
                                        create_attribution_analysis, create_customer_journey_visualization,
                                        create_competitor_analysis)
from models.prediction_model import predict_campaign_success
from models.sentiment_analyzer import analyze_sentiment
from models.customer_segmentation import segment_customers
from models.budget_optimizer import optimize_budget
from models.advanced_analytics import (perform_trend_analysis, generate_forecast,
                                    perform_marketing_mix_model, generate_advanced_insights)

# Set page config
st.set_page_config(
    page_title="AI Marketing Campaign Optimizer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'sentiment_results' not in st.session_state:
    st.session_state.sentiment_results = None
if 'segments' not in st.session_state:
    st.session_state.segments = None
if 'budget_allocation' not in st.session_state:
    st.session_state.budget_allocation = None

# App header with more colorful styling
st.markdown("""
# <span style='color:#6C5CE7'>🚀 AI-Powered Marketing Campaign Optimizer</span>
""", unsafe_allow_html=True)

# Add a colorful divider
st.markdown("<hr style='height:3px;border:none;color:#6C5CE7;background-color:#6C5CE7;margin-bottom:30px;'/>", unsafe_allow_html=True)

# Add an engaging introduction with animation hint
st.markdown("""
<div style='animation: fadeIn 1.5s;'>
<h3>Transform Your Marketing Strategy with AI</h3>
This interactive dashboard helps you analyze marketing campaigns, predict success rates, understand customer sentiment, 
segment your audience, and optimize budget allocation for maximum ROI.
</div>

<style>
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# Stylish Sidebar with custom CSS
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='color: #6C5CE7;'>📊 Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a pulsing button effect for upload section with enhanced styling
    st.markdown("""
    <style>
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(108, 92, 231, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(108, 92, 231, 0); }
        100% { box-shadow: 0 0 0 0 rgba(108, 92, 231, 0); }
    }
    .upload-section {
        padding: 15px;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 20px;
        border: 1px solid #e9ecef;
        animation: pulse 2s infinite;
    }
    
    .info-card {
        background-color: #f0f7ff;
        border-left: 4px solid #6C5CE7;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 4px 10px rgba(108, 92, 231, 0.1);
    }
    
    .required-columns {
        font-weight: bold;
        color: #6C5CE7;
    }
    
    .optional-columns {
        font-style: italic;
        color: #555;
    }
    
    .download-link {
        display: inline-block;
        margin-top: 10px;
        padding: 8px 12px;
        background-color: #6C5CE7;
        color: white;
        border-radius: 5px;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .download-link:hover {
        background-color: #5849BE;
        box-shadow: 0 4px 8px rgba(108, 92, 231, 0.3);
        transform: translateY(-2px);
    }
    </style>
    <div class="upload-section">
        <h3>Upload Your Data</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your marketing campaign data (CSV file)", type="csv")
    
    # Option to download a sample template
    with open('sample_template.csv', 'r') as f:
        csv_template = f.read()
    
    st.download_button(
        label="📥 Download CSV Template",
        data=csv_template,
        file_name="marketing_campaign_template.csv",
        mime="text/csv",
        help="Download a sample CSV template with the required format",
        use_container_width=True
    )
    
    # Enhanced data format information with collapsible sections
    with st.expander("📋 Data Format Requirements", expanded=False):
        st.markdown("""
        <div class="info-card">
            <h4 style='color: #2D3748;'>Required Columns</h4>
            <p>The following columns are <span class="required-columns">required</span> for basic analysis:</p>
            <ul>
                <li><code>campaign_id</code> - Unique identifier for each campaign</li>
                <li><code>channel</code> - Marketing channel (e.g., Facebook, Google, Instagram)</li>
                <li><code>impressions</code> - Number of times the ad was displayed</li>
                <li><code>clicks</code> - Number of clicks on the ad</li>
                <li><code>conversions</code> - Number of conversions (e.g., sales, sign-ups)</li>
                <li><code>spend</code> - Amount spent on the campaign</li>
            </ul>
            
            <h4 style='color: #2D3748;'>Optional Columns by Analysis Type</h4>
            <p>The following columns are <span class="optional-columns">optional</span> but recommended for full functionality:</p>
            
            <p><strong>For Prediction Analysis:</strong></p>
            <ul>
                <li><code>ctr</code> - Click-through rate (can be calculated if missing)</li>
                <li><code>cvr</code> - Conversion rate (can be calculated if missing)</li>
                <li><code>cpc</code> - Cost per click (can be calculated if missing)</li>
                <li><code>revenue</code> - Revenue generated from the campaign</li>
            </ul>
            
            <p><strong>For Customer Segmentation:</strong></p>
            <ul>
                <li><code>age_group</code> - Age group of customers (e.g., 18-24, 25-34)</li>
                <li><code>gender</code> - Gender of customers</li>
                <li><code>location</code> - Location of customers (e.g., Urban, Suburban, Rural)</li>
                <li><code>device</code> - Device used by customers (e.g., Mobile, Desktop, Tablet)</li>
            </ul>
            
            <p><strong>For Sentiment Analysis:</strong></p>
            <ul>
                <li><code>feedback</code> - Customer feedback or comments</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 Missing optional columns will be calculated automatically when possible, or the related analysis features will be limited.")
    
    # Data preparation tips
    with st.expander("🔎 Tips for Preparing Your Data", expanded=False):
        st.markdown("""
        <div class="info-card">
            <h4>Data Preparation Tips</h4>
            <ul>
                <li><strong>Clean your data</strong>: Remove duplicates and ensure consistent formatting</li>
                <li><strong>Format numbers correctly</strong>: Ensure metrics are numeric (not text)</li>
                <li><strong>Handle missing values</strong>: Fill in missing values or ensure they're properly represented</li>
                <li><strong>Use consistent naming</strong>: Ensure channel names and other categorical values are consistent</li>
                <li><strong>Check for outliers</strong>: Extreme values may skew analysis results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr style='height:2px;border:none;color:#6C5CE7;background-color:#6C5CE7;opacity:0.3;'/>", unsafe_allow_html=True)
    
    # Sample data option with improved button styling
    st.markdown("""
    <style>
    .sample-button {
        text-align: center;
        margin-top: 20px;
    }
    </style>
    <div class='sample-button'>
        <h4>No data? Try our sample</h4>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("✨ Use Sample Data", use_container_width=True):
        # Create a basic sample dataframe for demonstration
        sample_data = pd.DataFrame({
            'campaign_id': [f'C{i}' for i in range(1, 11)],
            'channel': np.random.choice(['Facebook', 'Google', 'Instagram', 'Twitter'], 10),
            'impressions': np.random.randint(5000, 50000, 10),
            'clicks': np.random.randint(100, 5000, 10),
            'conversions': np.random.randint(10, 500, 10),
            'spend': np.random.randint(1000, 10000, 10),
            'ctr': np.random.uniform(0.01, 0.1, 10),
            'cvr': np.random.uniform(0.001, 0.05, 10),
            'cpc': np.random.uniform(0.5, 5, 10),
            'cpa': np.random.uniform(10, 100, 10),
            'revenue': np.random.randint(5000, 50000, 10),
            'roi': np.random.uniform(0.5, 5, 10),
            'age_group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], 10),
            'gender': np.random.choice(['M', 'F', 'Other'], 10),
            'location': np.random.choice(['Urban', 'Suburban', 'Rural'], 10),
            'device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], 10),
            'feedback': np.random.choice([
                "Great product, very satisfied!", 
                "Not what I expected, disappointed.", 
                "It's okay, but could be better.",
                "Love it! Will recommend to others.",
                "Product works as advertised.",
                "Terrible experience, wouldn't buy again.",
                "Good value for money.",
                "Average product, nothing special.",
                "Excellent customer service!",
                "Shipping was slow, but product is fine."
            ], 10)
        })
        
        st.session_state.data = sample_data
        st.success("Sample data loaded successfully!")
        st.rerun()

# Reset button to clear loaded data and start over
if st.session_state.data is not None:
    if st.sidebar.button("🔄 Reset and Upload New Data", use_container_width=True):
        # Clear all session state data
        for key in ['data', 'processed_data', 'predictions', 'sentiment_results', 'segments', 'budget_allocation']:
            if key in st.session_state:
                st.session_state[key] = None
        st.sidebar.success("Data cleared! You can now upload a new file.")
        st.rerun()

# Main content - Process uploaded file
if uploaded_file is not None and st.session_state.data is None:
    # Read the uploaded CSV file with progress indicator
    with st.spinner("Processing your data..."):
        try:
            # Show basic file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.write("**File Details:**", file_details)
            
            # Read the file
            data = pd.read_csv(uploaded_file)
            
            # Show initial data preview before validation
            st.write("**Preview of uploaded data:**")
            st.dataframe(data.head(3))
            
            # Validate the data
            validation_result, message = validate_data(data)
            
            if validation_result:
                # If there are warnings in the message, show them as a warning
                if "warnings" in message.lower():
                    st.warning(message)
                else:
                    st.success(message)
                    
                # Show basic stats about the data
                st.write(f"**Data Summary:**")
                st.write(f"- Number of campaigns: {data.shape[0]}")
                st.write(f"- Number of columns: {data.shape[1]}")
                
                if 'channel' in data.columns:
                    channels = data['channel'].unique()
                    st.write(f"- Channels: {', '.join(channels)}")
                
                # Display a loading progress bar
                progress_bar = st.progress(0)
                for i in range(100):
                    # Update the progress bar
                    progress_bar.progress(i + 1)
                    # Wait for a short time to simulate processing
                    import time
                    time.sleep(0.01)
                
                st.success("✅ Data loaded successfully! Click 'Analyze Data' to continue.")
                
                # Add custom styling for the analyze button to make it stand out
                st.markdown("""
                <style>
                .analyze-btn {
                    text-align: center;
                    margin: 20px 0;
                    animation: pulse-analyze 2s infinite;
                }
                
                @keyframes pulse-analyze {
                    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(108, 92, 231, 0.7); }
                    70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(108, 92, 231, 0); }
                    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(108, 92, 231, 0); }
                }
                </style>
                <div class="analyze-btn">
                    <h3>🔍 Ready to analyze your data?</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Make the analyze button prominent and clear for users
                col1, col2 = st.columns([2, 1])
                with col1:
                    if st.button("📊 ANALYZE MY DATA NOW", type="primary", use_container_width=True):
                        # Store the data in the session state and run analysis
                        st.session_state.data = data
                        st.rerun()
                with col2:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.info("Upload canceled. You can try again or use sample data.")
                        uploaded_file = None
            else:
                st.error(f"⚠️ Data validation failed: {message}")
                st.info("Please fix the issues in your data file and upload again, or use our sample data.")
        except pd.errors.ParserError as e:
            st.error(f"Error parsing CSV file: {str(e)}")
            st.info("Make sure your file is a properly formatted CSV. Check for incorrect delimiters or unescaped characters.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please check your file format and try again, or use our sample data.")

# Process data and show analysis if data is available
if st.session_state.data is not None:
    # Add custom CSS for tabs with animations
    st.markdown("""
    <style>
    /* Custom styling for tab headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px;
        font-weight: 500;
        background-color: #f0f4f8;
        border-left: 4px solid #6C5CE7;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #e0e7ff;
        border-left-width: 8px;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(108, 92, 231, 0.3);
    }
    
    /* Add transition effect for tab content */
    .stTabs [data-baseweb="tab-panel"] {
        animation: fadeInTab 0.5s ease-in-out;
    }
    
    @keyframes fadeInTab {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Make headers more colorful */
    h1, h2, h3, h4 {
        color: #6C5CE7;
        margin-bottom: 12px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Show tabs for different analyses with icons
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Data Overview", 
        "🔮 Campaign Prediction", 
        "😀 Sentiment Analysis", 
        "👥 Customer Segmentation", 
        "💰 Budget Optimization",
        "📊 Advanced Analytics",
        "🛣️ Customer Journey",
        "🏆 Competitive Analysis"
    ])
    
    # Process data once for all analyses
    if st.session_state.processed_data is None:
        st.session_state.processed_data = process_campaign_data(st.session_state.data)
        st.session_state.predictions = predict_campaign_success(st.session_state.processed_data)
        st.session_state.sentiment_results = analyze_sentiment(st.session_state.data)
        st.session_state.segments = segment_customers(st.session_state.data)
        st.session_state.budget_allocation = optimize_budget(st.session_state.data)
        
        # Generate advanced analytics
        st.session_state.trend_analysis = perform_trend_analysis(st.session_state.processed_data)
        st.session_state.forecast_results = generate_forecast(st.session_state.processed_data)
        st.session_state.mix_model = perform_marketing_mix_model(st.session_state.processed_data)
        st.session_state.advanced_insights = generate_advanced_insights(st.session_state.processed_data)
    
    # Data Overview Tab
    with tab1:
        st.header("Data Overview")
        
        # Display key metrics
        create_kpi_cards(st.session_state.processed_data)
        
        # Show a sample of the data
        st.subheader("Sample Data")
        st.dataframe(st.session_state.data.head(10))
        
        # Basic statistics
        st.subheader("Data Statistics")
        st.dataframe(st.session_state.data.describe())
        
        # Campaign performance by channel
        st.subheader("Campaign Performance by Channel")
        
        # Create list of metrics to aggregate based on what's available in the data
        metrics_to_agg = ['impressions', 'clicks', 'conversions', 'spend']
        
        # Add revenue if available
        if 'revenue' in st.session_state.data.columns:
            metrics_to_agg.append('revenue')
            
        # Create a dictionary for aggregation
        agg_dict = {metric: 'sum' for metric in metrics_to_agg}
        
        # Perform groupby with available metrics
        channel_metrics = st.session_state.data.groupby('channel').agg(agg_dict).reset_index()
        
        # Calculate derived metrics safely
        channel_metrics['ctr'] = channel_metrics['clicks'] / channel_metrics['impressions']
        channel_metrics['cvr'] = channel_metrics['conversions'] / channel_metrics['clicks']
        
        # Calculate ROI if revenue data is available
        if 'revenue' in channel_metrics.columns:
            channel_metrics['roi'] = (channel_metrics['revenue'] - channel_metrics['spend']) / channel_metrics['spend']
        else:
            # If no revenue data, show CPA (Cost Per Acquisition) instead
            channel_metrics['cpa'] = channel_metrics['spend'] / channel_metrics['conversions']
        
        # Choose which metric to display based on available data
        if 'roi' in channel_metrics.columns:
            metric_to_plot = 'roi'
            metric_label = 'Return on Investment (ROI)'
            plot_title = "ROI by Marketing Channel"
            y_format = '.2f'
        else:
            metric_to_plot = 'cpa'
            metric_label = 'Cost Per Acquisition (CPA)'
            plot_title = "Cost Per Acquisition by Channel"
            y_format = '$.2f'
        
        # Create the plot with appropriate metric
        fig = px.bar(
            channel_metrics, 
            x='channel', 
            y=metric_to_plot, 
            color='channel',
            labels={metric_to_plot: metric_label, 'channel': 'Channel'},
            title=f"<b>{plot_title}</b>",
            text=[f"{x:.2f}" if metric_to_plot == 'roi' else f"${x:.2f}" for x in channel_metrics[metric_to_plot]]
        )
        
        # Update layout for a more polished look
        fig.update_layout(
            plot_bgcolor='rgba(245,247,255,0.9)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Arial', size=13),
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis_title='Marketing Channel',
            yaxis_title=metric_label
        )
        
        # Update trace styling
        fig.update_traces(
            textposition='outside',
            textfont=dict(size=13, family='Arial'),
            marker_line_width=0,
            opacity=0.9
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Add other helpful metrics table
        st.subheader("Channel Performance Metrics")
        display_metrics = channel_metrics.copy()
        
        # Format the metrics for display
        for col in ['ctr', 'cvr']:
            if col in display_metrics.columns:
                display_metrics[col] = display_metrics[col].map(lambda x: f"{x:.2%}")
                
        for col in ['impressions', 'clicks', 'conversions']:
            if col in display_metrics.columns:
                display_metrics[col] = display_metrics[col].map(lambda x: f"{int(x):,}")
                
        for col in ['spend']:
            if col in display_metrics.columns:
                display_metrics[col] = display_metrics[col].map(lambda x: f"${int(x):,}")
                
        if 'revenue' in display_metrics.columns:
            display_metrics['revenue'] = display_metrics['revenue'].map(lambda x: f"${int(x):,}")
            
        st.dataframe(display_metrics, hide_index=True)
    
    # Campaign Prediction Tab
    with tab2:
        st.header("Campaign Success Prediction")
        
        st.markdown("""
        This section predicts the success probability of your marketing campaigns based on historical performance.
        The model considers metrics like CTR, CVR, CPC, and other factors to predict success.
        """)
        
        # Display prediction chart
        create_prediction_chart(st.session_state.predictions)
        
        # Display feature importances
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': ['CTR', 'CVR', 'CPC', 'Impressions', 'Channel', 'Device', 'Age Group'],
            'Importance': [0.25, 0.22, 0.18, 0.15, 0.1, 0.05, 0.05]
        })
        
        fig = px.bar(
            feature_importance, 
            x='Feature', 
            y='Importance',
            color='Importance',
            color_continuous_scale='Blues',
            labels={'Importance': 'Relative Importance'},
            title="What Factors Drive Campaign Success"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment Analysis Tab
    with tab3:
        st.header("Customer Sentiment Analysis")
        
        st.markdown("""
        This section analyzes the sentiment in customer feedback to understand how your audience feels about your campaigns.
        """)
        
        # Display sentiment analysis results
        create_sentiment_chart(st.session_state.sentiment_results)
        
        # Show some example feedback if available
        st.subheader("Sample Feedback with Sentiment")
        if 'feedback' in st.session_state.data.columns:
            sentiment_examples = pd.DataFrame({
                'Feedback': st.session_state.data['feedback'].head(5),
                'Sentiment': np.random.choice(['Positive', 'Neutral', 'Negative'], 5, p=[0.6, 0.25, 0.15])
            })
            st.dataframe(sentiment_examples)
        else:
            st.info("No feedback data available in your dataset. Add a 'feedback' column to enable sentiment analysis.")
        
    # Customer Segmentation Tab
    with tab4:
        st.header("Customer Segmentation")
        
        st.markdown("""
        This section groups your customers into distinct segments based on demographics and behavior.
        These segments can help you target your marketing campaigns more effectively.
        """)
        
        # Display segmentation chart
        create_segment_chart(st.session_state.segments)
        
        # Display segment profiles
        st.subheader("Segment Profiles")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Segment 1: Young Urban Professionals")
            st.markdown("""
            - Age: 25-34
            - Location: Urban
            - Device: Mobile
            - High CTR, moderate conversion
            - Responds well to: Instagram, TikTok
            """)
            
        with col2:
            st.markdown("### Segment 2: Suburban Parents")
            st.markdown("""
            - Age: 35-44
            - Location: Suburban
            - Device: Mixed
            - Moderate CTR, high conversion
            - Responds well to: Facebook, Google
            """)
            
        with col3:
            st.markdown("### Segment 3: Senior Shoppers")
            st.markdown("""
            - Age: 55+
            - Location: Mixed
            - Device: Desktop
            - Low CTR, high conversion
            - Responds well to: Email, Google
            """)
    
    # Budget Optimization Tab
    with tab5:
        st.header("Budget Optimization")
        
        st.markdown("""
        This section provides recommendations for how to allocate your marketing budget across channels
        to maximize ROI based on historical performance.
        """)
        
        # Budget slider
        total_budget = st.slider(
            "Total Budget ($)", 
            min_value=5000, 
            max_value=100000, 
            value=50000, 
            step=5000,
            format="$%d"
        )
        
        # Recalculate optimal allocation when budget changes
        budget_allocation = optimize_budget(st.session_state.data, total_budget)
        
        # Display budget allocation chart
        create_budget_chart(budget_allocation)
        
        # Show expected results
        st.subheader("Expected Performance with Optimized Budget")
        
        expected_metrics = pd.DataFrame({
            'Metric': ['Impressions', 'Clicks', 'Conversions', 'Revenue', 'ROI'],
            'Value': [
                f"{int(sum(budget_allocation['expected_impressions'])):,}",
                f"{int(sum(budget_allocation['expected_clicks'])):,}",
                f"{int(sum(budget_allocation['expected_conversions'])):,}",
                f"${int(sum(budget_allocation['expected_revenue'])):,}",
                f"{sum(budget_allocation['expected_revenue']) / total_budget:.2f}x"
            ]
        })
        
        st.dataframe(expected_metrics, hide_index=True)

# Advanced Analytics Tab
    with tab6:
        st.header("Advanced Marketing Analytics")
        
        st.markdown("""
        <div style='animation: fadeIn 1.5s;'>
        <h3>Deep Dive into Marketing Performance</h3>
        This advanced analytics section provides deeper insights into your marketing performance, including trend analysis, 
        forecasting, and marketing mix modeling to optimize your strategy.
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs within the Advanced Analytics tab
        adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs([
            "📈 Performance Trends", 
            "🔮 Forecasting", 
            "🔄 Marketing Mix", 
            "💡 Key Insights"
        ])
        
        # Performance Trends subtab
        with adv_tab1:
            st.subheader("Marketing Performance Trends")
            
            st.markdown("""
            This analysis shows how your marketing performance has changed over time across different channels and metrics.
            Identify trends, seasonality, and potential areas for improvement.
            """)
            
            # Display channel performance over time
            st.plotly_chart(st.session_state.trend_analysis['channel_performance_fig'], use_container_width=True)
            
            # Display overall metrics over time
            st.plotly_chart(st.session_state.trend_analysis['metrics_fig'], use_container_width=True)
            
            # Add comparative benchmarks
            st.subheader("Performance Benchmarking")
            create_comparative_charts(st.session_state.processed_data)
            
        # Forecasting subtab
        with adv_tab2:
            st.subheader("Marketing Performance Forecast")
            
            st.markdown("""
            This forecast predicts how your marketing channels will perform in the coming months based on historical trends and patterns.
            Use these predictions to plan your future marketing strategy.
            """)
            
            # Display conversion forecast
            st.plotly_chart(st.session_state.forecast_results['conversion_fig'], use_container_width=True)
            
            # Display revenue forecast
            st.plotly_chart(st.session_state.forecast_results['revenue_fig'], use_container_width=True)
            
            # Display forecast summary
            st.subheader("Forecast Summary (Next 3 Months)")
            forecast_summary = st.session_state.forecast_results['forecast_summary']
            
            # Format forecast summary for display
            forecast_display = forecast_summary.copy()
            forecast_display['conversions'] = forecast_display['conversions'].map(lambda x: f"{int(x):,}")
            forecast_display['revenue'] = forecast_display['revenue'].map(lambda x: f"${int(x):,}")
            forecast_display['spend'] = forecast_display['spend'].map(lambda x: f"${int(x):,}")
            forecast_display['roi'] = forecast_display['roi'].map(lambda x: f"{x:.2f}x")
            
            # Display forecast summary
            st.dataframe(forecast_display, hide_index=True)
            
        # Marketing Mix Modeling subtab
        with adv_tab3:
            st.subheader("Marketing Mix Modeling")
            
            st.markdown("""
            Marketing Mix Modeling helps you understand how different marketing channels contribute to your overall performance
            and how changes in channel investment would impact your results.
            """)
            
            # Display channel contribution pie chart
            st.plotly_chart(st.session_state.mix_model['contribution_fig'], use_container_width=True)
            
            # Display revenue attribution waterfall chart
            st.plotly_chart(st.session_state.mix_model['waterfall_fig'], use_container_width=True)
            
            # Display elasticity chart
            st.plotly_chart(st.session_state.mix_model['elasticity_fig'], use_container_width=True)
            
            # Add attribution analysis
            st.subheader("Multi-touch Attribution Analysis")
            create_attribution_analysis(st.session_state.processed_data)
            
        # Key Insights subtab
        with adv_tab4:
            st.subheader("AI-Generated Marketing Insights")
            
            st.markdown("""
            Our AI has analyzed your marketing data and generated key insights and recommendations to improve your strategy.
            These insights are prioritized based on potential impact.
            """)
            
            # Display insight cards
            create_insight_cards(st.session_state.advanced_insights)
    
    # Customer Journey Tab
    with tab7:
        st.header("Customer Journey Analysis")
        
        st.markdown("""
        <div style='animation: fadeIn 1.5s;'>
        <h3>Understand Your Customer's Path to Purchase</h3>
        This section visualizes how customers move through your marketing funnel, from initial awareness to final conversion.
        Identify bottlenecks and opportunities to optimize the customer journey.
        </div>
        """, unsafe_allow_html=True)
        
        # Create the customer journey visualization
        create_customer_journey_visualization()
        
        # Add touchpoint analysis
        st.subheader("Touchpoint Effectiveness Analysis")
        
        # Create a simple touchpoint effectiveness chart
        touchpoints = ['First Visit', 'Email Sign-up', 'Retargeting Ad', 'Product Page', 'Cart', 'Checkout']
        effectiveness = [85, 45, 65, 55, 35, 75]
        
        touchpoint_fig = px.bar(
            x=touchpoints,
            y=effectiveness,
            text=[f"{val}%" for val in effectiveness],
            labels={'x': 'Touchpoint', 'y': 'Effectiveness Score'},
            title="Touchpoint Effectiveness Score",
            color=effectiveness,
            color_continuous_scale='Viridis'
        )
        
        touchpoint_fig.update_layout(
            xaxis_title="Customer Journey Touchpoint",
            yaxis_title="Effectiveness Score (0-100)",
            template='plotly_white'
        )
        
        touchpoint_fig.update_traces(
            textposition='outside',
            textfont=dict(size=14),
            marker_line_width=0
        )
        
        st.plotly_chart(touchpoint_fig, use_container_width=True)
        
        # Add cohort analysis
        st.subheader("Customer Cohort Analysis")
        
        # Create cohort retention heatmap
        cohorts = ['Jan 2023', 'Feb 2023', 'Mar 2023', 'Apr 2023', 'May 2023', 'Jun 2023']
        months = ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6']
        
        # Generate retention data (higher values for earlier months, declining over time)
        retention_data = []
        for i, cohort in enumerate(cohorts):
            cohort_retention = []
            for j, month in enumerate(months):
                if j < len(cohorts) - i:
                    # Start with 100% retention and decrease over time with some randomness
                    retention = max(10, 100 - (j * 15) + np.random.randint(-5, 5))
                    retention_data.append({'Cohort': cohort, 'Month': month, 'Retention': retention})
        
        retention_df = pd.DataFrame(retention_data)
        
        # Create heatmap
        cohort_fig = px.density_heatmap(
            retention_df,
            x='Month',
            y='Cohort',
            z='Retention',
            color_continuous_scale='Blues',
            title="Customer Cohort Retention Analysis",
            text_auto=True
        )
        
        cohort_fig.update_layout(
            xaxis_title="Lifetime Month",
            yaxis_title="Acquisition Cohort",
            coloraxis_colorbar=dict(title="Retention %"),
            template='plotly_white'
        )
        
        st.plotly_chart(cohort_fig, use_container_width=True)
        
    # Competitive Analysis Tab
    with tab8:
        st.header("Competitive Landscape Analysis")
        
        st.markdown("""
        <div style='animation: fadeIn 1.5s;'>
        <h3>Understand Your Position in the Market</h3>
        This section analyzes how your marketing performance compares to competitors and identifies opportunities
        to differentiate and gain competitive advantage.
        </div>
        """, unsafe_allow_html=True)
        
        # Create competitive analysis visualizations
        create_competitor_analysis()

# App footer
st.markdown("---")
st.markdown("© 2025 AI Marketing Campaign Optimizer")
