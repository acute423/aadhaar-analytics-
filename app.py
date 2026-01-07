import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from prophet import Prophet

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="UIDAI Sentinel", page_icon="🇮🇳", layout="wide")

# --- CSS STYLING (To make it look like a Govt Portal) ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    h1, h2, h3 { color: #2c3e50; }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER: GENERATE DUMMY DATA (If no CSV provided) ---
@st.cache_data
def load_data():
    # Simulate 365 days of data across 3 districts
    dates = pd.date_range(start='2024-01-01', periods=365)
    data = []
    
    # District A: Normal operations
    for d in dates:
        data.append(['District_A', d, np.random.randint(50, 150)])
        
    # District B: Has a massive fraud spike in May
    for d in dates:
        count = np.random.randint(40, 120)
        if d.month == 5 and d.day > 10 and d.day < 20: 
            count = np.random.randint(600, 800) # ANOMALY
        data.append(['District_B', d, count])
        
    df = pd.DataFrame(data, columns=['District', 'Date', 'Count'])
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Load data
df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("UIDAI Analytics")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/c/cf/Aadhaar_Logo.svg", width=150)
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Anomaly Detection", "Resource Planner"])

# --- PAGE 1: OVERVIEW ---
if page == "Dashboard Overview":
    st.title("🇮🇳 AadhaarPulse: Strategic Insights Dashboard")
    st.markdown("### Real-time monitoring of Enrolment & Update Trends")
    
    # Top Level Metrics
    col1, col2, col3 = st.columns(3)
    total_enrolments = df['Count'].sum()
    avg_daily = int(df['Count'].mean())
    
    col1.metric("Total Transactions (YTD)", f"{total_enrolments:,}")
    col2.metric("Avg Daily Footfall", f"{avg_daily}")
    col3.metric("Districts Monitored", df['District'].nunique())
    
    # Simple Time Series Plot
    st.subheader("National Trend Overview")
    fig = px.line(df.groupby('Date')['Count'].sum().reset_index(), x='Date', y='Count', title="Total Daily Transactions")
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: ANOMALY DETECTION ---
elif page == "Anomaly Detection":
    st.title("🚨 Fraud & Anomaly Scout")
    st.info("Detects unusual spikes in Aadhaar updates using Isolation Forest Algorithm.")
    
    # User inputs
    selected_district = st.selectbox("Select District to Analyze", df['District'].unique())
    contamination = st.slider("Sensitivity (Contamination Level)", 0.01, 0.1, 0.05)
    
    # Filter Data
    d_df = df[df['District'] == selected_district].copy()
    
    # Run Model
    model = IsolationForest(contamination=contamination, random_state=42)
    d_df['Anomaly'] = model.fit_predict(d_df[['Count']])
    d_df['Type'] = d_df['Anomaly'].map({1: 'Normal', -1: 'Critical Anomaly'})
    
    # Visualization
    fig = px.scatter(d_df, x='Date', y='Count', color='Type', 
                     color_discrete_map={'Normal': 'blue', 'Critical Anomaly': 'red'},
                     title=f"Anomaly Detection for {selected_district}",
                     size='Count', hover_data=['Count'])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data Table for Anomalies
    anomalies = d_df[d_df['Type'] == 'Critical Anomaly']
    if not anomalies.empty:
        st.error(f"⚠️ Found {len(anomalies)} suspicious events in {selected_district}!")
        st.dataframe(anomalies[['Date', 'Count']].style.format({"Date": lambda t: t.strftime("%Y-%m-%d")}))
    else:
        st.success("No anomalies detected in this timeframe.")

# --- PAGE 3: RESOURCE PLANNER (FORECASTING) ---
elif page == "Resource Planner":
    st.title("📅 Future Demand Forecasting")
    st.markdown("Predicts footfall for the next 30 days to optimize staff allocation.")
    
    district = st.selectbox("Select District for Forecasting", df['District'].unique())
    
    if st.button("Generate Forecast"):
        with st.spinner("Training Prophet Model..."):
            # Prepare data for Prophet
            p_df = df[df['District'] == district][['Date', 'Count']].rename(columns={'Date': 'ds', 'Count': 'y'})
            
            # Train Model
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            m.fit(p_df)
            
            # Predict
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)
            
            # Plot
            st.subheader(f"30-Day Forecast for {district}")
            
            # Custom Plotly Chart for Forecast
            fig = go.Figure()
            # Historical Data
            fig.add_trace(go.Scatter(x=p_df['ds'], y=p_df['y'], name='Actual', mode='markers', marker=dict(color='gray', opacity=0.5)))
            # Forecast Line
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prediction', line=dict(color='green')))
            # Confidence Interval
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)', name='Confidence Interval'))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation Logic
            peak_day = forecast.iloc[-30:]['yhat'].max()
            st.success(f"**Insight:** The predicted peak footfall is **{int(peak_day)}** people. Ensure full staffing on that day.")
