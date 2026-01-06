import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Aadhaar Analytics", layout="wide")

# ---------------- LANGUAGE TOGGLE ----------------
language = st.sidebar.radio("🌐 Language", ["English", "हिंदी"])

TEXT = {
    "title": {
        "English": "Aadhaar Enrolment & Update Intelligence System",
        "हिंदी": "आधार नामांकन और अद्यतन विश्लेषण प्रणाली"
    }
}

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/aadhaar.csv")

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Aadhaar Analytics")
state = st.sidebar.selectbox("Select State", df["State"].unique())

filtered = df[df["State"] == state]

# ---------------- TITLE ----------------
st.title(TEXT["title"][language])

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trends",
    "🧠 Societal Insights",
    "🚨 Anomalies",
    "🔮 Demand Prediction",
    "🗺 Heatmap"
])

# ---------------- TAB 1: TRENDS ----------------
with tab1:
    st.subheader("Enrolment & Update Trends")

    trend = filtered.groupby(["Year", "Month"]).sum(numeric_only=True).reset_index()

    fig, ax = plt.subplots()
    ax.plot(trend.index, trend["Enrollments"], label="Enrollments")
    ax.plot(trend.index, trend["Updates"], label="Updates")
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")

    st.pyplot(fig)

    st.success("📌 Identifies peak demand periods for better planning.")

# ---------------- TAB 2: SOCIETAL INSIGHTS ----------------
with tab2:
    st.subheader("District-wise Participation")

    district_data = filtered.groupby("District").sum(numeric_only=True)

    st.bar_chart(district_data[["Enrollments", "Updates"]])

    top_district = district_data["Updates"].idxmax()
    st.info(
        f"🧠 {top_district} shows highest Aadhaar update activity — likely due to migration or service demand."
    )

# ---------------- TAB 3: ANOMALY DETECTION + RISK SCORE ----------------
with tab3:
    st.subheader("Anomaly Detection")

    filtered = filtered.copy()
    filtered["Z_score"] = (
        filtered["Enrollments"] - filtered["Enrollments"].mean()
    ) / filtered["Enrollments"].std()

    anomalies = filtered[np.abs(filtered["Z_score"]) > 2]

    if not anomalies.empty:
        st.warning("🚨 Anomalies Detected")
        st.dataframe(anomalies)
    else:
        st.success("✅ No major anomalies detected.")

    st.markdown("### ⚠️ Center Overload Risk Score")

    filtered["RiskScore"] = (
        (filtered["Updates"] / filtered["Updates"].max()) * 0.6 +
        (np.abs(filtered["Z_score"]) / filtered["Z_score"].abs().max()) * 0.4
    )

    high_risk = filtered[filtered["RiskScore"] > 0.7]

    if not high_risk.empty:
        st.error("🚨 High Risk Districts")
        st.dataframe(high_risk[["District", "RiskScore"]])
    else:
        st.success("✅ No critical overload risks.")

# ---------------- TAB 4: DEMAND PREDICTION ----------------
with tab4:
    st.subheader("Future Aadhaar Update Demand")

    filtered["TimeIndex"] = range(len(filtered))

    X = filtered[["TimeIndex"]]
    y = filtered["Updates"]

    model = LinearRegression()
    model.fit(X, y)

    future_index = np.array([[len(filtered) + i] for i in range(3)])
    predictions = model.predict(future_index)

    for i, val in enumerate(predictions, 1):
        st.write(f"🔮 Month {i}: {int(val)} predicted updates")

    st.success("📌 Enables proactive staffing & infrastructure planning.")

# ---------------- TAB 5: STATE-WISE HEATMAP ----------------
with tab5:
    st.subheader("State-wise Aadhaar Activity Heatmap")

    state_data = df.groupby("State")[["Enrollments", "Updates"]].sum().reset_index()

    fig = px.density_heatmap(
        state_data,
        x="State",
        y="Updates",
        color_continuous_scale="Viridis",
        title="Aadhaar Update Intensity by State"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("📌 High intensity states require priority resource allocation.")


# ---------------- FINAL RECOMMENDATIONS ----------------
st.markdown("---")
st.subheader("📌 Policy Recommendations")

st.write("""
• Deploy mobile Aadhaar units in high-risk districts  
• Increase staffing during predicted peak months  
• Use anomaly alerts for early intervention  
• Adopt data-driven governance for better citizen experience  
""")
