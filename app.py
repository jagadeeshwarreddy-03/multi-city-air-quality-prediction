
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="India Air Quality Prediction",
    page_icon="🌍",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "dataset"

# 2. LOAD DATA AND MODELS
@st.cache_data
def load_data():
    air = pd.read_csv(DATA_DIR / "air_quality_daily.csv")
    weather = pd.read_csv(DATA_DIR / "weather_daily.csv")
    cities = pd.read_csv(DATA_DIR / "cities.csv")

    air["date"] = pd.to_datetime(air["date"])

    return air, weather, cities


@st.cache_resource
def load_models():
    return joblib.load(BASE_DIR / "all_models.joblib")


try:
    air, weather, cities = load_data()
    models = load_models()
except Exception as e:
    st.error(f"Error loading project files: {e}")
    st.stop()

# 3. SIDEBAR
st.sidebar.title("🌍 Air Quality Dashboard")

city_list = sorted(air["city"].dropna().unique())

selected_city = st.sidebar.selectbox(
    "Select a city",
    city_list
)

st.sidebar.markdown("---")
st.sidebar.info("Next-day PM2.5 prediction across Indian cities")

# 4. DASHBOARD HEADER
st.title("🌍 Multi-City Air Quality Prediction")
st.subheader("India | Machine Learning and Time-Series Analysis")

st.markdown(
    "An interactive dashboard for analyzing air pollution "
    "and predicting next-day PM2.5 levels."
)

st.divider()

# 5. KEY PERFORMANCE INDICATORS
city_data = air[air["city"] == selected_city].copy()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Cities", len(city_list))

with col2:
    st.metric("Total Records", f"{len(air):,}")

with col3:
    st.metric(
        "Average PM2.5",
        f"{city_data['pm2_5_mean'].mean():.2f} µg/m³"
    )

with col4:
    st.metric(
        "Highest PM2.5",
        f"{city_data['pm2_5_mean'].max():.2f} µg/m³"
    )

st.divider()


# 6. INTERACTIVE CITY-WISE POLLUTION
import plotly.express as px

st.header("🌆 City-wise PM2.5 Comparison")

city_avg = (
    air.groupby("city", as_index=False)["pm2_5_mean"]
    .mean()
    .sort_values("pm2_5_mean", ascending=False)
)

fig_city = px.bar(
    city_avg,
    x="city",
    y="pm2_5_mean",
    color="city",
    title="Average PM2.5 Levels Across 50 Indian Cities",
    labels={
        "city": "City",
        "pm2_5_mean": "Average PM2.5 (µg/m³)"
    },
    hover_data={"pm2_5_mean": ":.2f"}
)

fig_city.update_layout(
    xaxis_tickangle=-45,
    showlegend=False,
    template="plotly_white",
    height=550
)

st.plotly_chart(fig_city, use_container_width=True)


# 7. MULTI-CITY DAILY TREND
st.header("📈 Daily PM2.5 Trends by City")

selected_cities = st.multiselect(
    "Select cities to compare",
    city_list,
    default=city_list[:5]
)

if selected_cities:
    trend_data = air[air["city"].isin(selected_cities)]

    fig_trend = px.line(
        trend_data,
        x="date",
        y="pm2_5_mean",
        color="city",
        title="Daily PM2.5 Comparison",
        labels={
            "date": "Date",
            "pm2_5_mean": "PM2.5 (µg/m³)",
            "city": "City"
        }
    )

    fig_trend.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=550
    )

    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("Select at least one city to display the trend.")

# 8. SEASONAL PM2.5 ANALYSIS
st.header("🌦️ Seasonal Air Quality Analysis")

season_data = air.copy()

season_data["month"] = season_data["date"].dt.month

season_map = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Summer",
    4: "Summer",
    5: "Summer",
    6: "Monsoon",
    7: "Monsoon",
    8: "Monsoon",
    9: "Monsoon",
    10: "Post-Monsoon",
    11: "Post-Monsoon"
}

season_data["Season"] = season_data["month"].map(season_map)

season_avg = (
    season_data.groupby(
        ["city", "Season"], as_index=False
    )["pm2_5_mean"].mean()
)

season_order = [
    "Winter", "Summer", "Monsoon", "Post-Monsoon"
]

fig_season = px.bar(
    season_avg,
    x="Season",
    y="pm2_5_mean",
    color="city",
    barmode="group",
    category_orders={"Season": season_order},
    title="Seasonal PM2.5 Comparison Across Indian Cities",
    labels={
        "pm2_5_mean": "Average PM2.5 (µg/m³)",
        "Season": "Season",
        "city": "City"
    }
)

fig_season.update_layout(
    template="plotly_white",
    height=600
)

st.plotly_chart(fig_season, use_container_width=True)

# 9. CITY-WISE PM2.5 DISTRIBUTION
st.header("📦 City-wise PM2.5 Distribution")

box_cities = st.multiselect(
    "Choose cities for distribution comparison",
    city_list,
    default=city_list[:5],
    key="box_cities"
)

if box_cities:

    box_data = air[air["city"].isin(box_cities)]

    fig_box = px.box(
        box_data,
        x="city",
        y="pm2_5_mean",
        color="city",
        title="PM2.5 Distribution Across Selected Cities",
        labels={
            "city": "City",
            "pm2_5_mean": "PM2.5 (µg/m³)"
        }
    )

    fig_box.update_layout(
        template="plotly_white",
        showlegend=False,
        height=500
    )

    st.plotly_chart(fig_box, use_container_width=True)
else:
    st.info("Select cities to view their PM2.5 distributions.")

# 10. CORRELATION HEATMAP
st.header("🔥 Air Pollution Correlation Heatmap")

pollutant_columns = [
    "pm2_5_mean",
    "pm10_mean",
    "no2_mean",
    "so2_mean",
    "co_mean",
    "o3_mean"
]

available_columns = [
    col for col in pollutant_columns
    if col in air.columns
]

if len(available_columns) >= 2:

    correlation = air[available_columns].corr()

    fig_corr = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Correlation Between Air Pollutants"
    )

    fig_corr.update_layout(
        template="plotly_white",
        height=550
    )

    st.plotly_chart(fig_corr, use_container_width=True)

else:
    st.warning("Not enough pollutant columns available for correlation analysis.")

# 11. MODEL SELECTION
st.divider()
st.header("🔍 Available Trained Models")

model_name = st.selectbox(
    "Choose a trained model",
    list(models.keys())
)

st.success(f"Loaded model: {model_name}")

st.caption(
    "The model comparison values above are from the held-out "
    "test dataset. Interactive predictions will be added next."
)
# 12. FOOTER
st.divider()

st.markdown(
    "**Project:** Multi-City Air Quality Prediction in India "
    "Using Machine Learning and Time-Series Analysis"
)

st.caption(
    "Developed using Python, Pandas, Scikit-learn and Streamlit."
)

# ACTUAL VS PREDICTED PM2.5

st.header("Actual vs Predicted PM2.5")

# Load prediction results
predictions = pd.read_csv(BASE_DIR / "test_predictions.csv")

# Select model
selected_model = st.selectbox(
    "Choose a model",
    ["Linear Regression", "Decision Tree", "Random Forest"],
    key="prediction_model"
)

# Select number of observations
number_of_points = st.slider(
    "Number of test observations",
    min_value=100,
    max_value=2000,
    value=500,
    step=100
)

chart_data = predictions.head(number_of_points).copy()
chart_data["Observation"] = range(1, len(chart_data) + 1)

# Prepare data for Plotly
plot_data = chart_data.melt(
    id_vars="Observation",
    value_vars=["Actual", selected_model],
    var_name="Type",
    value_name="PM2.5"
)

fig = px.line(
    plot_data,
    x="Observation",
    y="PM2.5",
    color="Type",
    markers=True,
    title=f"Actual vs {selected_model} Predictions",
    color_discrete_map={
        "Actual": "#2563EB",
        selected_model: "#F97316"
    }
)

fig.update_layout(
    template="plotly_white",
    xaxis_title="Test Observation",
    yaxis_title="PM2.5 (µg/m³)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================
# MODEL PERFORMANCE COMPARISON
# =====================================

st.header("Machine Learning Model Comparison")

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "MAE": [6.6813, 7.0392, 6.5988],
    "RMSE": [10.0826, 10.9133, 10.0266],
    "R2 Score": [0.8649, 0.8418, 0.8664]
})

# Display results
st.dataframe(
    results.style.format({
        "MAE": "{:.4f}",
        "RMSE": "{:.4f}",
        "R2 Score": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

# Compare MAE and RMSE
error_data = results.melt(
    id_vars="Model",
    value_vars=["MAE", "RMSE"],
    var_name="Metric",
    value_name="Error"
)

fig = px.bar(
    error_data,
    x="Model",
    y="Error",
    color="Metric",
    barmode="group",
    text_auto=".3f",
    title="MAE and RMSE Comparison"
)

fig.update_layout(template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# Compare R2
fig2 = px.bar(
    results,
    x="Model",
    y="R2 Score",
    color="Model",
    text_auto=".4f",
    title="R² Score Comparison"
)

fig2.update_layout(
    template="plotly_white",
    yaxis_range=[0, 1]
)

st.plotly_chart(fig2, use_container_width=True)

# =====================================
# FEATURE IMPORTANCE ANALYSIS
# =====================================

st.header("Feature Importance Analysis")

importance_df = pd.DataFrame({
    "Feature": [
        "PM2.5 (Current Day)",
        "PM2.5 Lag 1",
        "PM2.5 Rolling 30 Days",
        "Precipitation",
        "Wind Speed",
        "PM2.5 Lag 3",
        "PM2.5 Lag 7",
        "SO2",
        "PM10",
        "PM2.5 Rolling 7 Days"
    ],
    "Importance": [
        18.03459,
        0.66850,
        0.62284,
        0.11153,
        0.05214,
        0.02842,
        0.02610,
        0.02232,
        0.02186,
        0.02054
    ]
})

importance_df = importance_df.sort_values(
    "Importance", ascending=True
)

fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale="Viridis",
    text_auto=".3f",
    title="Top 10 Features Influencing PM2.5 Prediction"
)

fig.update_layout(
    template="plotly_white",
    xaxis_title="Importance (Increase in MAE)",
    yaxis_title="Feature",
    coloraxis_showscale=False
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    "Importance represents the increase in MAE when a feature "
    "is shuffled. Higher values indicate greater predictive influence."
)