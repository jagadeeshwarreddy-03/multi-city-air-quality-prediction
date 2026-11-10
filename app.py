import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# -------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="Air Quality Prediction Across Multiple Indian Cities Using Machine Learning",
    page_icon="🌍",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "dataset"

# -------------------------------------------------
# 2. LOAD PROJECT FILES
# -------------------------------------------------
@st.cache_data
def load_data():
    air = pd.read_csv(DATA_DIR / "air_quality_daily.csv")
    weather = pd.read_csv(DATA_DIR / "weather_daily.csv")
    cities = pd.read_csv(DATA_DIR / "cities.csv")

    if "date" in air.columns:
        air["date"] = pd.to_datetime(air["date"], errors="coerce")

    return air, weather, cities


@st.cache_resource
def load_models():
    return joblib.load(BASE_DIR / "all_models.joblib")


@st.cache_data
def load_predictions():
    return pd.read_csv(BASE_DIR / "test_predictions.csv")


try:
    air, weather, cities = load_data()
    models = load_models()
    predictions = load_predictions()
except Exception as e:
    st.error(f"Could not load a required project file: {e}")
    st.info(
        "Check that the folder 'datasets' contains air_quality_daily.csv, "
        "weather_daily.csv and cities.csv, and that all_models.joblib and "
        "test_predictions.csv are in the same folder as app.py."
    )
    st.stop()

required_prediction_columns = [
    "Actual_PM2.5",
    "Linear_Regression",
    "Decision_Tree",
    "Random_Forest"
]
missing_prediction_columns = [
    col for col in required_prediction_columns if col not in predictions.columns
]
if missing_prediction_columns:
    st.error(
        "The prediction CSV does not have the expected columns: "
        + ", ".join(missing_prediction_columns)
    )
    st.info(
        "Regenerate test_predictions.csv from the updated notebook using "
        "Actual_PM2.5, Linear_Regression, Decision_Tree and Random_Forest."
    )
    st.stop()

# -------------------------------------------------
# 3. SIDEBAR
# -------------------------------------------------
st.sidebar.title("🌍 Air Quality Dashboard")

city_list = sorted(air["city"].dropna().unique())
selected_city = st.sidebar.selectbox("Select a city", city_list)

st.sidebar.markdown("---")
st.sidebar.info("Current PM2.5 concentration analysis using machine learning")

# -------------------------------------------------
# 4. DASHBOARD HEADER
# -------------------------------------------------
st.title("🌍 Air Quality Prediction Across Multiple Indian Cities Using Machine Learning")
st.subheader("Multi-city air quality analysis across India")

st.markdown(
    "This dashboard explores air-pollution measurements across Indian cities "
    "and compares machine-learning models that estimate PM2.5 concentration "
    "from available pollutant, weather and location features. "
    "**This project estimates PM2.5 for the same observation; it does not "
    "forecast future or next-day values.**"
)

st.divider()

# -------------------------------------------------
# 5. KEY PERFORMANCE INDICATORS
# -------------------------------------------------
city_data = air[air["city"] == selected_city].copy()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Cities Covered", f"{air['city'].nunique():,}")

with col2:
    st.metric("Air Quality Records", f"{len(air):,}")

with col3:
    city_mean = city_data["pm2_5_mean"].mean()
    st.metric(
        f"{selected_city} Average PM2.5",
        f"{city_mean:.2f} µg/m³" if pd.notna(city_mean) else "N/A"
    )

with col4:
    city_max = city_data["pm2_5_mean"].max()
    st.metric(
        f"{selected_city} Highest PM2.5",
        f"{city_max:.2f} µg/m³" if pd.notna(city_max) else "N/A"
    )

st.divider()

# -------------------------------------------------
# 6. CITY-WISE PM2.5 COMPARISON
# -------------------------------------------------
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
    color="pm2_5_mean",
    color_continuous_scale="Reds",
    title="Average PM2.5 Concentration Across Indian Cities",
    labels={
        "city": "City",
        "pm2_5_mean": "Average PM2.5 (µg/m³)"
    },
    hover_data={"pm2_5_mean": ":.2f"}
)
fig_city.update_layout(
    xaxis_tickangle=-45,
    template="plotly_white",
    height=520,
    coloraxis_showscale=False
)
st.plotly_chart(fig_city, use_container_width=True)

# -------------------------------------------------
# 7. CITY-WISE PM2.5 DISTRIBUTION
# -------------------------------------------------
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
        height=480
    )
    st.plotly_chart(fig_box, use_container_width=True)
else:
    st.info("Select at least one city to view its PM2.5 distribution.")

# -------------------------------------------------
# 8. POLLUTANT CORRELATION HEATMAP
# -------------------------------------------------
st.header("🔥 Air Pollutant Correlation Heatmap")

pollutant_columns = [
    "pm2_5_mean", "pm10_mean", "no2_mean",
    "so2_mean", "co_8h_max", "o3_8h_max"
]
available_columns = [c for c in pollutant_columns if c in air.columns]

if len(available_columns) >= 2:
    correlation = air[available_columns].corr(numeric_only=True)
    fig_corr = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Correlation Between Available Air Pollutants"
    )
    fig_corr.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.warning("Not enough pollutant columns are available for correlation analysis.")

st.divider()

# -------------------------------------------------
# 9. MODEL PERFORMANCE COMPARISON
# -------------------------------------------------
st.header("🤖 Machine Learning Model Comparison")
st.caption("Final results on the held-out test dataset (15,480 observations).")

results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
    "MAE": [4.848399, 4.282588, 4.396402],
    "RMSE": [7.356955, 7.186011, 7.056477],
    "R2 Score": [0.932747, 0.935836, 0.938129]
})

st.dataframe(
    results.style.format({
        "MAE": "{:.4f}",
        "RMSE": "{:.4f}",
        "R2 Score": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

left, right = st.columns(2)

with left:
    error_data = results.melt(
        id_vars="Model",
        value_vars=["MAE", "RMSE"],
        var_name="Metric",
        value_name="Error"
    )
    fig_error = px.bar(
        error_data,
        x="Model",
        y="Error",
        color="Metric",
        barmode="group",
        text_auto=".3f",
        title="MAE and RMSE Comparison"
    )
    fig_error.update_layout(template="plotly_white")
    st.plotly_chart(fig_error, use_container_width=True)

with right:
    fig_r2 = px.bar(
        results,
        x="Model",
        y="R2 Score",
        color="Model",
        text_auto=".4f",
        title="R² Score Comparison"
    )
    fig_r2.update_layout(
        template="plotly_white",
        yaxis_range=[0, 1]
    )
    st.plotly_chart(fig_r2, use_container_width=True)

st.success(
    "Best overall model: Random Forest "
    "(test RMSE = 7.0565; test R² = 0.9381). "
    "Decision Tree has the lowest MAE (4.2826)."
)

# -------------------------------------------------
# 10. ACTUAL VS PREDICTED PM2.5
# -------------------------------------------------
st.header("📊 Actual vs Predicted PM2.5")
model_column_map = {
    "Linear Regression": "Linear_Regression",
    "Decision Tree": "Decision_Tree",
    "Random Forest": "Random_Forest"
}

selected_model = st.selectbox(
    "Choose a model to view its test predictions",
    list(model_column_map.keys()),
    index=2,
    key="prediction_model"
)

number_of_points = st.slider(
    "Number of test observations to display",
    min_value=100,
    max_value=min(2000, len(predictions)),
    value=min(500, len(predictions)),
    step=100
)

prediction_column = model_column_map[selected_model]
chart_data = predictions.head(number_of_points).copy()
chart_data["Observation"] = np.arange(1, len(chart_data) + 1)

plot_data = chart_data.melt(
    id_vars="Observation",
    value_vars=["Actual_PM2.5", prediction_column],
    var_name="Type",
    value_name="PM2.5"
)
plot_data["Type"] = plot_data["Type"].replace({
    "Actual_PM2.5": "Actual",
    prediction_column: "Predicted"
})

fig_actual = px.line(
    plot_data,
    x="Observation",
    y="PM2.5",
    color="Type",
    markers=True,
    title=f"Actual vs Predicted PM2.5 - {selected_model}",
    color_discrete_map={
        "Actual": "#2563EB",
        "Predicted": "#F97316"
    }
)
fig_actual.update_layout(
    template="plotly_white",
    xaxis_title="Test Observation",
    yaxis_title="PM2.5 (µg/m³)",
    hovermode="x unified"
)
st.plotly_chart(fig_actual, use_container_width=True)

# -------------------------------------------------
# 11. RANDOM FOREST FEATURE IMPORTANCE
# -------------------------------------------------
st.header("🌳 Random Forest Feature Importance")

try:
    rf_pipeline = models["Random Forest"]
    rf_model = rf_pipeline.named_steps["model"]

    feature_names = list(rf_pipeline.feature_names_in_)
    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(15)

    fig_importance = px.bar(
        importance_df.sort_values("Importance", ascending=True),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis",
        text_auto=".3f",
        title="Top 15 Features Influencing PM2.5 Estimation"
    )
    fig_importance.update_layout(
        template="plotly_white",
        xaxis_title="Random Forest Feature Importance",
        yaxis_title="Feature",
        coloraxis_showscale=False,
        height=600
    )
    st.plotly_chart(fig_importance, use_container_width=True)
    st.caption(
        "Feature importance shows how much the fitted Random Forest uses "
        "each feature for its splits. It does not establish causation."
    )
except Exception as e:
    st.warning(f"Could not display feature importance from the saved model: {e}")

# -------------------------------------------------
# 12. MODEL INFORMATION
# -------------------------------------------------
st.header("📌 Project Model Summary")

st.write(
    "- **Target variable:** `pm2_5_mean` (PM2.5 concentration)\n"
    "- **Task:** Regression to estimate PM2.5 for the same observation\n"
    "- **Input features:** 107\n"
    "- **Models:** Linear Regression, Decision Tree and Random Forest\n"
    "- **Best overall model:** Random Forest\n"
    "- **Test dataset:** 15,480 observations\n"
    "- **Baseline RMSE:** 28.3757\n"
    "- **Random Forest test RMSE:** 7.0565"
)

st.divider()
st.markdown(
    "**Project:** Air Quality Prediction Across Multiple Indian Cities Using Machine Learning"
)
st.caption("Developed using Python, Pandas, Scikit-learn, Plotly and Streamlit.")
