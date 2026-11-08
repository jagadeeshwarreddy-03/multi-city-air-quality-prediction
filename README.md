# Multi-City Air Quality Prediction in India Using Machine Learning and Time-Series Analysis

## Project Overview

This project predicts next-day PM2.5 air pollution levels across 50 Indian cities using Machine Learning and time-series analysis.

It uses historical air-quality and weather information to identify pollution patterns and forecast future PM2.5 concentrations.

## Objectives

- Analyze air-quality trends across multiple Indian cities.
- Study the relationship between weather conditions and pollution.
- Apply time-series feature engineering.
- Build and compare multiple Machine Learning models.
- Develop an interactive dashboard using Streamlit.

## Dataset

**Source:** [India Daily Weather & Air Quality – Kaggle](https://www.kaggle.com/datasets/akshaykarthickms007/india-daily-weather-and-air-quality)

The project uses:
- `air_quality_daily.csv`
- `weather_daily.csv`
- `cities.csv`

The dataset covers 50 Indian cities.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Plotly
- Streamlit
- Jupyter Notebook

## Machine Learning Models

Three regression algorithms were implemented:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

## Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 6.6813 | 10.0826 | 0.8649 |
| Decision Tree | 7.0392 | 10.9133 | 0.8418 |
| Random Forest | 6.5988 | 10.0266 | 0.8664 |

**Best-performing model:** Random Forest Regressor.

## Dashboard Features

- City-wise PM2.5 comparison
- Daily and monthly pollution trends
- Seasonal analysis
- PM2.5 distribution
- Pollutant correlation heatmap
- Machine Learning model comparison
- Actual vs. predicted values
- Feature importance visualization

## Project Structure

```text
ML/
│
├── Air_Quality.ipynb
├── app.py
├── all_models.joblib
├── test_predictions.csv
├── README.md
│
└── dataset/
    ├── air_quality_daily.csv
    ├── cities.csv
    └── weather_daily.csv
```

## How to Run the Project

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit joblib
```

### 2. Download the dataset

Download the Kaggle dataset and place its three CSV files inside the `dataset` folder.

### 3. Run the dashboard

```bash
streamlit run app.py
```

### 4. Open the local dashboard

Open the localhost URL displayed in your terminal.

## Conclusion

The project demonstrates how Machine Learning and time-series feature engineering can be used to forecast next-day PM2.5 levels across multiple Indian cities.

The Random Forest model achieved an R² score of 0.8664 on the test dataset.

## Dashboard Screenshots

### 1. Dashboard Overview
![Dashboard Overview](screenshots/Screenshot%202026-11-08%20214707.png)

### 2. Daily PM2.5 Trends
![Daily Trends](screenshots/Screenshot%202026-11-08%20214201.png)

### 3. Seasonal Air Quality Analysis
![Seasonal Analysis](screenshots/Screenshot%202026-11-08%20214214.png)

### 4. City-wise PM2.5 Distribution
![PM2.5 Distribution](screenshots/Screenshot%202026-11-08%20214227.png)

### 5. Air Pollution Correlation Heatmap
![Correlation Heatmap](screenshots/Screenshot%202026-11-08%20214241.png)

### 6. Actual vs Predicted PM2.5
![Actual vs Predicted](screenshots/Screenshot%202026-11-08%20214259.png)

### 7. Machine Learning Model Comparison
![Model Comparison](screenshots/Screenshot%202026-11-08%20214314.png)

### 8. Feature Importance Analysis
![Feature Importance](screenshots/Screenshot%202026-11-08%20214325.png)


**Project:** Multi-City Air Quality Prediction in India Using Machine Learning and Time-Series Analysis
