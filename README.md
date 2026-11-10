# 🌍 Air Quality Prediction Across Multiple Indian Cities Using Machine Learning

## 📌 Project Overview

Air pollution is a major environmental concern affecting public health and quality of life. This project uses Machine Learning techniques to estimate PM2.5 concentration across multiple Indian cities based on available air quality and weather-related parameters.

The project analyzes air quality data, performs exploratory data analysis, trains multiple regression models, compares their performance, and presents the results through an interactive Streamlit dashboard.

## 🎯 Project Objectives

- Analyze air quality patterns across multiple Indian cities.
- Estimate PM2.5 concentration using Machine Learning.
- Identify important factors influencing PM2.5 estimation.
- Compare the performance of different regression algorithms.
- Develop an interactive dashboard for visualizing air quality information.

## 📂 Dataset Description

**Dataset:** India Daily Weather and Air Quality

**Source:** [Kaggle – India Daily Weather and Air Quality](https://www.kaggle.com/datasets/akshaykarthickms007/india-daily-weather-and-air-quality)

Dataset details:

- Total records: 77,400
- Number of cities: 50
- Input features: 107
- Target variable: `pm2_5_mean`

The dataset contains air quality measurements and weather-related parameters collected across multiple Indian cities.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Plotly
- Jupyter Notebook / Google Colab
- Git and GitHub

## ⚙️ Project Workflow

1. Dataset collection
2. Data integration
3. Data preprocessing
4. Exploratory Data Analysis (EDA)
5. Feature selection
6. Feature scaling
7. Train-test splitting
8. Machine Learning model training
9. Model evaluation and comparison
10. Interactive dashboard development

## 🧹 Data Preprocessing

The following preprocessing activities were performed:

- Integrated air quality and weather datasets.
- Examined missing values and duplicate records.
- Prepared relevant input features.
- Excluded target-related variables that could cause data leakage.
- Applied feature scaling using StandardScaler.
- Divided the dataset into training and testing sets.

## 📊 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand air quality patterns and relationships between variables.

Key visualizations include:

- Average PM2.5 concentration across Indian cities.
- PM2.5 distribution across selected cities.
- Correlation heatmap of available air pollutants and environmental variables.
- Random Forest feature importance analysis.

## 🤖 Machine Learning Models

Three regression algorithms were implemented:

### 1. Linear Regression
Used as a basic regression model to estimate PM2.5 concentration.

### 2. Decision Tree Regressor
Used to identify nonlinear relationships between input features and PM2.5 concentration.

### 3. Random Forest Regressor
An ensemble learning algorithm that combines multiple decision trees to improve prediction performance.

A Dummy Regressor was also used as a baseline for comparison.

## 📈 Model Evaluation

The models were evaluated using the following metrics:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Coefficient of Determination

### Model Performance Comparison

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 4.8484 | 7.3570 | 0.9327 |
| Decision Tree | 4.2826 | 7.1860 | 0.9358 |
| Random Forest | 4.3964 | 7.0565 | 0.9381 |

**Best Overall Model:** Random Forest Regressor

The Random Forest model achieved the lowest RMSE and highest R² score among the three models. The Decision Tree achieved the lowest MAE.

## 📊 Dashboard Features

The project includes an interactive dashboard developed using **Streamlit**.

### 1. City Selection
- Select a city from the available list of 50 Indian cities.
- Explore air quality information for the selected city.

### 2. Average PM2.5 Concentration
- Compares average PM2.5 concentration across Indian cities.
- Helps identify cities with relatively higher and lower PM2.5 levels.

### 3. PM2.5 Distribution
- Visualizes the distribution of PM2.5 values.
- Helps understand variations in particulate matter concentration.

### 4. Pollutant Correlation Heatmap
- Displays correlations between available air pollutants and environmental variables.
- Helps identify relationships between different measurements.

### 5. Machine Learning Model Comparison
- Compares Linear Regression, Decision Tree, and Random Forest.
- Displays MAE, RMSE, and R² evaluation metrics.

### 6. Actual vs Predicted PM2.5
- Compares actual PM2.5 values with Random Forest model predictions.
- Helps evaluate the model's estimation performance.

### 7. Feature Importance Analysis
- Displays the top 15 important features influencing PM2.5 estimation.
- Uses Random Forest feature importance scores.

### 8. Interactive Visualizations
- Interactive charts and graphs.
- City-wise air quality exploration.
- Visual comparison of model results.

## 📸 Dashboard Screenshots

### 1. Main Dashboard Overview

![Dashboard Overview](screenshots/Screenshot%202026-11-10%20171444.png)

### 2. Average PM2.5 Concentration Across Indian Cities

![Average PM2.5](screenshots/average-pm25-concentration-across-indian.png)

### 3. PM2.5 Distribution Across Selected Cities

![PM2.5 Distribution](screenshots/pm25-distribution-across-selected-cities.png)

### 4. Pollutant Correlation Heatmap

![Pollutant Correlation](screenshots/correlation-between-available-air-pollut.png)

### 5. Actual vs Predicted PM2.5 – Random Forest

![Actual vs Predicted](screenshots/actual-vs-predicted-pm25-random-forest.png)

### 6. R² Score Comparison

![R2 Score Comparison](screenshots/r%C2%B2-score-comparison.png)

### 7. Top 15 Features Influencing PM2.5 Estimation

![Feature Importance](screenshots/top-15-features-influencing-pm25-estimat.png)
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

**Project:** Air Quality Prediction Across Multiple Indian Cities Using Machine Learning
