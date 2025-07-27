# 🏃‍♂️ Garmin Running Performance Dashboard

This project analyses outdoor running activities extracted from Garmin smartwatches through a gradient boosting model to give personalized feedback on performance. Using XGBoost regression and SHAP explainability, it evaluates training sessions based on to minimise heart rate and pace while accounting for elevation to generate a performance score and suggestions on running dynamics to improve, all displayed on a visually appealing and easy to navigate web dashboard.

<p align="center">
  <img src="https://github.com/yourusername/running-performance-analyzer/assets/demo.gif" width="600"/>
</p>

## Features

- **Performance Scoring**  
  Calculates a performance score based heart rate and pace compared to an elevation-adjusted baseline.

- **Personalized Feedback**  
  SHAP values explain which features most helped or hurt your performance in each activity.

- **Customizable Scoring Weights**  
  Adjust the weightings of heart rate and pace in performance score calculation to suit personal training goals.

- **Recent Trends**  
  Compare latest workouts to historical averages with performance deltas and metrics.

- **Interactive Visualizations**  
  Explore trends in performance and features over time using Streamlit's interactive charts.

## Implementation

### 1. Preprocessing
- Converts activity data exported from Garmin Connect
- Extracts and type convert useful features to produce a clean dataset to train on 
- Filters for outdoor running activities

### 2. Feature Engineering
- Derives new features that may contribute to performance such as `Elevation Rate`, `Active Ratio`
- Residualizes pace and heart rate against elevation to remove bias that may occur as heart rate will fluctuate depending on elevation changes.

### 3. Scoring
- Calculates z-scores of residuals
- Combines them into a single **Performance Score** using user-defined weights

### 4. Model Training
- Uses XGBoost regression to learn how features impact performance
- SHAP (SHapley Additive exPlanations) explains model predictions for each feature

## Usage

### 1. Clone the repository
```
git clone https://github.com/anton-tr3/garmin-performance-dashboard.git
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Extract activity data from Garmin Connect
Find your running activity data [here](https://connect.garmin.com/modern/activities?activityType=running) and export to csv. Replace `Activities.csv` with the exported file.

### 4. Run dashboard
```streamlit run activities_streamlit_app.py```
