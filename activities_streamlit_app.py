import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import streamlit as st
from streamlit_shap import st_shap
st.set_page_config(layout="wide")

def time_to_seconds(time_str):
    """Convert time string in MM:SS or HH:MM:SS format to seconds."""

    time_list = time_str.split(':')

    # Get rid of leading zeros for each component
    time_list = [x.lstrip('0') for x in time_list]
    # Replace empty strings with '0' in the time components
    time_list = [x if x != '' else '0' for x in time_list]
    # Finally convert to integers
    time_list = [int(float(x)) for x in time_list]
    
    # If the length of the list is 2, we have MM:SS format
    if len(time_list) == 2:  
        return time_list[0] * 60 + time_list[1]
    
    # Otherwise it will be HH:MM:SS
    else:
        return time_list[0] * 3600 + time_list[1] * 60 + time_list[2]
    
def z_score(value, mean, std):
    return (value - mean) / std

def feature_feedback(shap_value, feature, value, threshold):
    if shap_value < 0:  # This feature hurt performance

        difference = round(abs(value - threshold), 2)

        if value > threshold:
            feedback = f"- ❌ Try reducing {feature} by ~{(difference):.2f}."
        else:
            feedback = f"- ❌ Try increasing {feature} by ~{(difference):.2f}."
    else:
        feedback = f"- ✅ Good job keeping {feature} at {value:.2f}!"

    return feedback

# ! Import Activities, drop unnecessary columns and format data types
# Import activities as csv
initial_activities_df = pd.read_csv('activities.csv')

# List of columns to drop
cut_cols = ['Favorite', 'Training Stress Score®', 'Min Temp', 'Max Temp', 'Decompression', 'Best Lap Time', 'Number of Laps', 
            'Min Resp', 'Max Resp', 'Best Pace', 'Total Descent', 'Avg GCT Balance', 'Avg GAP',
            'Max Power', 'Steps', 'Min Elevation', 'Max Elevation', 'Calories', 'Max HR', 'Max Run Cadence', 'Avg Power', 'Aerobic TE']

# Get a list of possible names for outdoor and indoor running activities
outdoor = ['Running', 'Track Running']
indoor = ['Treadmill Running']

# Create new dataframes for outdoor, indoor, and base activities
outdoor_df = initial_activities_df[initial_activities_df['Activity Type'].isin(outdoor)]

# Drop columns from activity dataframe
cut_activities_df = outdoor_df.drop(cut_cols, axis = 1)

# Print the number of activities for each type
print(f'{len(outdoor_df)} outdoor activities logged')

# Define the columns which contain time in MM:SS or HH:MM:SS format
time_cols = ['Time', 'Avg Pace', 'Moving Time', 'Elapsed Time']
# Define the columns which will be converted to integers and floats
int_cols = ['Avg HR', 'Avg Run Cadence', 'Total Ascent', 'Avg Ground Contact Time', 'Normalized Power® (NP®)']
float_cols = ['Distance', 'Avg Stride Length', 'Avg Vertical Ratio', 'Avg Vertical Oscillation', 'Avg Resp']

# Create a new dataframe with numerical time columns and dropped the final categorical column
clean_df = cut_activities_df.drop(columns = 'Activity Type')

# Remove any rows with a null value in the row
clean_df = clean_df.replace('--', np.nan)
clean_df = clean_df.dropna(axis = 0, ignore_index = True)

# Convert time columns to seconds, and int/float columns to their respective types
clean_df[time_cols] = clean_df[time_cols].map(time_to_seconds)
clean_df[int_cols] = clean_df[int_cols].map(int)
clean_df[float_cols] = clean_df[float_cols].map(float)

# ! Feature Engineering
# Create active ratio feature and drop moving and elapsed columns
clean_df['Active Ratio'] = clean_df['Moving Time'] / clean_df['Elapsed Time']
clean_df = clean_df.drop(columns=['Moving Time', 'Elapsed Time'])

clean_df['Elevation Rate'] = clean_df['Total Ascent'] / clean_df['Distance']
clean_df = clean_df.drop(columns=['Total Ascent'])

# ! Creating Performance Score (y variable)
# We use residualisation to calculate the performance score 
# of pace and heart rate without the influence of elevation rate
X = clean_df[['Elevation Rate']]

y_pace = clean_df['Avg Pace']
model = LinearRegression()
model.fit(X, y_pace)
clean_df['Pace Residuals'] = y_pace - model.predict(X)

y_hr = clean_df['Avg HR']
model = LinearRegression()
model.fit(X, y_hr)
clean_df['HR Residuals'] = y_hr - model.predict(X)

# Calculate z scores for pace and heart rate residuals
clean_df['z Pace Residuals'] = clean_df['Pace Residuals'].apply(lambda x: z_score(x, clean_df['Pace Residuals'].mean(), clean_df['Pace Residuals'].std()))
clean_df['z HR Residuals'] = clean_df['HR Residuals'].apply(lambda x: z_score(x, clean_df['HR Residuals'].mean(), clean_df['HR Residuals'].std()))

# Settings panel to change weighings of pace and heart rate on the performance score
settings_popover = st.popover('Settings')
settings_popover.write('## Performance Score Settings')
settings_popover.write('The weighting of pace and heart rate on performance score can be adjusted for more customised and targeted feedback.')
HR_weight = settings_popover.slider('HR Weight', 0.0, 1.0, 1.0)
pace_weight = settings_popover.slider('Pace Weight', 0.0, 1.0, 1.0)

# Calculate performance score as a combination of the z scores of pace and heart rate residuals using given weights
clean_df['Score'] = -1 * (HR_weight * clean_df['z HR Residuals'] + pace_weight * clean_df['z Pace Residuals'])

# Drop residual columns since they are not needed anymore
clean_df = clean_df.drop(columns=['Pace Residuals', 'HR Residuals', 'z Pace Residuals', 'z HR Residuals'])
# Also drop the Avg HR and Avg Pace columns since they are directly correlated to the performance score
clean_df = clean_df.drop(columns=['Avg HR', 'Avg Pace'])

# ! Train XGBoost model
activities_df = clean_df.copy()
activities_df = activities_df.drop(['Title', 'Date', 'Distance', 'Time', 'Elevation Rate'], axis = 1)

X = activities_df.drop(columns=['Score'])
y = activities_df['Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 314)

# Train an XGBoost model
model = xgb.XGBRegressor(objective = 'reg:squarederror', n_estimators = 500, learning_rate = 0.15, max_depth = 5)
model.fit(X_train, y_train) 

# Evaluate the model
y_pred = model.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2)
print(f'Mean Squared Error: {mse}') 

# ! SHAP Analysis
explainer = shap.Explainer(model)
shap_values = explainer(X)

st.title('Individual Activity Performance Analysis')
st.write('This app analyzes your personal running activities and provides insights into the factors that affected your overall performance both positively and negatively within each activity. ')

# ? Slider to select a specific activity (sorted by date), defaults to most recent activity
activity_date = st.select_slider('Select Activity', value = clean_df['Date'][0], options = clean_df['Date'].T,)
data_point = clean_df.index[clean_df['Date'] == activity_date][0]

# ? Force plot of the activity
shap.initjs()
st_shap(shap.plots.force(shap_values[data_point]), height = 150, width = 3000)

# ? Waterfall plot of activity
left_1, right_1 = st.columns(2)
with left_1:
    st_shap(shap.plots.waterfall(shap_values[data_point]), height = 480, width = 660)

# ? Activity explanation
with right_1:

    # Take the row of the selected activity from the original (outside activities) dataset to get the details
    activity_details = clean_df.iloc[data_point]
    
    # Also get the performance score for the selected activity
    score = activities_df.iloc[data_point]['Score']

    # Take the top 10% of activities based on the performance score
    top_activities = activities_df[activities_df['Score'] > activities_df['Score'].quantile(0.9)]
    means = top_activities.mean()

    # Display some activity details, including the date, title, distance, and performance score (red if negative, green if positive)
    st.write(f"## {activity_details['Date'].split(' ')[0]}, {activity_details['Title']}")
    if score < 0:
        st.markdown(f"Your activity totalling **{activity_details['Distance']} km** with an elevation rate of **{activity_details['Elevation Rate']:.2f}m/km** had a performance score of :red[{score:.2f}]")
    else:
        st.markdown(f"Your activity totalling **{activity_details['Distance']} km** with an elevation rate of **{activity_details['Elevation Rate']:.2f}m/km** had a performance score of :green[{score:.2f}]")

    positive_shap = []
    negative_shap = []

    # List of features to exclude from the analysis
    features_not_included = ['Date', 'Title', 'Score', 'Distance', 'Time', 'Elevation Rate']

    # For each feature, separate into lists based on positive and negative SHAP values
    for feature, value in activity_details.items():
        if feature not in features_not_included:

            # Get the SHAP value for the feature
            shap_value = shap_values[data_point][feature].values
            threshold = means[feature]

            if shap_value > 0:
                # Append into a tuple with the exact order to match the feedback function
                positive_shap.append((shap_value, feature, value, threshold))
            else:
                negative_shap.append((shap_value, feature, value, threshold))

    # Create 2 columns to write positive and negative feedback for each feature
    left_2, right_2 = st.columns(2)
    with left_2:
        for i in positive_shap:        
            # Provide feedback based on the SHAP value and feature value
            feedback = feature_feedback(i[0], i[1], i[2], i[3])
            st.write(feedback)
    
    with right_2:
        for i in negative_shap:
            feedback = feature_feedback(i[0], i[1], i[2], i[3])
            st.write(feedback)


# ! Recent Activity Performance
st.sidebar.title('Recent Activity Performance')
st.sidebar.write(f'Compare your last activities to your average performance')
n_activities = st.sidebar.selectbox('Number of activities to compare', options=[5, 10, 20, 30], index=1)

# DF with only the features to analyse
features_activities = clean_df.drop(columns = features_not_included)

# Get the mean of the features over the last n activities
recent_activities_means = features_activities.head(n_activities).mean().to_frame().T
# Get the means of the features over all activities to compare against
feature_overall_means = features_activities.mean().to_frame().T
# Get the total difference between the means of recent activities and overall
difference_df = recent_activities_means - feature_overall_means

# ? Loop through each feature and create a metric for each one
# Create a 2 column layout for the metrics
column_counter = 0
recent_col1, recent_col2 = st.sidebar.columns(2)

# Define which features are good being higher and which are good being lower
features_good_higher = ['Avg Run Cadence', 'Normalized Power® (NP®)', 'Active Ratio']
features_neutral = ['Avg Stride Length']
features_good_lower = ['Avg Ground Contact Time', 'Avg Resp', 'Avg Vertical Ratio', 'Avg Vertical Oscillation']

for feature in features_activities.columns:
    # Get the value of the feature for the recent activities and overall
    recent_value = recent_activities_means[feature].values[0]
    difference = difference_df[feature].values[0]

    # If the column counter is even, use the first column, otherwise use the second
    if column_counter % 2 == 0:
        col = recent_col1
    else:
        col = recent_col2

    # Different coloured metrics depending on if the feature is better to be higher, lower or neutral
    if feature in features_good_higher:
        # Green if difference > 0 (recent value is higher)
        col.metric(
            label = feature,
            value = f"{recent_value:.2f}",
            delta = f"{difference:.2f}",
            delta_color = "normal"
        )

    elif feature in features_good_lower:
        # Green if difference < 0 (recent value is lower)
        col.metric(
            label = feature,
            value = f"{recent_value:.2f}",
            delta = f"{difference:.2f}",
            delta_color = "inverse"
        )

    elif feature in features_neutral:
        # No color
        col.metric(
            label = feature,
            value = f"{recent_value:.2f}",
            delta = f"{difference:.2f}",
            delta_color = "off"
        )

    column_counter += 1

# ! Overall Trends
st.title('Overall Activity Trends')

performance_tab, features_tab = st.tabs(["Performance", "Features"])

# ? Performance Score Chart
with performance_tab:
    st.write('### Performance Score over Time')
    # Create a version of clean df but with only the date and not the time
    split_date_clean_df = clean_df.copy()
    split_date_clean_df['Date'] = split_date_clean_df['Date'].apply(lambda x: x.split(' ')[0])

    # Group by date and get performance scores
    # If there were multiple activities on the same date, take the mean of the performance scores
    performance_chart_df = split_date_clean_df.groupby('Date')['Score'].mean().sort_index().to_frame()
    # Also get a column of zeros to plot along with the scores
    performance_chart_df['Zeros'] = 0

    # Plot on a line chart
    st.line_chart(performance_chart_df,
                y = ['Score', 'Zeros'],
                color = ['#ff0051', '#a8a8a8'],
                x_label = 'Date', 
                y_label = 'Performance Score', 
                use_container_width=True)

# ? Feature Scores Chart
with features_tab:
    st.write('### Feature Scores over Time')
    possible_features = clean_df.drop(columns=['Date', 'Title', 'Distance', 'Time', 'Score']).columns
    selected_feature = st.selectbox('Select Feature to Plot', options=possible_features)

    # Group by date and get the mean of the selected feature
    feature_chart_df = split_date_clean_df.groupby('Date')[selected_feature].mean().sort_index().to_frame()
    st.line_chart(feature_chart_df,
                y = selected_feature,
                color = '#008bfb',
                x_label = 'Date', 
                y_label = selected_feature,
                use_container_width=True)
    