import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import requests
from datetime import timedelta

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Indian Metro City AQI Forecaster",
    page_icon="🇮🇳",
    layout="wide"
)

# =============================================================================
# City Configuration
# =============================================================================
CITY_DATA = {
    "Kolkata": { "lat": 22.57, "lon": 88.36, "file": "kolkata_aqi_data.xlsx" },
    "Delhi": { "lat": 28.70, "lon": 77.10, "file": "delhi_aqi_data.xlsx" },
    "Mumbai": { "lat": 19.07, "lon": 72.87, "file": "mumbai_aqi_data.xlsx" },
    "Chennai": { "lat": 13.08, "lon": 80.27, "file": "chennai_aqi_data.xlsx" },
    "Bengaluru": { "lat": 12.97, "lon": 77.59, "file": "bengaluru_aqi_data.xlsx" },
    "Ahmedabad": { "lat": 23.02, "lon": 72.57, "file": "ahmedabad_aqi_data.xlsx" },
    "Hyderabad": { "lat": 17.38, "lon": 78.48, "file": "hyderabad_aqi_data.xlsx" },
    "Pune": { "lat": 18.52, "lon": 73.85, "file": "pune_aqi_data.xlsx" }
}

# =============================================================================
# Helper Functions
# =============================================================================
def pm25_to_aqi(pm25):
    # (AQI calculation function remains the same)
    if 0 <= pm25 <= 12.0: return ((50 - 0) / (12.0 - 0)) * (pm25 - 0) + 0
    elif 12.1 <= pm25 <= 35.4: return ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51
    elif 35.5 <= pm25 <= 55.4: return ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101
    elif 55.5 <= pm25 <= 150.4: return ((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151
    elif 150.5 <= pm25 <= 250.4: return ((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201
    else: return 500 # Simplified for brevity

def get_aqi_category(aqi):
    # (AQI category function remains the same)
    aqi = int(aqi)
    if 0 <= aqi <= 50: return ("Good", "Air quality is satisfactory.", "#00e400")
    elif 51 <= aqi <= 100: return ("Moderate", "Air quality is acceptable for most people.", "#ffff00")
    elif 101 <= aqi <= 150: return ("Unhealthy for Sensitive Groups", "Sensitive groups may experience health effects.", "#ff7e00")
    elif 151 <= aqi <= 200: return ("Unhealthy", "Everyone may begin to experience health effects.", "#ff0000")
    elif 201 <= aqi <= 300: return ("Very Unhealthy", "Health warnings of emergency conditions.", "#8f3f97")
    else: return ("Hazardous", "Health alert: everyone may experience more serious health effects.", "#7e0023")

# =============================================================================
# Caching Functions
# =============================================================================
@st.cache_data
def load_and_process_data(city_name):
    city_info = CITY_DATA[city_name]
    file_path = os.path.join("data", city_info["file"])

    if not os.path.exists(file_path):
        st.error(f"Data file not found for {city_name}: {file_path}")
        return None

    # 1. Load AQI Data
    is_csv = file_path.endswith('.csv')
    try:
        temp_df = pd.read_csv(file_path, header=None, nrows=20, on_bad_lines='skip', encoding='utf-8') if is_csv else pd.read_excel(file_path, header=None, nrows=20)
    except Exception as e:
        st.error(f"Could not read the top of the file {file_path}. Error: {e}")
        return None
        
    header_row_index = next((i for i, row in temp_df.iterrows() if 'From Date' in row.astype(str).values), -1)
    if header_row_index == -1: 
        st.error(f"Could not find a header row with 'From Date' in {file_path}.")
        return None

    df_raw = pd.read_csv(file_path, header=header_row_index, on_bad_lines='skip', encoding='utf-8') if is_csv else pd.read_excel(file_path, header=header_row_index)
    df_raw.columns = df_raw.columns.str.strip()
    pm25_col_name = next((col for col in df_raw.columns if 'PM2.5' in str(col)), None)
    if not pm25_col_name: 
        st.error(f"Could not find a 'PM2.5' column in {file_path}.")
        return None

    df_clean = df_raw[['To Date', pm25_col_name]].copy()
    df_clean.rename(columns={'To Date': 'datetime', pm25_col_name: 'pm2_5'}, inplace=True)
    df_clean['datetime'] = pd.to_datetime(df_clean['datetime'], format='%d-%m-%Y %H:%M', errors='coerce')
    df_clean.dropna(subset=['datetime'], inplace=True)
    df_clean.set_index('datetime', inplace=True)
    df_clean['pm2_5'] = pd.to_numeric(df_clean['pm2_5'], errors='coerce')
    aqi_df = df_clean.sort_index()
    last_valid_index = aqi_df['pm2_5'].last_valid_index()
    if last_valid_index: aqi_df = aqi_df.loc[:last_valid_index]
    aqi_df['pm2_5'].fillna(method='ffill', inplace=True)

    # 2. Fetch Weather Data
    start_date = aqi_df.index.min().strftime('%Y-%m-%d')
    end_date = aqi_df.index.max().strftime('%Y-%m-%d')
    api_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={city_info['lat']}&longitude={city_info['lon']}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    response = requests.get(api_url)
    data = response.json()
    weather_df = pd.DataFrame(data['hourly'])
    weather_df['datetime'] = pd.to_datetime(weather_df['time'])
    weather_df.set_index('datetime', inplace=True)
    weather_df.rename(columns={'temperature_2m': 'temp', 'relative_humidity_2m': 'humidity', 'wind_speed_10m': 'wind_speed'}, inplace=True)
    weather_features = weather_df[['temp', 'humidity', 'wind_speed']]

    # 3. Merge and Finalize
    full_df = pd.merge(aqi_df, weather_features, left_index=True, right_index=True, how='left').ffill().dropna()
    return full_df

@st.cache_resource
def train_model_and_get_metrics(df):
    # (This function remains the same)
    df_features = df.copy()
    df_features['hour'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    df_features['pm2_5_lag_1hr'] = df_features['pm2_5'].shift(1)
    df_features['pm2_5_lag_24hr'] = df_features['pm2_5'].shift(24)
    df_features['pm2_5_rolling_mean_3hr'] = df_features['pm2_5'].rolling(window=3).mean()
    df_features.dropna(inplace=True)
    
    X = df_features.drop('pm2_5', axis=1)
    y = df_features['pm2_5']
    
    split_point = int(len(df_features) * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    return model, rmse, importances, df_features, y_test, predictions

# =============================================================================
# Main App UI
# =============================================================================
st.title("Indian Metro City Air Quality (PM2.5) Forecaster")

# --- Sidebar ---
st.sidebar.header("Dashboard Controls")
available_cities = sorted([city for city, data in CITY_DATA.items() if os.path.exists(os.path.join("data", data["file"]))])

if not available_cities:
    st.error("No data files found in the 'data' folder. Please add correctly named data files (e.g., 'kolkata_aqi_data.xlsx') to the 'data' directory.")
else:
    selected_city = st.sidebar.selectbox("Select a City", available_cities)
    
    if st.sidebar.button("Analyze and Forecast"):
        st.header(f"Analysis for {selected_city}")
        
        with st.spinner(f"Processing data and training model for {selected_city}..."):
            processed_data = load_and_process_data(selected_city)
            
            if processed_data is not None and not processed_data.empty:
                model, rmse, importances, df_features, y_test, predictions = train_model_and_get_metrics(processed_data)
                
                # --- Main Page Layout ---
                st.subheader("24-Hour Forecast")
                
                # --- Forecasting Logic ---
                last_known_data = df_features.iloc[-1]
                future_dates = pd.to_datetime([df_features.index[-1] + timedelta(hours=i) for i in range(1, 25)])
                future_df = pd.DataFrame(index=future_dates)
                
                future_df['temp'] = last_known_data['temp']
                future_df['humidity'] = last_known_data['humidity']
                future_df['wind_speed'] = last_known_data['wind_speed']
                future_df['hour'] = future_df.index.hour
                future_df['day_of_week'] = future_df.index.dayofweek
                future_df['month'] = future_df.index.month
                future_df['pm2_5_lag_1hr'] = last_known_data['pm2_5']
                future_df['pm2_5_lag_24hr'] = df_features['pm2_5'].iloc[-24]
                future_df['pm2_5_rolling_mean_3hr'] = (df_features['pm2_5'].iloc[-1] + df_features['pm2_5'].iloc[-2] + df_features['pm2_5'].iloc[-3]) / 3
                
                forecast = model.predict(future_df)
                forecast_df = pd.DataFrame(forecast, index=future_dates, columns=['Predicted PM2.5'])
                
                avg_forecast_pm25 = forecast_df['Predicted PM2.5'].mean()
                avg_aqi = pm25_to_aqi(avg_forecast_pm25)
                category, message, color = get_aqi_category(avg_aqi)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Avg. Predicted PM2.5", value=f"{avg_forecast_pm25:.2f} µg/m³")
                    st.metric(label="Predicted AQI", value=f"{int(avg_aqi)}")
                    st.markdown(f"**<p style='color:{color}; font-size: 20px;'>{category}</p>**", unsafe_allow_html=True)
                    st.info(message)
                with col2:
                    fig, ax = plt.subplots()
                    forecast_df.plot(ax=ax, legend=None, title="PM2.5 Forecast", grid=True)
                    ax.set_ylabel("PM2.5 (µg/m³)")
                    st.pyplot(fig)

                # --- Tabs for Deeper Analysis ---
                tab1, tab2, tab3 = st.tabs(["Historical Data", "Model Performance", "View Data"])
                
                feature_name_map = {'pm2_5': 'PM2.5 (µg/m³)', 'pm2_5_lag_1hr': 'PM2.5 Lag (1 hr)', 'pm2_5_lag_24hr': 'PM2.5 Lag (24 hr)', 'pm2_5_rolling_mean_3hr': 'PM2.5 Rolling Mean (3 hr)', 'hour': 'Hour of Day', 'day_of_week': 'Day of Week', 'month': 'Month', 'temp': 'Temperature (°C)', 'humidity': 'Humidity (%)', 'wind_speed': 'Wind Speed (km/h)'}

                with tab1:
                    st.subheader("Historical PM2.5 Levels")
                    st.line_chart(processed_data['pm2_5'])
                with tab2:
                    st.subheader("Model Performance Insights")
                    st.metric(label="Model Error (RMSE)", value=f"{rmse:.2f} µg/m³")
                    st.write("This means, on average, the model's predictions are off by this amount.")
                    st.subheader("Predictions vs. Actual Values (on Test Set)")
                    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
                    fig, ax = plt.subplots(figsize=(12, 6))
                    results_df.plot(ax=ax, style={'Actual': 'b-', 'Predicted': 'r--'}, grid=True)
                    ax.set_title("Model's Performance on Unseen Data")
                    ax.set_ylabel("PM2.5 (µg/m³)")
                    st.pyplot(fig)
                    st.subheader("Feature Importance")
                    display_importances = importances.copy()
                    display_importances['feature'] = display_importances['feature'].map(feature_name_map)
                    fig_imp, ax_imp = plt.subplots()
                    ax_imp.barh(display_importances['feature'], display_importances['importance'])
                    ax_imp.invert_yaxis()
                    st.pyplot(fig_imp)
                with tab3:
                    st.subheader("Full Processed Data")
                    display_df = df_features.copy()
                    display_df.rename(columns=feature_name_map, inplace=True)
                    st.dataframe(display_df)
            else:
                st.error("Could not process the data for the selected city. Please check the data file.")
