🇮🇳 India Metro Air Quality (PM2.5) Forecaster
This is a Streamlit web application that uses a machine learning model to analyze historical air quality data and forecast future PM2.5 levels for major metropolitan cities in India.

<img width="1918" height="962" alt="ss1" src="https://github.com/user-attachments/assets/0c1be88d-7f8b-4197-a342-2f9f6fa692a5" />
<img width="1918" height="966" alt="ss2" src="https://github.com/user-attachments/assets/3b3e755c-3960-488f-959b-0168466ebbc3" />
<img width="1918" height="965" alt="ss3" src="https://github.com/user-attachments/assets/a4ea472c-b4c5-4d25-8402-d8d793bcaf89" />

Features
Multi-City Analysis: Select from a dropdown of major Indian cities (Kolkata, Delhi, Mumbai, Chennai, Bengaluru, Ahmedabad, Hyderabad, and Pune) to get a specific forecast.

Live Weather Integration: Automatically fetches historical weather data from the Open-Meteo API to provide crucial environmental context to the model.

24-Hour Forecast: Generates an hourly forecast for the next 24 hours based on the latest available data.

AQI Conversion: Converts raw PM2.5 predictions into the official Air Quality Index (AQI) and provides clear health advisories.

Model Insights: Includes detailed performance metrics (RMSE) and visualizations of the model's predictions vs. actuals and feature importances.

Project Workflow
This project demonstrates a complete end-to-end machine learning workflow.

1. Data Collection & Inspection
The project uses hourly PM2.5 data obtained from the official Central Pollution Control Board (CPCB) portal for various monitoring stations in each city.

2. Data Cleaning and Preparation
A robust data processing pipeline was built to handle the raw data files. This pipeline:

Automatically detects the correct header row.

Standardizes column names.

Converts date and time information into a proper datetime format.

Intelligently handles missing values using the forward-fill method.

3. Feature Engineering
To improve the model's predictive power, several new features were engineered:

Time-based Features: hour, day_of_week, and month to capture daily, weekly, and seasonal patterns.

Lag Features: The PM2.5 value from 1 hour ago and 24 hours ago to give the model a sense of recent trends.

Rolling Mean Feature: A 3-hour rolling average to capture the short-term trend.

Weather Features: Temperature, Humidity, and Wind Speed are integrated from the Open-Meteo API.

4. Model Training and Evaluation
Model: The project uses a RandomForestRegressor from the scikit-learn library, a powerful ensemble model well-suited for this type of regression task.

Training: A new model is trained from scratch on the specific historical data for each city selected by the user.

Evaluation: The model's performance is evaluated using the Root Mean Squared Error (RMSE) on a held-out test set (the final 20% of the data).

How to Run This Project Locally
Clone the repository:

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

Prepare the data:

Create a data folder inside the main project directory.

Download the hourly PM2.5 data for each city from the CPCB portal.

Save the files inside the data folder with the correct names (e.g., kolkata_aqi_data.xlsx, delhi_aqi_data.xlsx, etc.).

Install the dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

