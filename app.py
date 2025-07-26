# app.py

# Bateri Web prediction
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
import pytz
import holidays
import matplotlib.pyplot as plt
import zipfile
import sklearn, joblib, sys
import streamlit as st

LAT, LON = -33.45, -70.66  # Santiago
TIMEZONE = "America/Santiago"

def load_model_from_zip(zip_path, inner_filename):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(inner_filename) as f:
            model = joblib.load(f)
    return model

def get_forecast():
    tomorrow = (datetime.now(pytz.timezone(TIMEZONE)) + timedelta(days=1)).strftime("%Y-%m-%d")
    hourly_vars = ",".join([
        "temperature_2m",
        "apparent_temperature",
        "relative_humidity_2m",
        "dew_point_2m",
        "surface_pressure",
        "precipitation",
        "cloudcover",
        "wind_speed_10m",
        "wind_direction_10m",       
        "shortwave_radiation",
        "rain",
        "snowfall"
    ])

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={LAT}&longitude={LON}"
        f"&hourly={hourly_vars}"
        f"&timezone=auto"
        f"&start_date={tomorrow}&end_date={tomorrow}"
    )
    response = requests.get(url)
    hourly = response.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["datetime"] = pd.to_datetime(df["time"])
    
    # Rename to match training variable names
    df = df.rename(columns={
        "cloudcover": "cloud_cover",
        # Add more mappings if needed:
        # e.g., "wind_speed_10m": "wind_speed",
    })

    return df


def create_lagged_features(df, required_features):
    df = df.copy()
    base_lags = set()

    for col in required_features:
        if "_lag" in col:
            var, lag = col.rsplit("_lag", 1)
            base_lags.add((var, int(lag)))

    for var, lag in base_lags:
        if lag == 0:
            df[f"{var}_lag0"] = df[var]
        else:
            df[f"{var}_lag{lag}"] = df[var].shift(lag)

    return df.dropna().reset_index(drop=True)



# Add time features
def add_time_features(df):
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek  # Monday=0
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    
    cl_holidays = holidays.CL()
    df["is_holiday"] = df["datetime"].dt.date.apply(lambda x: x in cl_holidays)
    
    return df


# App layout
st.title("Electricity Demand Forecast")
st.markdown("Predicting hourly electricity demand for **tomorrow** using weather forecast and Random Forest.")

with st.spinner("Loading forecast data and model..."):
    forecast_df = get_forecast()

    model = load_model_from_zip(
        zip_path="rfmodel_streamlit.zip",
        inner_filename="rfmodel_streamlit.joblib"
    )
    required_features = list(model.feature_names_in_)

    forecast_df = create_lagged_features(forecast_df, required_features)
    forecast_df = add_time_features(forecast_df)

    # Filter only needed columns
    X = forecast_df[required_features]

    # Predict
    forecast_df["predicted_demand"] = model.predict(X)


# Show results
dfplot = forecast_df.loc[forecast_df.hour.isin([18, 19, 20, 21, 22])]
st.subheader("Predicted Demand for Tomorrow")
fig, ax = plt.subplots()
ax.plot(dfplot["datetime"], dfplot["predicted_demand"], marker='o')
ax.set_xlabel("Hour")
ax.set_ylabel("Predicted Demand (MW)")
ax.set_title("Hourly Electricity Demand Forecast")
ax.grid(True)
plt.setp(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

# Show data table
st.subheader("Forecast Data")
st.dataframe(dfplot[["datetime", "predicted_demand"]])
