import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TITLE

st.title(" NYC Taxi Trip Duration Prediction")

st.write("Predict taxi trip duration using location and time features")

# LOAD DATA

@st.cache_data
def load_data():
    data = pd.read_csv("train.csv")
    return data

data = load_data()

st.subheader("Step 1 : Original Dataset")
st.write(data.head())

# HAVERSINE DISTANCE

st.subheader("Step 2 : Geospatial Feature Engineering")

def haversine(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(np.radians,
    [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    r = 6371
    return c * r


data["distance_km"] = haversine(
    data["pickup_longitude"],
    data["pickup_latitude"],
    data["dropoff_longitude"],
    data["dropoff_latitude"]
)

st.write("Dataset after distance calculation")
st.write(data.head())

# TIME FEATURES

st.subheader("Step 3 : Time Feature Extraction")

data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])

data["hour"] = data["pickup_datetime"].dt.hour
data["day_of_week"] = data["pickup_datetime"].dt.dayofweek

# Rush hour indicator
data["rush_hour"] = data["hour"].apply(
    lambda x: 1 if x in [7,8,9,16,17,18,19] else 0
)

st.write(data.head())

# OUTLIER REMOVAL

st.subheader("Step 4 : Removing Extreme Trips")

data = data[(data["trip_duration"] > 60) &
            (data["trip_duration"] < 7200)]

st.write("Dataset after removing outliers")
st.write(data.head())

# LOG TRANSFORMATION

st.subheader("Step 5 : Log Transformation")

data["log_trip_duration"] = np.log1p(data["trip_duration"])
st.write(data[["trip_duration","log_trip_duration"]])

# SELECT FEATURES

features = [
    "passenger_count",
    "distance_km",
    "hour",
    "day_of_week",
    "rush_hour"
]

X = data[features]
y = data["log_trip_duration"]

# TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# MODEL TRAINING

st.subheader("Step 6 : Training Linear Regression")

model = LinearRegression()
model.fit(X_train,y_train)

st.success("Model Trained")

st.write("Coefficients:")
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})
st.write(coef_df)


# MODEL EVALUATION

st.subheader("Step 7 : Model Evaluation")

pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,pred))
r2 = r2_score(y_test,pred)

st.write("RMSE:",rmse)
st.write("R² Score:",r2)

# RESIDUAL DIAGNOSTICS

st.subheader("Step 8 : Residual Diagnostics")

residuals = y_test - pred

fig, ax = plt.subplots()

ax.scatter(pred, residuals)

ax.set_xlabel("Predicted Values")
ax.set_ylabel("Residuals")
ax.set_title("Residual Plot")

st.pyplot(fig)

# BIAS vs VARIANCE

st.subheader("Step 9 : Bias vs Variance")

train_pred = model.predict(X_train)

train_rmse = np.sqrt(mean_squared_error(y_train,train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test,pred))

st.write("Training RMSE:",train_rmse)
st.write("Testing RMSE:",test_rmse)

if train_rmse < test_rmse:
    st.write("Model shows some variance (overfitting risk)")
else:
    st.write("Model shows high bias")

# PREDICTION FORM

st.subheader("Step 10 : Predict Taxi Trip Duration")

passengers = st.slider("Passenger Count",1,6)

distance = st.number_input("Distance (km)",0.1,50.0)

hour = st.slider("Pickup Hour",0,23)

day = st.slider("Day of Week (0=Mon)",0,6)

rush = st.selectbox("Rush Hour",[0,1])

if st.button("Predict Trip Duration"):

    input_data = np.array([[

        passengers,
        distance,
        hour,
        day,
        rush

    ]])

    log_pred = model.predict(input_data)

    duration = np.expm1(log_pred)

    st.success(f"Estimated Trip Duration: {duration[0]:.2f} seconds")
