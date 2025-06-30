import pandas as pd
import streamlit as st
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
st.title("📈 AI-Powered Sales Forecasting Dashboard")

CSV_URL = "https://raw.githubusercontent.com/Anupam1707/FUTURE_ML_01/main/data/sales_data_sample.csv"

st.markdown(
    f"🔗 **Dataset Source:** [sales_data_sample.csv]({CSV_URL})",
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    return pd.read_csv(CSV_URL, encoding="cp1252")

df = load_data()

df.columns = df.columns.str.strip()

df.rename(columns={"ORDERDATE": "ds", "SALES": "y"}, inplace=True)

if "ds" not in df.columns or "y" not in df.columns:
    st.error("❌ Required columns 'ORDERDATE' and 'SALES' not found.")
    st.stop()

df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
df.dropna(subset=["ds", "y"], inplace=True)

st.subheader("📊 Cleaned Sales Data")
st.dataframe(df[["ds", "y"]].head())

model = Prophet()
model.fit(df[["ds", "y"]])

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

st.subheader("🔮 Forecast for Next 90 Days")
fig1 = model.plot(forecast)
st.pyplot(fig1)

st.subheader("📉 Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)