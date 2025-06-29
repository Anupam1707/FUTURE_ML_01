import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from io import StringIO

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
st.title("📈 AI-Powered Sales Forecasting Dashboard")

st.markdown("Upload a CSV file with historical sales data to generate future sales predictions.")

# File uploader
uploaded_file = st.file_uploader("Upload sales_data_sample.csv", type=["csv"])

if uploaded_file is not None:
    # Read and parse the CSV
    df = pd.read_csv(uploaded_file, encoding='latin1')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')
    df = df.dropna(subset=['ORDERDATE'])

    # Prepare time series
    daily_sales = df.groupby(df['ORDERDATE'].dt.date)['SALES'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']

    st.subheader("📊 Raw Aggregated Sales Data")
    st.dataframe(daily_sales.tail())

    # Prophet modeling
    model = Prophet()
    model.fit(daily_sales)

    # Forecasting
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    # Plotting
    st.subheader("🔮 Forecast Plot")
    fig_forecast = plot_plotly(model, forecast)
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.subheader("📂 Forecast Components")
    fig_components = plot_components_plotly(model, forecast)
    st.plotly_chart(fig_components, use_container_width=True)

    st.success("Forecasting completed! 🎉")
else:
    st.info("👈 Upload your dataset to begin.")
