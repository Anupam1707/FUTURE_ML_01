import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px

st.set_page_config(page_title="📊 Sales Forecasting Dashboard", layout="wide")
st.title("📈 AI-Powered Sales Forecasting - Superstore Edition")

st.markdown("**Data Source:** [Sample Superstore](https://raw.githubusercontent.com/Anupam1707/FUTURE_ML_01/main/data/SampleSuperstore.csv) — Tableau Community")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Anupam1707/FUTURE_ML_01/main/data/SampleSuperstore.csv"
    df = pd.read_csv(url)
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    return df

df = load_data()

st.sidebar.header("🔍 Filters")

with st.sidebar.expander("📍 Region"):
    all_regions = df['Region'].unique()
    selected_region = [r for r in all_regions if st.checkbox(r, value=True, key=f"region_{r}")]
    if len(selected_region) == 0:
        st.error("⚠️ Please select at least one Region.")
        st.stop()

with st.sidebar.expander("🗂 Category"):
    all_categories = df['Category'].unique()
    selected_category = [c for c in all_categories if st.checkbox(c, value=True, key=f"category_{c}")]
    if len(selected_category) == 0:
        st.error("⚠️ Please select at least one Category.")
        st.stop()

with st.sidebar.expander("👥 Segment"):
    all_segments = df['Segment'].unique()
    selected_segment = [s for s in all_segments if st.checkbox(s, value=True, key=f"segment_{s}")]
    if len(selected_segment) == 0:
        st.error("⚠️ Please select at least one Segment.")
        st.stop()

filtered_df = df[(df['Region'].isin(selected_region)) &
                 (df['Category'].isin(selected_category)) &
                 (df['Segment'].isin(selected_segment))]

daily_sales = filtered_df.groupby(filtered_df['Order Date'].dt.date)['Sales'].sum().reset_index()
daily_sales.columns = ['ds', 'y']

model = Prophet()
model.fit(daily_sales)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

col1, col2 = st.columns(2)

with col1:
    total_sales = round(filtered_df['Sales'].sum(), 2)
    st.metric("💰 Total Sales", f"${total_sales:,}")

    top_product = filtered_df.groupby("Product Name")['Sales'].sum().idxmax()
    st.metric("🏆 Top Product", top_product)

with col2:
    low_month = filtered_df.groupby(filtered_df['Order Date'].dt.month)['Sales'].sum().idxmin()
    st.metric("📉 Lowest Sales Month", f"Month {low_month}")

    high_month = filtered_df.groupby(filtered_df['Order Date'].dt.month)['Sales'].sum().idxmax()
    st.metric("📈 Peak Sales Month", f"Month {high_month}")

st.subheader("📊 Sales Forecast vs Actual")
fig_forecast = plot_plotly(model, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

st.subheader("📆 Monthly Sales Trend")
filtered_df['Month'] = filtered_df['Order Date'].dt.to_period("M").astype(str)
monthly = filtered_df.groupby('Month')['Sales'].sum().reset_index()
fig_monthly = px.line(monthly, x='Month', y='Sales', title='Monthly Sales')
st.plotly_chart(fig_monthly, use_container_width=True)

st.subheader("📆 Yearly Sales Trend")
filtered_df['Year'] = filtered_df['Order Date'].dt.year
yearly = filtered_df.groupby('Year')['Sales'].sum().reset_index()
fig_yearly = px.bar(yearly, x='Year', y='Sales', title='Yearly Sales Comparison')
st.plotly_chart(fig_yearly, use_container_width=True)