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
    df['Weekday'] = df['Order Date'].dt.day_name()
    df['IsWeekend'] = df['Weekday'].isin(['Saturday', 'Sunday'])
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

st.subheader("📊 Sales by Day of Week")
weekday_stats = filtered_df.groupby('Weekday')['Sales'].mean().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
fig_weekday = px.bar(weekday_stats, x=weekday_stats.index, y=weekday_stats.values,
                     labels={'x': 'Day of Week', 'y': 'Average Sales'},
                     title='Average Sales by Day of the Week')
st.plotly_chart(fig_weekday, use_container_width=True)

st.subheader("📌 Business Insights & Recommendations")
recommendations = []

if high_month != low_month:
    recommendations.append(f"Increase inventory and marketing during peak months like **Month {high_month}**")
    recommendations.append(f"Offer promotions or bundles in low-performing months like **Month {low_month}**")

if all(x in filtered_df['IsWeekend'].unique() for x in [True, False]):
    avg_weekend = filtered_df[filtered_df['IsWeekend'] == True]['Sales'].mean()
    avg_weekday = filtered_df[filtered_df['IsWeekend'] == False]['Sales'].mean()
    if avg_weekend > avg_weekday:
        recommendations.append("Leverage higher weekend sales by scheduling campaigns on Saturdays and Sundays")
    else:
        recommendations.append("Strengthen weekday strategies, as they outperform weekends in this segment")

st.markdown(f"""
- 🛒 **Top-selling product**: `{top_product}`
- 📆 **Peak sales month**: `Month {high_month}`
- 📉 **Lowest sales month**: `Month {low_month}`
- 💡 **Recommendations**:
""")

for rec in recommendations:
    st.markdown(f"  - {rec}")

export_df = filtered_df.copy()
export_df['Year'] = export_df['Order Date'].dt.year
export_df['Month'] = export_df['Order Date'].dt.strftime('%B')
export_df['Day'] = export_df['Order Date'].dt.day
export_df['Month_Num'] = export_df['Order Date'].dt.month
export_df['Weekday'] = export_df['Order Date'].dt.day_name()
export_df['IsWeekend'] = export_df['Weekday'].isin(['Saturday', 'Sunday'])

csv_cleaned = export_df.to_csv(index=False)
st.download_button(
    label="⬇️ Download Cleaned Dataset (Filtered Superstore)",
    data=csv_cleaned,
    file_name='superstore_cleaned.csv',
    mime='text/csv'
)
forecast_export = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast_export['ds'] = forecast_export['ds'].dt.date
actual_sales = daily_sales.rename(columns={'ds': 'ds', 'y': 'actual_sales'})
forecast_merged = pd.merge(forecast_export, actual_sales, on='ds', how='left')

csv_forecast = forecast_merged.to_csv(index=False)
st.download_button(
    label="📈 Download Forecast Output (Actual vs Predicted)",
    data=csv_forecast,
    file_name='sales_forecast.csv',
    mime='text/csv'
)