import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv("data/sales_data_sample.csv", encoding='latin1')

df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')
df = df.dropna(subset=['ORDERDATE'])

daily_sales = df.groupby(df['ORDERDATE'].dt.date)['SALES'].sum().reset_index()
daily_sales.columns = ['ds', 'y']

model = Prophet()
model.fit(daily_sales)

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

fig1 = model.plot(forecast)
plt.title("Sales Forecast for Next 90 Days")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
fig1.savefig("images/forecast_plot.png")

fig2 = model.plot_components(forecast)
plt.tight_layout()
fig2.savefig("images/forecast_components.png")