import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Step 3: Load the dataset
df = pd.read_csv("data/sales_data_sample.csv", encoding='latin1')

# Step 4: Clean and prepare the data
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')
df = df.dropna(subset=['ORDERDATE'])

# Step 5: Group by date and sum sales
daily_sales = df.groupby(df['ORDERDATE'].dt.date)['SALES'].sum().reset_index()
daily_sales.columns = ['ds', 'y']  # Prophet expects 'ds' for date and 'y' for value

# Step 6: Initialize and fit the model
model = Prophet()
model.fit(daily_sales)

# Step 7: Make future dataframe for next 90 days
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Step 8: Plot the forecast
fig1 = model.plot(forecast)
plt.title("Sales Forecast for Next 90 Days")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
fig1.savefig("images/forecast_plot.png")

# Step 9: Plot forecast components (trend, seasonality, etc.)
fig2 = model.plot_components(forecast)
plt.tight_layout()
fig2.savefig("images/forecast_components.png")