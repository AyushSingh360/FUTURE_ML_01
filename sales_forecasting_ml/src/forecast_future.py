import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def forecast_rf(model, last_30_days, steps=30):
    # This is a naive iterative forecast just for demonstration.
    # In a real scenario, you'd iteratively update lag features.
    
    # We will just predict using the ARIMA model for the final visualization
    # since it's cleaner for raw time series forecasting without having to rebuild all exogenous features iteratively.
    pass

def forecast_arima(df, steps=30):
    print(f"Forecasting next {steps} days...")
    sns.set_theme(style="whitegrid")
    
    # For a fresh forecast, we should retrain ARIMA on the full dataset
    series = df['Sales'].values
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    
    forecast = model_fit.forecast(steps=steps)
    
    # Generate future dates
    last_date = df['Order Date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, steps+1)]
    
    # Plotting
    # Show last 90 days of actual data + 30 days of forecast
    plot_df = df.tail(90).copy()
    
    plt.figure(figsize=(12, 6))
    plt.plot(plot_df['Order Date'], plot_df['Sales'], label='Actual Sales (Last 90 Days)')
    plt.plot(future_dates, forecast, label='Forecasted Sales (Next 30 Days)', color='orange', linestyle='--')
    
    plt.title('Sales Forecast for Next 30 Days')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.tight_layout()
    
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/forecast_results.png')
    print("Saved output/forecast_results.png")

if __name__ == '__main__':
    try:
        df = pd.read_csv('data/featured_data.csv')
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        forecast_arima(df, steps=30)
    except Exception as e:
        print(f"Error forecasting: {e}")
