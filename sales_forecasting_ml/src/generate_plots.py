import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_eda_plots():
    # Load data
    try:
        df = pd.read_csv('data/processed_data.csv')
    except FileNotFoundError:
        print("processed_data.csv not found.")
        return
        
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    sns.set_theme(style="whitegrid")
    os.makedirs('output', exist_ok=True)
    
    # 1. Historical Sales Trend
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=df, x='Order Date', y='Sales')
    plt.title('Daily Sales Overview (2020-2023)')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.tight_layout()
    plt.savefig('output/sales_trend.png')
    print("Saved output/sales_trend.png")
    # 2. Monthly Seasonality
    df['Month'] = df['Order Date'].dt.month
    monthly_sales = df.groupby('Month')['Sales'].mean().reset_index()
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=monthly_sales, x='Month', y='Sales')
    plt.title('Average Daily Sales by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.tight_layout()
    plt.savefig('output/seasonal_pattern.png')
    print("Saved output/seasonal_pattern.png")

if __name__ == '__main__':
    generate_eda_plots()
