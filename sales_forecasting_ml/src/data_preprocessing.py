import pandas as pd
import os

SUPERSTORE_PATH = r"C:\Users\spide\OneDrive\Documents\FUTURE_ML_01\sales_forecasting_ml\Dataset\Sample - Superstore.csv"

def load_data(filepath=SUPERSTORE_PATH):
    """Load the raw dataset using Pandas."""
    try:
        # Superstore often uses windows-1252 encoding
        df = pd.read_csv(filepath, encoding='windows-1252')
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None

def preprocess_data(df):
    """
    Preprocess the data:
    1. Handle missing values
    2. Convert date column to datetime
    3. Aggregate sales by date
    """
    if df is None: return None
    
    # 1. Handle missing values
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    print(f"Missing values handled.")
    
    # 2. Convert date column to datetime
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        print(f"Converted 'Order Date' to datetime.")
    else:
        print("Warning: 'Order Date' column not found.")
        
    return df

def aggregate_by_date(df):
    """Aggregate sales and profit by date for time-series forecasting."""
    if df is None or 'Order Date' not in df.columns: return None
    
    # Ensure Sales, Profit, Quantity are numeric
    for col in ['Sales', 'Profit', 'Quantity']:
        if col in df.columns:
            if df[col].dtype == object:
                # Remove common currency symbols and commas if any exist
                df[col] = df[col].str.replace(r'[$,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Aggregate numeric columns by order date
    daily_sales = df.groupby('Order Date').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    # Ensure all dates in range exist (fill gaps with 0)
    idx = pd.date_range(daily_sales['Order Date'].min(), daily_sales['Order Date'].max())
    daily_sales.set_index('Order Date', inplace=True)
    daily_sales = daily_sales.reindex(idx, fill_value=0).reset_index()
    daily_sales = daily_sales.rename(columns={'index': 'Order Date'})
    
    print(f"Aggregated daily sales. min_date: {daily_sales['Order Date'].min()} max_date: {daily_sales['Order Date'].max()} Total days: {len(daily_sales)}")
    
    # Save the processed data
    os.makedirs('data', exist_ok=True)
    daily_sales.to_csv('data/processed_data.csv', index=False)
    print("Saved processed data to data/processed_data.csv")
    
    return daily_sales

if __name__ == "__main__":
    df = load_data()
    df_clean = preprocess_data(df)
    df_daily = aggregate_by_date(df_clean)
