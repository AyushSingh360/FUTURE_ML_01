import pandas as pd
import os

HOLIDAYS_PATH = r"C:\Users\spide\OneDrive\Documents\FUTURE_ML_01\sales_forecasting_ml\Dataset\holidays_events.csv"

def load_holidays(filepath=HOLIDAYS_PATH):
    try:
        holidays_df = pd.read_csv(filepath)
        # convert date to datetime
        holidays_df['date'] = pd.to_datetime(holidays_df['date'])
        
        # Consider a day holiday if it's not "transferred"
        if 'transferred' in holidays_df.columns:
            actual_holidays = holidays_df[~holidays_df['transferred']]
        else:
            actual_holidays = holidays_df
            
        holiday_dates = set(actual_holidays['date'].dt.date)
        return holiday_dates
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Skipping holiday features.")
        return set()

def engineer_features(df):
    """
    Create time-based and rolling features from the daily sales data.
    Features:
    - year, month, day, day_of_week
    - is_holiday (from external dataset)
    - rolling average (7, 30 days)
    - lag features (t-1, t-7, t-30)
    """
    if df is None or 'Order Date' not in df.columns: return None
    
    # Make sure it's datetime
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df = df.sort_values('Order Date')
    
    # 1. Time-based features
    df['year'] = df['Order Date'].dt.year
    df['month'] = df['Order Date'].dt.month
    df['day'] = df['Order Date'].dt.day
    df['day_of_week'] = df['Order Date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    print("Added time-based features.")
    
    # 2. Holiday features
    holiday_dates = load_holidays()
    df['is_holiday'] = df['Order Date'].dt.date.isin(holiday_dates).astype(int)
    print(f"Added holiday features. Total holidays matching days: {df['is_holiday'].sum()}")
    
    # 3. Lag features
    df['sales_lag_1'] = df['Sales'].shift(1)
    df['sales_lag_7'] = df['Sales'].shift(7)
    df['sales_lag_30'] = df['Sales'].shift(30)
    
    print("Added lag features.")
    
    # 4. Rolling average features
    df['rolling_mean_7d'] = df['Sales'].rolling(window=7, min_periods=1).mean()
    df['rolling_mean_30d'] = df['Sales'].rolling(window=30, min_periods=1).mean()
    
    print("Added rolling average features.")
    
    # Drop rows with NaNs caused by lagging
    df = df.dropna().reset_index(drop=True)
    
    print(f"Final shape after dropping NA due to lags: {df.shape}")
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/featured_data.csv', index=False)
    print("Saved feature engineered data to data/featured_data.csv")
    
    return df

if __name__ == "__main__":
    try:
        df = pd.read_csv('data/processed_data.csv')
        engineer_features(df)
    except FileNotFoundError:
        print("Error: data/processed_data.csv not found. Run data_preprocessing.py first.")
