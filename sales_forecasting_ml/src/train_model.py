import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import pickle
import os

def prepare_data_for_rf(df):
    """Prepare features and target for Random Forest."""
    # We predict 'Sales'
    # Drop columns that we can't use directly or that leak future
    drop_cols = ['Order Date', 'Sales']
    
    X = df.drop(columns=drop_cols)
    y = df['Sales']
    
    # Train test split (temporal)
    # Let's say last 90 days is test set
    split_idx = len(df) - 90
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    with open('models/rf_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    print("Saved Random Forest to models/rf_model.pkl")
    return rf

def train_arima(df):
    print("Training ARIMA...")
    # ARIMA uses just the target series
    # Using order (5,1,0) as a baseline
    series = df['Sales'].values
    train, test = series[:-90], series[-90:]
    
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()
    
    with open('models/arima_model.pkl', 'wb') as f:
        pickle.dump(model_fit, f)
    print("Saved ARIMA to models/arima_model.pkl")
    return model_fit

if __name__ == '__main__':
    try:
        df = pd.read_csv('data/featured_data.csv')
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        
        # Train Random Forest
        X_train, X_test, y_train, y_test = prepare_data_for_rf(df)
        rf_model = train_random_forest(X_train, y_train)
        
        # Train ARIMA
        arima_model = train_arima(df)
        
    except FileNotFoundError:
        print("Error: data/featured_data.csv not found.")
