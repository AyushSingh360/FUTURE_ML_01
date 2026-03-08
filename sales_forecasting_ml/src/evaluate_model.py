import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

def evaluate_predictions(y_true, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\\n--- {model_name} Evaluation ---")
    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    return mae, rmse, r2

def main():
    df = pd.read_csv('data/featured_data.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    
    # Needs to match train/test split in train_model!
    split_idx = len(df) - 90
    X_test = df.drop(columns=['Order Date', 'Sales']).iloc[split_idx:]
    y_test = df['Sales'].iloc[split_idx:]
    
    # Evaluate Random Forest
    try:
        with open('models/rf_model.pkl', 'rb') as f:
            rf = pickle.load(f)
        rf_preds = rf.predict(X_test)
        evaluate_predictions(y_test, rf_preds, "Random Forest")
    except FileNotFoundError:
        print("Random Forest model not found.")
        
    # Evaluate ARIMA
    try:
        with open('models/arima_model.pkl', 'rb') as f:
            arima = pickle.load(f)
        # Predict the next 90 days matching the test set
        arima_preds = arima.forecast(steps=90)
        evaluate_predictions(y_test, arima_preds, "ARIMA")
    except FileNotFoundError:
        print("ARIMA model not found.")

if __name__ == '__main__':
    main()
