# Sales & Demand Forecasting for Businesses

## Project Objective
The goal of this project is to create an end-to-end Machine Learning system that predicts future sales using historical retail sales data. Accurately forecasting sales is crucial for businesses as it helps optimize inventory, improve staffing schedules, and maximize overall profitability.

## Dataset Explanation
This project uses a synthetic dataset designed to mimic the well-known "Superstore Sales" dataset. It contains 10,000 records of daily transactions across different categories (Furniture, Office Supplies, Technology) and regions. The columns include:
- `Order Date`: The date the order was placed.
- `Category`: Product category.
- `Region`: Store location region.
- `Quantity`: Number of items sold.
- `Sales`: Total revenue from the sale.
- `Profit`: Profit earned from the sale.

## Model Explanation
We implemented and compared two models for forecasting daily sales:
1. **RandomForestRegressor**: An ensemble machine learning model. To make it work with time-series data, we engineered features such as day, month, year, day of the week, and lag features (previous day's sales, 7-day lags) along with rolling averages.
2. **ARIMA (AutoRegressive Integrated Moving Average)**: A classic statistical time-series forecasting model that relies entirely on historical sales trends and seasonality patterns.

## Results and Interpretation
- The evaluation script compares both models using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.
- Historically, Machine Learning models like Random Forest perform well when there are complex non-linear relationships and many engineered features, while ARIMA is solid for pure trend and seasonality capture.

### Business Insights
- **Demand Trends**: By analyzing the historical sales (see `output/sales_trend.png`), businesses can observe overall growth trajectories over the years.
- **Seasonal Patterns**: There are visible spikes in demand during holiday months (November, December). Businesses can use this insight to proactively stock up on inventory and hire temporary staff during these peak periods.

## Visualizations
The project generates several key visualizations stored in the `output/` folder:
- `sales_trend.png`: A line chart showing daily sales over time.
- `seasonal_pattern.png`: A bar chart showing the average daily sales for each month to highlight seasonality.
- `forecast_results.png`: A line chart showing the actual sales for the last 90 days alongside the forecasted sales for the next 30 days.

## How to Run the Project

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Data & Preprocess**:
   ```bash
   python generate_data.py
   python src/data_preprocessing.py
   python src/feature_engineering.py
   ```

3. **Exploratory Data Analysis (EDA)**:
   ```bash
   python src/generate_plots.py
   ```
   *(Alternatively, you can run the Jupyter notebook `notebooks/sales_forecasting_analysis.ipynb`)*

4. **Model Training & Evaluation**:
   ```bash
   python src/train_model.py
   python src/evaluate_model.py
   ```

5. **Future Forecasting**:
   ```bash
   python src/forecast_future.py
   ```
