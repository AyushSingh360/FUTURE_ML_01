import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_superstore_data(num_records=5000):
    np.random.seed(42)
    random.seed(42)
    
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days
    
    categories = ['Furniture', 'Office Supplies', 'Technology']
    regions = ['East', 'West', 'Central', 'South']
    
    data = []
    
    for _ in range(num_records):
        # Generate random date with trend and seasonality built-in
        # more sales in holidays (Nov, Dec)
        random_day = random.randint(0, date_range)
        order_date = start_date + timedelta(days=random_day)
        
        category = random.choice(categories)
        region = random.choice(regions)
        quantity = random.randint(1, 14)
        
        # Base price depends on category
        if category == 'Furniture':
            base_price = np.random.normal(300, 100)
        elif category == 'Technology':
            base_price = np.random.normal(500, 200)
        else:
            base_price = np.random.normal(50, 20)
            
        base_price = max(10, base_price) # Ensure positive
        
        # Add seasonality and trend
        # Trend: increase over years
        year_multiplier = 1 + (order_date.year - 2020) * 0.1
        
        # Seasonality: bump in Nov/Dec
        month_multiplier = 1.0
        if order_date.month in [11, 12]:
            month_multiplier = 1.5
        elif order_date.month in [1, 2]:
            month_multiplier = 0.8
            
        sales = quantity * base_price * year_multiplier * month_multiplier
        
        # Profit margin depends on category, sometimes negative (discounts)
        if category == 'Technology':
            margin = np.random.normal(0.2, 0.1)
        elif category == 'Furniture':
            margin = np.random.normal(0.1, 0.2)
        else:
            margin = np.random.normal(0.3, 0.1)
            
        profit = sales * margin
        
        data.append({
            'Order Date': order_date.strftime('%Y-%m-%d'),
            'Category': category,
            'Region': region,
            'Quantity': quantity,
            'Sales': round(sales, 2),
            'Profit': round(profit, 2)
        })
        
    df = pd.DataFrame(data)
    # Sort by date
    df = df.sort_values('Order Date').reset_index(drop=True)
    
    # Save
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/raw_data.csv', index=False)
    print(f"Generated {len(df)} records of synthetic sales data in data/raw_data.csv")

if __name__ == '__main__':
    generate_superstore_data(10000)
