import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(
    filename='rossmann_eda.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(train_path, test_path, store_path):
    """
    Loads the training, testing, and store data.
    """
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        store = pd.read_csv(store_path)
        logging.info("Datasets loaded successfully.")
        return train, test, store
    except Exception as e:
        logging.error(f"Error loading datasets: {e}")
        raise

def merge_data(train, test, store):
    """
    Merges training and testing data with the store data.
    """
    try:
        train_store = pd.merge(train, store, on='Store', how='left')
        test_store = pd.merge(test, store, on='Store', how='left')
        logging.info("Datasets merged successfully.")
        return train_store, test_store
    except Exception as e:
        logging.error(f"Error merging datasets: {e}")
        raise

def check_missing_values(df, name):
    """
    Checks and logs missing values in the given dataframe.
    """
    missing_values = df.isnull().sum()
    missing_percent = (df.isnull().sum() / df.shape[0]) * 100
    logging.info(f"Checked missing values for {name}")
    return pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})

def handle_missing_values(train_store, test_store):
    """
    Handles missing values by filling NaNs with 0.
    """
    train_store.fillna(0, inplace=True)
    test_store.fillna(0, inplace=True)
    logging.info("Handled missing values by filling NaNs with 0.")
    
def detect_outliers(df, column):
    """
    Detects and logs outliers in a specific column of the dataframe.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    logging.info(f"Detected outliers in column {column}")
    return outliers

def cap_outliers(train_store, column):
    """
    Caps outliers in the specified column by setting them to the 99th percentile.
    """
    train_store[column] = np.where(
        train_store[column] > train_store[column].quantile(0.99), 
        train_store[column].quantile(0.99), 
        train_store[column]
    )
    logging.info(f"Capped outliers in {column} column.")
    
def plot_sales_behavior(train_store):
    """
    Plots and logs various sales behavior.
    """
    plt.figure(figsize=(12,6))
    sns.lineplot(data=train_store, x='Date', y='Sales', hue='StateHoliday')
    plt.title('Sales behavior before, during, and after holidays')
    plt.show()
    logging.info("Plotted sales behavior before, during, and after holidays.")
    

def find_seasonal_behavior(train_store):
    """
    Find and plot sales behavior around key seasonal holidays such as Christmas and Easter.
    """
    # Convert the 'Date' column to datetime if it's not already
    train_store['Date'] = pd.to_datetime(train_store['Date'])
    
    # Define key holiday periods
    holidays = {
        'Christmas': ['2013-12-24', '2014-12-24', '2015-12-24'],
        'Easter': ['2014-04-20', '2015-04-05'],
        'New Year': ['2014-01-01', '2015-01-01']
    }
    
    plt.figure(figsize=(14,8))
    
    for holiday, dates in holidays.items():
        # Filter for 2 weeks around each holiday
        for date in dates:
            start_date = pd.to_datetime(date) - pd.Timedelta(days=14)
            end_date = pd.to_datetime(date) + pd.Timedelta(days=14)
            holiday_sales = train_store[(train_store['Date'] >= start_date) & (train_store['Date'] <= end_date)]
            plt.plot(holiday_sales['Date'], holiday_sales['Sales'], label=f'{holiday} {date}')
    
    plt.title('Seasonal Purchase Behavior around Key Holidays')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()
    logging.info("Analyzed and plotted seasonal purchase behaviors.")



def analyze_promo_effect(train_store):
    """
    Plots and logs the effect of promotions on sales.
    """
    plt.figure(figsize=(12,6))
    sns.boxplot(data=train_store, x='Promo', y='Sales')
    plt.title('Promo Effect on Sales')
    plt.show()
    logging.info("Plotted promo effect on sales.")
    
    
def analyze_store_opening_closing(train_store):
    """
    Analyzes customer behavior based on store opening and closing times.
    """
    # Aggregate sales data based on whether the store was open or closed
    sales_open = train_store[train_store['Open'] == 1]['Sales'].sum()
    sales_closed = train_store[train_store['Open'] == 0]['Sales'].sum()
    
    # Count number of days stores were open and closed
    open_days = train_store['Open'].value_counts()
    
    # Plot sales comparison for open and closed days
    plt.figure(figsize=(10,6))
    sns.barplot(x=['Open', 'Closed'], y=[sales_open, sales_closed], palette='viridis')
    plt.title('Sales During Store Open and Closed Days')
    plt.ylabel('Total Sales')
    plt.show()
    logging.info("Analyzed and plotted sales during store opening and closing times.")

    # Plot distribution of sales only on open days
    plt.figure(figsize=(10,6))
    sns.histplot(train_store[train_store['Open'] == 1]['Sales'], bins=30, kde=True)
    plt.title('Distribution of Sales During Open Days')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    plt.show()
    logging.info("Plotted sales distribution on open days.")
    
    # Count the number of customers on open days
    if 'Customers' in train_store.columns:
        plt.figure(figsize=(10,6))
        sns.lineplot(data=train_store[train_store['Open'] == 1], x='Date', y='Customers')
        plt.title('Customer Trends During Open Days')
        plt.xlabel('Date')
        plt.ylabel('Number of Customers')
        plt.show()
        logging.info("Analyzed and plotted customer trends during store opening times.")

