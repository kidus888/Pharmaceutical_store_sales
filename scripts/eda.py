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
    Analyze how promotions affect sales and customer behavior.
    """
    # Compare total sales during promotions vs non-promotion periods
    promo_sales = train_store.groupby('Promo')['Sales'].sum().reset_index()
    
    # Plot total sales during promotion vs non-promotion periods
    plt.figure(figsize=(10,6))
    sns.barplot(x='Promo', y='Sales', data=promo_sales, palette='viridis')
    plt.title('Total Sales: Promo vs No Promo')
    plt.xlabel('Promotion (1 = Yes, 0 = No)')
    plt.ylabel('Total Sales')
    plt.show()
    logging.info("Analyzed and plotted total sales during promo vs non-promo periods.")
    
    # Compare the number of customers during promotions vs non-promotion periods
    if 'Customers' in train_store.columns:
        promo_customers = train_store.groupby('Promo')['Customers'].mean().reset_index()
        
        # Plot average customers during promotion vs non-promotion periods
        plt.figure(figsize=(10,6))
        sns.barplot(x='Promo', y='Customers', data=promo_customers, palette='plasma')
        plt.title('Average Number of Customers: Promo vs No Promo')
        plt.xlabel('Promotion (1 = Yes, 0 = No)')
        plt.ylabel('Average Number of Customers')
        plt.show()
        logging.info("Analyzed and plotted average customers during promo vs non-promo periods.")
    
    # Check how promotions affect existing customers by checking average sales per customer
    if 'Customers' in train_store.columns:
        train_store['Sales_per_Customer'] = train_store['Sales'] / train_store['Customers']
        promo_sales_per_customer = train_store.groupby('Promo')['Sales_per_Customer'].mean().reset_index()
        
        # Plot average sales per customer during promotion vs non-promotion periods
        plt.figure(figsize=(10,6))
        sns.barplot(x='Promo', y='Sales_per_Customer', data=promo_sales_per_customer, palette='coolwarm')
        plt.title('Average Sales per Customer: Promo vs No Promo')
        plt.xlabel('Promotion (1 = Yes, 0 = No)')
        plt.ylabel('Average Sales per Customer')
        plt.show()
        logging.info("Analyzed and plotted average sales per customer during promo vs non-promo periods.")



def analyze_promo_effectiveness_by_store(train_store):
    """
    Analyze how promotions affect sales at the store level to determine which stores should receive promotions.
    """
    # Calculate average sales with and without promotions for each store
    promo_sales_by_store = train_store.groupby(['Store', 'Promo'])['Sales'].mean().unstack()
    
    # Stores with significant difference in sales during promos vs non-promo periods
    promo_sales_by_store['Promo_Difference'] = promo_sales_by_store[1] - promo_sales_by_store[0]
    
    # Identify stores where promotions significantly boost sales
    promo_boost_stores = promo_sales_by_store.sort_values('Promo_Difference', ascending=False).head(10)
    promo_ineffective_stores = promo_sales_by_store.sort_values('Promo_Difference', ascending=True).head(10)

    # Plot stores where promotions have the biggest and smallest effects
    plt.figure(figsize=(12,6))
    sns.barplot(x=promo_boost_stores.index, y=promo_boost_stores['Promo_Difference'], palette='Blues_d')
    plt.title('Top 10 Stores with Most Significant Promo Effect on Sales')
    plt.xlabel('Store')
    plt.ylabel('Sales Difference (Promo vs No Promo)')
    plt.show()
    logging.info("Identified and plotted top 10 stores with most effective promo impact on sales.")

    plt.figure(figsize=(12,6))
    sns.barplot(x=promo_ineffective_stores.index, y=promo_ineffective_stores['Promo_Difference'], palette='Reds_d')
    plt.title('Top 10 Stores with Least Promo Effect on Sales')
    plt.xlabel('Store')
    plt.ylabel('Sales Difference (Promo vs No Promo)')
    plt.show()
    logging.info("Identified and plotted top 10 stores with least effective promo impact on sales.")
    
    # Analyze by store type to see if certain types of stores benefit more from promotions
    store_type_sales = train_store.groupby(['StoreType', 'Promo'])['Sales'].mean().unstack()
    store_type_sales['Promo_Difference'] = store_type_sales[1] - store_type_sales[0]
    
    plt.figure(figsize=(10,6))
    store_type_sales['Promo_Difference'].plot(kind='bar', color='lightgreen')
    plt.title('Promo Effectiveness by Store Type')
    plt.xlabel('Store Type')
    plt.ylabel('Sales Difference (Promo vs No Promo)')
    plt.show()
    logging.info("Analyzed and plotted promo effectiveness by store type.")

    # Analyze effect of competitor distance on promo effectiveness
    if 'CompetitionDistance' in train_store.columns:
        train_store['CompetitionDistance'] = train_store['CompetitionDistance'].fillna(train_store['CompetitionDistance'].mean())
        store_competitor_sales = train_store.groupby(['Store', 'Promo'])['Sales'].mean().unstack()
        store_competitor_sales['CompetitionDistance'] = train_store.groupby('Store')['CompetitionDistance'].mean()

        # Plot sales difference by competitor distance
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=store_competitor_sales['CompetitionDistance'], y=store_competitor_sales[1] - store_competitor_sales[0])
        plt.title('Promo Effectiveness vs. Competitor Distance')
        plt.xlabel('Competitor Distance')
        plt.ylabel('Sales Difference (Promo vs No Promo)')
        plt.show()
        logging.info("Analyzed and plotted relationship between promo effectiveness and competitor distance.")


    
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




def analyze_weekday_store_sales(train_store):
    """
    Analyze stores that are open on all weekdays and how that affects their weekend sales.
    """
    # Group data by Store and DayOfWeek, and check if stores are open on all weekdays (Mon-Fri)
    weekday_data = train_store[(train_store['DayOfWeek'] <= 5) & (train_store['Open'] == 1)]
    stores_open_all_weekdays = weekday_data.groupby('Store').size() == 5  # Stores open all weekdays
    
    # Get the list of stores open all weekdays
    stores_open_all_weekdays = stores_open_all_weekdays[stores_open_all_weekdays].index.tolist()
    
    # Filter weekend data (Saturday and Sunday)
    weekend_data = train_store[(train_store['DayOfWeek'] > 5) & (train_store['Open'] == 1)]
    
    # Separate weekend data into stores that are open on all weekdays and those that aren't
    weekend_data['Store_Open_All_Weekdays'] = weekend_data['Store'].apply(lambda x: 1 if x in stores_open_all_weekdays else 0)
    
    # Compare weekend sales for stores open all weekdays vs. those that are not
    avg_sales_weekend = weekend_data.groupby('Store_Open_All_Weekdays')['Sales'].mean().reset_index()
    
    # Plot average weekend sales
    plt.figure(figsize=(10,6))
    sns.barplot(x='Store_Open_All_Weekdays', y='Sales', data=avg_sales_weekend, palette='Set2')
    plt.title('Average Weekend Sales: Stores Open All Weekdays vs. Others')
    plt.xticks([0, 1], ['Not Open All Weekdays', 'Open All Weekdays'])
    plt.xlabel('Store Status')
    plt.ylabel('Average Weekend Sales')
    plt.show()
    logging.info("Analyzed and plotted weekend sales for stores open all weekdays vs. other stores.")
    
    # Return summary data
    return avg_sales_weekend



def analyze_assortment_type_effect_on_sales(train_store):
    """
    Analyze how the assortment type affects sales across stores.
    """
    # Calculate average sales by assortment type
    assortment_sales = train_store.groupby('Assortment')['Sales'].mean().reset_index()

    # Plot the sales by assortment type
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Assortment', y='Sales', data=assortment_sales, palette='Set2')
    plt.title('Average Sales by Assortment Type')
    plt.xlabel('Assortment Type')
    plt.ylabel('Average Sales')
    plt.show()
    logging.info("Analyzed and plotted sales by assortment type.")
    
    return assortment_sales



def analyze_competition_distance_effect_on_sales(train_store):
    """
    Analyze how the distance to the next competitor affects sales.
    """
    # Check for missing values in critical columns
    if train_store[['Store', 'CompetitionDistance', 'Sales']].isnull().sum().any():
        logging.warning("Missing values found in 'Store', 'CompetitionDistance', or 'Sales' columns.")
        return
    
    # Calculate the correlation between CompetitionDistance and Sales
    correlation = train_store[['CompetitionDistance', 'Sales']].corr().iloc[0, 1]
    logging.info(f"Correlation between Competition Distance and Sales: {correlation:.4f}")
    
    # Plot a scatter plot of CompetitionDistance vs Sales
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=train_store, alpha=0.5)
    plt.title(f'Competition Distance vs Sales (Correlation: {correlation:.4f})')
    plt.xlabel('Competition Distance')
    plt.ylabel('Sales')
    plt.show()
    
    return correlation

def analyze_city_center_store_sales(train_store):
    """
    Analyze if distance matters for stores in city centers and how it affects sales.
    Assumes city center stores are identified by 'StoreType' or other proxies.
    """
    # Define a proxy for city center stores (e.g., StoreType, or stores with high competition presence)
    # Assuming 'StoreType' == 'c' represents city center stores for the sake of this analysis.
    city_center_stores = train_store[train_store['StoreType'] == 'c']
    non_city_center_stores = train_store[train_store['StoreType'] != 'c']
    
    # Calculate the correlation between CompetitionDistance and Sales for city center stores
    city_center_correlation = city_center_stores[['CompetitionDistance', 'Sales']].corr().iloc[0, 1]
    non_city_center_correlation = non_city_center_stores[['CompetitionDistance', 'Sales']].corr().iloc[0, 1]
    
    logging.info(f"Correlation for city center stores: {city_center_correlation:.4f}")
    logging.info(f"Correlation for non-city center stores: {non_city_center_correlation:.4f}")
    
    # Plot scatter plot for city center stores
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=city_center_stores, alpha=0.5)
    plt.title(f'City Center Stores: Competition Distance vs Sales (Correlation: {city_center_correlation:.4f})')
    plt.xlabel('Competition Distance')
    plt.ylabel('Sales')
    plt.show()

    # Plot scatter plot for non-city center stores
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=non_city_center_stores, alpha=0.5)
    plt.title(f'Non-City Center Stores: Competition Distance vs Sales (Correlation: {non_city_center_correlation:.4f})')
    plt.xlabel('Competition Distance')
    plt.ylabel('Sales')
    plt.show()
    
    return city_center_correlation, non_city_center_correlation



def analyze_competition_distance_effect_on_sales(train_store):
    """
    Analyze how the distance to the next competitor affects sales.
    """
    # Calculate the correlation between CompetitionDistance and Sales
    correlation = train_store[['CompetitionDistance', 'Sales']].corr().iloc[0, 1]
    logging.info(f"Correlation between Competition Distance and Sales: {correlation:.4f}")
    
    # Plot a scatter plot of CompetitionDistance vs Sales
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=train_store, alpha=0.5)
    plt.title(f'Competition Distance vs Sales (Correlation: {correlation:.4f})')
    plt.xlabel('Competition Distance')
    plt.ylabel('Sales')
    plt.show()
    
    return correlation

def analyze_city_center_store_sales(train_store):
    """
    Analyze if distance matters for stores in city centers and how it affects sales.
    Assumes city center stores are identified by 'StoreType' or other proxies.
    """
    # Define a proxy for city center stores (e.g., StoreType, or stores with high competition presence)
    # Assuming 'StoreType' == 'c' represents city center stores for the sake of this analysis.
    city_center_stores = train_store[train_store['StoreType'] == 'c']
    non_city_center_stores = train_store[train_store['StoreType'] != 'c']
    
    # Calculate the correlation between CompetitionDistance and Sales for city center stores
    city_center_correlation = city_center_stores[['CompetitionDistance', 'Sales']].corr().iloc[0, 1]
    non_city_center_correlation = non_city_center_stores[['CompetitionDistance', 'Sales']].corr().iloc[0, 1]
    
    logging.info(f"Correlation for city center stores: {city_center_correlation:.4f}")
    logging.info(f"Correlation for non-city center stores: {non_city_center_correlation:.4f}")
    
    # Plot scatter plot for city center stores
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=city_center_stores, alpha=0.5)
    plt.title(f'City Center Stores: Competition Distance vs Sales (Correlation: {city_center_correlation:.4f})')
    plt.xlabel('Competition Distance')
    plt.ylabel('Sales')
    plt.show()

    # Plot scatter plot for non-city center stores
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=non_city_center_stores, alpha=0.5)
    plt.title(f'Non-City Center Stores: Competition Distance vs Sales (Correlation: {non_city_center_correlation:.4f})')
    plt.xlabel('Competition Distance')
    plt.ylabel('Sales')
    plt.show()
    
    return city_center_correlation, non_city_center_correlation



def analyze_competitor_opening_effect(train_store):
    """
    Analyze how the opening or reopening of competitors affects stores by checking stores that initially had no competitors (NA in CompetitionDistance).
    """
    # Find stores with NA in 'CompetitionDistance' at first, but later with a value
    train_store['CompetitionDistance'] = train_store['CompetitionDistance'].fillna(method='ffill') # forward fill for missing values

    stores_with_late_competition = train_store[train_store['CompetitionDistance'].isnull()]
    train_store_with_competition = train_store[train_store['CompetitionDistance'].notnull()]

    # Flag when a competitor appeared (previously NA, now a valid distance)
    stores_with_new_competitors = train_store_with_competition.groupby('Store').filter(
        lambda x: x['CompetitionDistance'].isnull().sum() > 0 and x['CompetitionDistance'].notnull().sum() > 0
    )
    
    logging.info(f"Found {stores_with_new_competitors['Store'].nunique()} stores that had NA competition distance and later had valid competition distance.")

    # Calculate average sales before and after competition appeared
    sales_before_competition = stores_with_new_competitors[stores_with_new_competitors['CompetitionDistance'].isnull()]
    sales_after_competition = stores_with_new_competitors[stores_with_new_competitors['CompetitionDistance'].notnull()]
    
    avg_sales_before = sales_before_competition.groupby('Store')['Sales'].mean().reset_index()
    avg_sales_after = sales_after_competition.groupby('Store')['Sales'].mean().reset_index()

    # Merge and calculate the percentage change in sales
    sales_comparison = pd.merge(avg_sales_before, avg_sales_after, on='Store', suffixes=('_before', '_after'))
    sales_comparison['Sales_Change_Percent'] = 100 * (sales_comparison['Sales_after'] - sales_comparison['Sales_before']) / sales_comparison['Sales_before']

    # Plot the percentage change in sales due to competition
    plt.figure(figsize=(12, 6))
    sns.histplot(sales_comparison['Sales_Change_Percent'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Sales Percentage Change After Competitor Entry')
    plt.xlabel('Sales Change (%)')
    plt.ylabel('Frequency')
    plt.show()

    logging.info("Analyzed and plotted sales change due to competitor opening or reopening.")
    
    return sales_comparison



def plot_sales_trends_around_competitor_opening(train_store):
    """
    Visualize sales trends for stores before and after a new competitor opened.
    """
    # Filter stores with competitors that opened or reopened
    stores_with_new_competitors = train_store.groupby('Store').filter(
        lambda x: x['CompetitionDistance'].isnull().sum() > 0 and x['CompetitionDistance'].notnull().sum() > 0
    )
    
    if stores_with_new_competitors.empty:
        print("No stores found with new competitor openings.")
        return

    plt.figure(figsize=(14, 8))
    
    # Loop through stores and plot sales trends for each store with new competitor
    for store_id in stores_with_new_competitors['Store'].unique():
        store_sales = stores_with_new_competitors[stores_with_new_competitors['Store'] == store_id]
        store_sales = store_sales.sort_values('Date')
        
        if not store_sales.empty:
            plt.plot(store_sales['Date'], store_sales['Sales'], label=f'Store {store_id}')
        
    # Highlight when a competitor appeared for any store
    competitor_entry_date = stores_with_new_competitors[stores_with_new_competitors['CompetitionDistance'].notnull()]['Date'].min()
    
    if pd.notnull(competitor_entry_date):
        plt.axvline(x=competitor_entry_date, color='red', linestyle='--', label='Competitor Entry')
    
    plt.title('Sales Trends for Stores with Competitor Entry')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

    logging.info("Plotted sales trends around competitor opening or reopening.")

    

