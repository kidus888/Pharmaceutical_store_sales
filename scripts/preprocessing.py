
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_merge_data(train_file, store_file):
    """
    Load train and store datasets and merge them on the 'Store' column.
    """
    train_df = pd.read_csv(train_file, parse_dates=['Date'])
    store_df = pd.read_csv(store_file)
    merged_df = pd.merge(train_df, store_df, on='Store', how='left')
    
    return merged_df

def preprocess_data(merged_df):
    """
    Perform data preprocessing: feature engineering, handling missing values,
    encoding, and scaling.
    """
    # Fill missing values
    merged_df['CompetitionDistance'].fillna(merged_df['CompetitionDistance'].max(), inplace=True)
    merged_df['CompetitionOpenSinceMonth'].fillna(merged_df['CompetitionOpenSinceMonth'].mode()[0], inplace=True)
    merged_df['CompetitionOpenSinceYear'].fillna(merged_df['CompetitionOpenSinceYear'].mode()[0], inplace=True)
    merged_df['Promo2SinceWeek'].fillna(0, inplace=True)
    merged_df['Promo2SinceYear'].fillna(0, inplace=True)
    merged_df['PromoInterval'].fillna('None', inplace=True)

    # Feature Engineering: Extract date-based features from the Date column
    merged_df['Year'] = merged_df['Date'].dt.year
    merged_df['Month'] = merged_df['Date'].dt.month
    merged_df['Day'] = merged_df['Date'].dt.day
    merged_df['WeekOfYear'] = merged_df['Date'].dt.isocalendar().week
    merged_df['IsWeekend'] = merged_df['DayOfWeek'].apply(lambda x: 1 if x >= 6 else 0)
    merged_df['IsMonthStart'] = merged_df['Date'].dt.is_month_start.astype(int)
    merged_df['IsMonthEnd'] = merged_df['Date'].dt.is_month_end.astype(int)

    # Additional feature: Days since competition started (if available)
    merged_df['CompetitionOpenSince'] = (
        (merged_df['Year'] - merged_df['CompetitionOpenSinceYear']) * 12 +
        (merged_df['Month'] - merged_df['CompetitionOpenSinceMonth'])
    ).fillna(0)

    # Convert categorical columns to numeric using one-hot encoding
    merged_df = pd.get_dummies(merged_df, columns=['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'], drop_first=True)

    # Drop unnecessary columns
    merged_df.drop(['Date', 'Customers'], axis=1, inplace=True)

    # Prepare features (X) and target (y)
    X = merged_df.drop(['Sales'], axis=1)
    y = merged_df['Sales']

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
