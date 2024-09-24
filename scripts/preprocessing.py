import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def load_data(train_path, store_path):
    train = pd.read_csv(train_path)
    store = pd.read_csv(store_path)
    df = pd.merge(train, store, on='Store', how='left')
    return df


def preprocess_data(df):
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract date-related features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    # Feature: Beginning, Mid, End of Month
    df['IsBeginningOfMonth'] = df['Day'].apply(lambda x: 1 if x <= 10 else 0)
    df['IsMidMonth'] = df['Day'].apply(lambda x: 1 if 10 < x <= 20 else 0)
    df['IsEndOfMonth'] = df['Day'].apply(lambda x: 1 if x > 20 else 0)

    # Encode categorical variables
    df['StateHoliday'] = df['StateHoliday'].replace({'a': 1, 'b': 2, 'c': 3, '0': 0})
    df['StoreType'] = df['StoreType'].map({'a': 1, 'b': 2, 'c': 3, 'd': 4})
    df['Assortment'] = df['Assortment'].map({'a': 1, 'b': 2, 'c': 3})

    # Handle 'PromoInterval' - Convert it to binary columns indicating the months
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

    for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
        df[f'Promo_{month}'] = df['PromoInterval'].apply(lambda x: 1 if isinstance(x, str) and month in x else 0)

    # Fill missing values
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
    df['CompetitionOpenSinceYear'].fillna(df['CompetitionOpenSinceYear'].mode()[0], inplace=True)
    df['Promo2SinceYear'].fillna(df['Promo2SinceYear'].mode()[0], inplace=True)
    df.fillna(0, inplace=True)  # Fill remaining NaNs with 0

    # Feature engineering
    df['CompetitionOpenTime'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + (df['Month'] - df['CompetitionOpenSinceMonth'])
    df['Promo2Time'] = 12 * (df['Year'] - df['Promo2SinceYear']) + (df['WeekOfYear'] - df['Promo2SinceWeek'])

    # Drop unneeded columns
    df.drop(columns=['Date', 'PromoInterval'], inplace=True)

    return df



def scale_data(df, target_column='Sales'):
    features = df.drop(columns=[target_column])
    target = df[target_column]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, target


def build_pipeline():
    # Create a pipeline with scaling and a random forest model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    return pipeline

def train_and_evaluate(df, target_column='Sales'):
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get the pipeline
    pipeline = build_pipeline()
    
    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Root Mean Squared Error: {rmse}")
    
    return pipeline

def train_and_evaluate_model(df_processed):
    # Splitting data into train and test sets
    X = df_processed.drop(columns=['Sales'])
    y = df_processed['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Building a pipeline with Random Forest Regressor
    pipeline = Pipeline([
        ('regressor', RandomForestRegressor(criterion='absolute_error', random_state=42))
    ])

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    
    return mae


def get_feature_importance(model, feature_names):
    """
    Get feature importance from the trained RandomForest model.
    """
    importances = model.named_steps['regressor'].feature_importances_
    feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return feature_importance

def estimate_confidence_interval(model, X_test, y_test, n_iterations=1000, alpha=0.95):
    """
    Estimate confidence intervals using bootstrapping.
    """
    predictions = []
    n_size = X_test.shape[0]
    
    for _ in range(n_iterations):
        # Resample the test data
        X_resampled, y_resampled = resample(X_test, y_test, n_samples=n_size, random_state=42)
        
        # Predict using the model
        y_pred_resampled = model.predict(X_resampled)
        predictions.append(y_pred_resampled)
    
    # Convert predictions to numpy array for easy calculations
    predictions = np.array(predictions)
    
    # Calculate mean and confidence intervals
    lower_bound = np.percentile(predictions, (1 - alpha) / 2 * 100, axis=0)
    upper_bound = np.percentile(predictions, (1 + alpha) / 2 * 100, axis=0)
    
    return lower_bound, upper_bound

def post_prediction_analysis(df_processed, model):
    """
    Performs post-prediction analysis including feature importance and confidence interval estimation.
    """
    # Splitting data into test set
    X_test = df_processed.drop(columns=['Sales'])
    y_test = df_processed['Sales']

    # Feature importance
    feature_names = X_test.columns
    feature_importance = get_feature_importance(model, feature_names)
    
    print("Feature Importance Ranking:")
    for feature, importance in feature_importance:
        print(f"{feature}: {importance}")
    
    # Confidence interval estimation
    lower_bound, upper_bound = estimate_confidence_interval(model, X_test, y_test)
    
    print(f"Lower Bound of Predictions: {lower_bound}")
    print(f"Upper Bound of Predictions: {upper_bound}")
    
    return feature_importance, lower_bound, upper_bound


def serialize_model(model, folder_path='models/'):
    """
    Serialize and save the model with a timestamped filename.
    """
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Generate the timestamped filename
    timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S-%f')
    filename = os.path.join(folder_path, f'{timestamp}.pkl')
    
    # Save the model using joblib
    joblib.dump(model, filename)
    
    print(f"Model serialized and saved as {filename}")
    return filename
