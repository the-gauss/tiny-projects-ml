import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def parse_and_normalize_data(file_path, date_format='%d-%m-%Y'):
    """
    Load and preprocess the dataset: parse dates and normalize features.

    Parameters:
    - file_path: Path to the CSV file.
    - date_format: Format of the date column in the dataset.

    Returns:
    - scaled_data: Normalized data as a NumPy array.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Parse the 'Date' column
    df['Date'] = pd.to_datetime(df['Date'], format=date_format)
    df['Date'] = df['Date'].map(lambda x: x.toordinal())  # Convert to numerical format

    # Normalize all features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    return scaled_data


def create_sequences_multivariate(data, lookback):
    """
    Create sequences of features and targets for multivariate time series.

    Parameters:
    - data: Normalized dataset as a NumPy array.
    - lookback: Number of timesteps to include in each sequence.

    Returns:
    - feature_sequences: Feature sequences as a NumPy array.
    - target_sequences: Target sequences as a NumPy array.
    """
    feature_sequences = []
    target_sequences = []
    for i in range(lookback, len(data)):
        feature_sequences.append(data[i - lookback:i, :])  # Feature sequences
        target_sequences.append(data[i, :])  # Predict the next step for all columns
    return np.array(feature_sequences), np.array(target_sequences)


def split_data(X, y, test_size=0.3, val_size=0.5, shuffle=False):
    """
    Split data into training, validation, and test sets.

    Parameters:
    - X: Feature sequences.
    - y: Target sequences.
    - test_size: Proportion of data to include in the test set.
    - val_size: Proportion of test data to include in the validation set.
    - shuffle: Whether to shuffle the data before splitting.

    Returns:
    - X_train, X_val, X_test: Split feature sets.
    - y_train, y_val, y_test: Split target sets.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, shuffle=shuffle)
    return X_train, X_val, X_test, y_train, y_val, y_test


def main_load_prep():
    '''
    Do not use, only for backup/future purposes.
    '''
    # File path to the dataset
    data_path = 'data/MSFT.csv'

    # Parameters
    lookback = 75

    # Parse and normalize data
    scaled_data = parse_and_normalize_data(data_path)

    # Create sequences for features and targets
    X, y = create_sequences_multivariate(scaled_data, lookback)

    # Split into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Print data shapes
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

def plot_candlestick_close_intermittent(data, date_col, close_col, interval=50):
    """
    Plot a simplified candlestick-style chart based on only the 'Close' prices, with intermittent points.

    Parameters:
    - data: DataFrame containing stock price data
    - date_col: Column name for dates
    - close_col: Column name for close prices
    - interval: Interval for showing data points (e.g., every 5th day)
    """
    # Ensure the date column is in datetime format
    data[date_col] = pd.to_datetime(data[date_col])

    # Filter the data to show only intermittent points
    filtered_data = data.iloc[::interval].copy()

    # Determine up (green) or down (red) days
    filtered_data['Previous_Close'] = filtered_data[close_col].shift(1)
    filtered_data['Change'] = filtered_data[close_col] - filtered_data['Previous_Close']
    filtered_data['Color'] = filtered_data['Change'].apply(lambda x: 'green' if x >= 0 else 'red')

    # Plot the candlestick-style chart
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, row in filtered_data.iterrows():
        color = row['Color']
        if pd.notna(row['Previous_Close']):
            plt.plot([row[date_col], row[date_col]], [row['Previous_Close'], row[close_col]], color=color, lw=2)

    # Add scatter points for clarity
    plt.scatter(filtered_data[date_col], filtered_data[close_col], c=filtered_data['Color'], label='Close', edgecolor='k')

    # Add labels, title, and grid
    plt.title(f'Candlestick-Style Chart for Close Prices (Interval: {interval} Days)')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.5)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_model_comparison(models, rmse_values, mse_values):
    """
    Plot a bar chart comparing RMSE and MSE values for different models.

    Parameters:
    - models: List of model names (e.g., ['SimpleRNN', 'Conv1D', 'Bi-LSTM'])
    - rmse_values: List of RMSE values corresponding to the models
    - mse_values: List of MSE values corresponding to the models
    """
    plt.figure(figsize=(10, 6))

    # Plot RMSE
    plt.bar(models, rmse_values, alpha=0.7, label='Test RMSE', width=0.4, align='center')

    # Overlay MSE on the same chart
    plt.bar(models, mse_values, alpha=0.7, label='Test MSE', width=0.4, align='edge')

    # Add labels, title, and legend
    plt.xlabel('Model')
    plt.ylabel('Error')
    plt.title('Model Comparison: Test RMSE and MSE')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    plt.tight_layout()
    plt.show()


# Define a function to calculate RMSE and plot learning curves for it
def plot_rmse_learning_curves(history, title='Learning Curves - RMSE', figsize=(12, 6)):
    """
    Plots the RMSE learning curves for training and validation.

    Parameters:
    - history: Training history object from the Keras model
    - title: Title of the plot
    - figsize: Figure size
    """
    # Calculate RMSE from MSE
    train_rmse = np.sqrt(history.history['mse'])
    val_rmse = np.sqrt(history.history['val_mse'])

    plt.figure(figsize=figsize)
    sns.lineplot(data=train_rmse, label='Training RMSE', linewidth=2, color='green')
    sns.lineplot(data=val_rmse, label='Validation RMSE', linewidth=2, linestyle="--", color='red')
    plt.title(title, fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.show()