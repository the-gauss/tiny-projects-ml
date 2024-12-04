from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam

# Define a function to build the SimpleRNN model
def build_simple_rnn_model(input_shape, output_dim, units=64, learning_rate=0.001):
    """
    Builds a SimpleRNN model.

    Parameters:
    - input_shape: Shape of the input data (time_steps, features)
    - output_dim: Number of output columns (target variables)
    - units: Number of units in the RNN layer
    - learning_rate: Learning rate for the optimizer

    Returns:
    - model: Compiled Keras model
    """
    model = Sequential([
        Input(shape=input_shape),
        SimpleRNN(units, activation='tanh', return_sequences=False),
        Dense(output_dim)  # Predict all columns simultaneously
    ])

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    return model


# Define a function to build the Bidirectional LSTM model
def build_bi_lstm_model(input_shape, output_dim, units=64, learning_rate=0.001):
    """
    Builds a Bidirectional LSTM model.

    Parameters:
    - input_shape: Shape of the input data (time_steps, features)
    - output_dim: Number of output columns (target variables)
    - units: Number of units in the LSTM layer
    - learning_rate: Learning rate for the optimizer

    Returns:
    - model: Compiled Keras model
    """
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(units, activation='tanh', return_sequences=False)),
        Dense(output_dim)  # Predict all columns simultaneously
    ])

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    return model


# Define a function to build the Conv1D model
def build_conv1d_model(input_shape, output_dim, filters=64, kernel_size=3, learning_rate=0.001):
    """
    Builds a Conv1D model.

    Parameters:
    - input_shape: Shape of the input data (time_steps, features)
    - output_dim: Number of output columns (target variables)
    - filters: Number of filters for Conv1D layer
    - kernel_size: Size of the convolution kernel
    - learning_rate: Learning rate for the optimizer

    Returns:
    - model: Compiled Keras model
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'),
        Flatten(),
        Dense(output_dim)  # Predict all columns simultaneously
    ])

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    return model

# Calculate RMSE ratios for Conv1D, Bi-LSTM, and SimpleRNN models
def calculate_rmse_ratios(model, X_test, y_test):
    """
    Calculate the RMSE ratio for each target attribute based on its mean.

    Parameters:
    - model: Trained Keras model
    - X_test: Test features
    - y_test: Test targets (scaled)

    Returns:
    - rmse_ratios: Dictionary with RMSE ratios for each attribute
    """
    predictions = model.predict(X_test)
    rmse_per_attribute = np.sqrt(np.mean((predictions - y_test) ** 2, axis=0))
    mean_per_attribute = np.mean(y_test, axis=0)
    rmse_ratios = rmse_per_attribute / mean_per_attribute
    return rmse_ratios
