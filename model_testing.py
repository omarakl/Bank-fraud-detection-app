import pandas as pd
import tensorflow as tf
import pickle
import numpy as np

# Load the saved scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Example input data
input_data = pd.DataFrame({
        'step': [1],
        'type': ['TRANSFER'],
        'amount': [300],
        'nameOrig': ['C1579132337'],
        'oldbalanceOrg': [1000],
        'newbalanceOrig': [700],
        'nameDest': ['C2354792876'],
        'oldbalanceDest': [300],
        'newbalanceDest': [600]
})

# Manually encoding categorical columns
def encode_column(data, column):
    """ Manually encode categorical columns without using transform() """
    categories = data[column].unique()  # Get unique categories
    category_dict = {category: idx for idx, category in enumerate(categories)}  # Assign each category an integer index
    data[column] = data[column].map(category_dict)  # Map the category to its integer index
    return data

# Preprocessing function
def preprocess_input(data, scaler):
    data = data.copy()  # Avoid modifying the original DataFrame

    # Manually encode 'type', 'nameOrig', and 'nameDest' columns
    data = encode_column(data, 'type')
    data = encode_column(data, 'nameOrig')
    data = encode_column(data, 'nameDest')

    # Normalize numerical features using the preloaded scaler
    numerical_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    return data

# Preprocess the input data
X_input = preprocess_input(input_data, scaler)

# Reshape input data to match LSTM model's expected shape (batch_size, sequence_length, num_features)
sequence_length = 1  # Adjust based on your model's expected input
num_features = X_input.shape[1]  # Number of features
X_input_reshaped = np.reshape(X_input.values, (X_input.shape[0], sequence_length, num_features))

# Load the trained LSTM model
model = tf.keras.models.load_model('models/fraud_detection_lstm.keras')

# Make predictions
predictions = model.predict(X_input_reshaped)
predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions (0 or 1)

# Add predictions to the input data
input_data['isFraud'] = predictions

# Display predictions
print("\nPredictions:")
print(input_data[['step', 'type', 'amount', 'isFraud']])
