import pickle
import uvicorn
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model (.h5 format)
model = tf.keras.models.load_model("models/fraud_detection_lstm.h5")

# Load the label encoders
with open("models/label_encoders.pkl", "rb") as le_file:
    label_encoders = pickle.load(le_file)

# Initialize FastAPI app
app = FastAPI()

# Define request data model
class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrig: float
    newbalanceOrig: float
    nameDest: str
    oldBalanceDest: float
    newBalanceDest: float

# Function to preprocess data
def preprocess(data: Transaction):
    # Convert to dictionary
    data_dict = data.dict()

    # Convert to DataFrame
    df = pd.DataFrame([data_dict])

    # Encode categorical features
    for col in ["type", "nameOrig", "nameDest"]:
        if col in label_encoders:
            df[col] = df[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

    # Convert to NumPy array and reshape for TensorFlow
    features = np.array(df.values, dtype=np.float32)

    # Reshape to (batch_size, time_step, features) -> (1, 1, 9) if model expects 3D input
    return features.reshape(1, 1, -1)  # Reshape for LSTM/RNN models


# Define prediction endpoint
@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Preprocess input data
        features = preprocess(transaction)

        # Make prediction
        prediction = model.predict(features)

        # Convert prediction to binary output (assuming sigmoid activation in the last layer)
        result = "Fraud" if prediction[0][0] > 0.5 else "Not Fraud"

        return {"prediction": result, "raw_score": float(prediction[0][0])}
    
    except Exception as e:
        return {"error": str(e)}

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
