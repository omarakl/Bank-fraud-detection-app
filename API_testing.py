import requests

# Define test data
test_transaction = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 200,
    "nameOrig": "C12345",
    "oldbalanceOrig": 200,
    "newbalanceOrig": 0,
    "nameDest": "M67890",
    "oldBalanceDest": 0,
    "newBalanceDest": 400
}

# Send request to API
response = requests.post("http://127.0.0.1:8000/predict", json=test_transaction)

# Print response
print(response.json())