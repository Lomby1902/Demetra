import requests
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
path = 'datasets/LEAF_LEVEL_DATASET_Yufeng_Ge.xlsx'

# Initialize the MLflow client
client = MlflowClient()

print("Registered models:")
# Fetch all registered models
registered_models = client.search_registered_models()

# Display the names of all registered models
for i,model in enumerate(registered_models):
    print(i,")", model.name)
choice = int(input("Selected a model: "))
model = registered_models[choice].name
print("Selected model ", model)
print("Reading Dataset...")
# Read and load dataset
X= pd.read_csv("data.csv")
X = (X.iloc[:1,1:]).values

server_url = 'http://127.0.0.1:5001/response'

# Prepare the data to be sent
data = {"model":model,"values": X.tolist()}

# Send a GET request
response = requests.get(server_url, json=data)

# Print the server's response
if response.status_code == 200:
    print(f"Nitrogen: {response.json()['message']}")
else:
    print("Failed to connect to the server.")
