import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from azure.storage.blob import BlobServiceClient

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--diabetes-csv', type=str, help='Path to the diabetes CSV file')
args = parser.parse_args()

# Load the prod-diabetes data
data = pd.read_csv(args.diabetes_csv)
X, y = data.drop('Outcome', axis=1), data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a blob service client
blob_service_client = BlobServiceClient.from_connection_string("<your_connection_string>")

# Get the blob client for the model
blob_client = blob_service_client.get_blob_client("<your_container_name>", "model.pkl")

# Download the model to a local file
with open("model.pkl", "wb") as download_file:
    download_file.write(blob_client.download_blob().readall())

# Load the model from the .pkl file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Continue training the model
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f'Model score: {score}')

# Save the retrained model to a local file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Upload the retrained model to Azure Blob Storage
with open('model.pkl', 'rb') as file:
    blob_client.upload_blob(file, overwrite=True)