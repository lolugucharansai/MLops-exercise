import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--diabetes-csv', type=str, help='Path to the diabetes CSV file')
args = parser.parse_args()

# Load the diabetes data
data = pd.read_csv(args.diabetes_csv)
X, y = data.drop('Outcome', axis=1), data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the model from the .pkl file
with open('./outputs/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Continue training the model
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f'Model score: {score}')

# Save the retrained model to a local file
with open('./outputs/model.pkl', 'wb') as file:
    pickle.dump(model, file)