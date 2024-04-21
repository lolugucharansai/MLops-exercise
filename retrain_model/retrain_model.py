import argparse
import glob
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--diabetes-csv', type=str, help='Path to the diabetes CSV file')
args = parser.parse_args()

# Read data
data_path = args.diabetes_csv
all_files = glob.glob(data_path + "/*.csv")
data = pd.concat((pd.read_csv(f) for f in all_files), sort=False)
X, y = data[
        [
            'Pregnancies',
            'PlasmaGlucose',
            'DiastolicBloodPressure',
            'TricepsThickness',
            'SerumInsulin',
            'BMI',
            'DiabetesPedigree',
            'Age'
        ]
    ].values, data['Diabetic'].values

    # train/test split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0
    )

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