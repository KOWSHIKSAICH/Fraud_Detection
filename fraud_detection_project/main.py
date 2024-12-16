# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import pickle
from flask import Flask, request, render_template, jsonify

# Load dataset
data = pd.read_csv("dataset.csv")

# Data preprocessing
# Drop unnecessary columns
data.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

# One-hot encode 'type' column
data = pd.get_dummies(data, columns=['type'], drop_first=True)

# Handle class imbalance
fraud = data[data['isFraud'] == 1]
non_fraud = data[data['isFraud'] == 0]
non_fraud_downsampled = resample(non_fraud, replace=False, n_samples=len(fraud), random_state=42)
data_balanced = pd.concat([fraud, non_fraud_downsampled])

# Feature and target split
X = data_balanced.drop(['isFraud'], axis=1)
y = data_balanced['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
with open('models/fraud_detection_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Evaluate model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array([list(data.values())])
        features = scaler.transform(features)
        prediction = model.predict(features)[0]
        return jsonify({"isFraud": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
