import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle

# Load the dataset
data = pd.read_csv("datasets/crop_data.csv")

# Print the columns to debug
print("Columns in the dataset:", data.columns.tolist())

# Define features (X) and target (y)
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']  # Updated to 'label' based on the dataset

# Encode the target variable (crop names to integers)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = XGBClassifier()
model.fit(X_train, y_train)

# Save the model and label encoder
with open("model/crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Crop recommendation model trained and saved successfully.")
