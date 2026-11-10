
import csv
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier

# Class names and numeric labels
class_names = {
    1: "Built-up",
    2: "Agriculture",
    3: "Vegetation",
    4: "Water",
    5: "Barren"
}

feature_columns = [
    "Blue", "Green", "Red", "NIR", "NDVI", "NDWI"
]

# 1. Read training samples

X = []
y = []

with open("data/samples/training_samples.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        features = [
            float(row[column])
            for column in feature_columns
        ]

        X.append(features)

        label = row["label"]

        for class_id, class_name in class_names.items():
            if label == class_name:
                y.append(class_id)
                break

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("Training samples:", len(X))
print("Feature count:", X.shape[1])

# 2. Train Random Forest

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    oob_score=True
)

model.fit(X, y)

print("\nModel trained successfully!")
print("OOB score:", model.oob_score_)

# 3. Save trained model

joblib.dump(model, "models/lulc_random_forest.pkl")

print("\nModel saved successfully!")
print("Location: models/lulc_random_forest.pkl")

print("\nClass IDs:")

for class_id, name in class_names.items():
    print(class_id, "=", name)
