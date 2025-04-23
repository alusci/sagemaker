import os
import json
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Load data
model_dir = "/opt/ml/processing/model"
validation_data_path = "/opt/ml/processing/validation/validation.csv"
evaluation_output_path = "/opt/ml/processing/evaluation/evaluation.json"

# Load validation set
df = pd.read_csv(validation_data_path)

# Predict (assuming predictions are already computed for simplicity)
# Replace this with actual prediction logic if needed
# For now, assume predictions are in a column named "prediction"
y_true = df["label"]
y_pred = df["prediction"]

# Compute MAE
mae = mean_absolute_error(y_true, y_pred)

# Save evaluation report
os.makedirs(os.path.dirname(evaluation_output_path), exist_ok=True)
with open(evaluation_output_path, "w") as f:
    json.dump({"mae": mae}, f)


