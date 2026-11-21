import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "freight_cost_prediction" / "models" / "predict_freight_model.pkl"

def load_model(model_path: str = MODEL_PATH):
    if not Path(model_path).exists():
        print(f"Error: Model file not found at {model_path}")
        return None
    try:
        # Using a direct load to handle the version warning gracefully
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_freight_cost(input_data):
    model = load_model()
    if model is None:
        return None
        
    input_df = pd.DataFrame(input_data)

    try:
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is not None:
            missing_features = [
                feature for feature in feature_names if feature not in input_df
            ]
            if missing_features:
                print(f"Missing freight features: {', '.join(missing_features)}")
                return None
            data_values = input_df[list(feature_names)].values
        elif getattr(model, "n_features_in_", 1) == 1:
            if "Dollars" not in input_df:
                print("Missing freight feature: Dollars")
                return None
            data_values = input_df[["Dollars"]].values
        else:
            missing_features = [
                feature for feature in ("Quantity", "Dollars")
                if feature not in input_df
            ]
            if missing_features:
                print(f"Missing freight features: {', '.join(missing_features)}")
                return None
            data_values = input_df[["Quantity", "Dollars"]].values

        predictions = model.predict(data_values)
        input_df['Predicted_Freight'] = np.array(predictions).flatten().round(2)
        return input_df

    except Exception as e:
        print(f"Prediction Error: {e}")
        return None

if __name__ == "__main__":
    # Test it locally with the single feature logic
    sample_data = {"Quantity": [100], "Dollars": [5000]}
    result = predict_freight_cost(sample_data)
    if result is not None:
        print("\n--- Success! Prediction Match Found ---")
        print(result)