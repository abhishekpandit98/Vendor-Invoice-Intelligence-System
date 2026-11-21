import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]

def load_artifacts():
    """Helper to load the model and scaler once."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None

# Load them globally so they stay in memory
MODEL, SCALER = load_artifacts()

def predict_invoice_flag(input_data):
    if MODEL is None or SCALER is None:
        raise RuntimeError(
            f"Could not load prediction artifacts from {MODEL_PATH} and {SCALER_PATH}"
        )

    input_df = pd.DataFrame(input_data)
    missing_features = [feature for feature in FEATURES if feature not in input_df]
    if missing_features:
        raise ValueError(f"Missing invoice features: {', '.join(missing_features)}")

    input_ready = input_df[FEATURES].apply(pd.to_numeric, errors="raise")
    input_ready = input_ready.astype("float64")
    input_scaled = SCALER.transform(input_ready)
    input_df["Predicted_Flag"] = MODEL.predict(input_scaled).astype(int)
    return input_df

if __name__ == "__main__":
    # Test data - exactly 5 features
    sample_data = {
        "invoice_quantity": [50],
        "invoice_dollars": [1200.0],
        "Freight": [45.0],
        "total_item_quantity": [50],
        "total_item_dollars": [1195.0]
    }

    print("\n--- Running Local Terminal Test ---")
    result = predict_invoice_flag(sample_data)
    
    print("SUCCESS!")
    print(result[["invoice_dollars", "total_item_dollars", "Predicted_Flag"]])

    flag_val = result["Predicted_Flag"].iloc[0]
    status = "HIGH RISK" if flag_val == 1 else "LOW RISK"
    print(f"\nResult: {status}")