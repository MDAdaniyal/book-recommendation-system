import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === CONFIG ===
RESULTS_PATH = os.path.join("..", "..", "results")
TRAD_RESULTS_FILE = os.path.join(RESULTS_PATH, "TRAD_results.csv")

# === Load predictions ===
def load_results(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

# === Evaluate metrics ===
def evaluate(df, model_name):
    if df.empty or not {"actual", "predicted"}.issubset(df.columns):
        print(f"Skipping {model_name}: Missing data.")
        return None

    mae = mean_absolute_error(df["actual"], df["predicted"])
    rmse = mean_squared_error(df["actual"], df["predicted"], squared=False)
    print(f"{model_name} → MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return {"Model": model_name, "MAE": mae, "RMSE": rmse}

# === Main ===
if __name__ == "__main__":
    result_list = []

    # Evaluate Traditional CF/MF Results
    trad_df = load_results(TRAD_RESULTS_FILE)
    if not trad_df.empty:
        if "Model" in trad_df.columns:
            grouped = trad_df.groupby("Model")
            for model_name, group in grouped:
                result = evaluate(group, model_name)
                if result:
                    result_list.append(result)
        else:
            result = evaluate(trad_df, "Traditional")
            if result:
                result_list.append(result)

    # Output summary
    if result_list:
        summary_df = pd.DataFrame(result_list)
        print("\n--- Evaluation Summary ---")
        print(summary_df)
    else:
        print("No results to evaluate.")
