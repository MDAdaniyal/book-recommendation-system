import os
import subprocess
from pathlib import Path

# === CONFIG ===
KFOLD_DATA_PATH = os.path.join("..", "..", "data", "kfold_data")
TAWA_SCRIPT = os.path.join("..", "..", "src", "tawa", "run_ident.sh")
RESULTS_DIR = os.path.join("..", "..", "results", "tawa")

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# === Train TAWA Models ===
def train_tawa_for_all_users():
    user_dirs = [d for d in os.listdir(KFOLD_DATA_PATH) if os.path.isdir(os.path.join(KFOLD_DATA_PATH, d))]

    for user_id in user_dirs:
        print(f"Training TAWA for user {user_id}...")
        user_dir = os.path.join(KFOLD_DATA_PATH, user_id)

        for i in range(5):
            train_file = os.path.join(user_dir, f"train_fold{i}.txt")
            test_file = os.path.join(user_dir, f"test_fold{i}.txt")
            output_file = os.path.join(RESULTS_DIR, f"tawa_user{user_id}_fold{i}.txt")

            cmd = ["bash", TAWA_SCRIPT, train_file, test_file, output_file]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error while training fold {i} for user {user_id}: {e}")

if __name__ == "__main__":
    train_tawa_for_all_users()
    print("✓ All TAWA models trained.")
