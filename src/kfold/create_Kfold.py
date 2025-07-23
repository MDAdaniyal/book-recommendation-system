import os
import pandas as pd
from sklearn.model_selection import KFold
from pathlib import Path

# === CONFIG ===
BOOKS_FILTERED_PATH = os.path.join("..", "..", "data", "processed", "BooksFiltered.csv")
FINAL_BOOKS_DIR = os.path.join("..", "..", "data", "final_books")
KFOLD_OUTPUT_DIR = os.path.join("..", "..", "data", "kfold_data")
NUM_FOLDS = 5

# === Ensure output folder exists ===
Path(KFOLD_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# === Load filtered data ===
print("Loading BooksFiltered.csv...")
books_df = pd.read_csv(BOOKS_FILTERED_PATH)

# === Process each user ===
print("Creating k-fold data for users...")
user_ids = books_df["User-ID"].unique()
for user_id in user_ids:
    user_books = books_df[books_df["User-ID"] == user_id].drop_duplicates(subset="Book-Title")
    book_titles = user_books["Book-Title"].tolist()

    if len(book_titles) < NUM_FOLDS:
        continue

    user_folder = os.path.join(KFOLD_OUTPUT_DIR, str(user_id))
    Path(user_folder).mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    book_paths = [
        os.path.join(FINAL_BOOKS_DIR, title.replace(" ", "_").lower() + ".txt")
        for title in book_titles
    ]

    book_texts = []
    for path in book_paths:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                book_texts.append(f.read().strip())
        except FileNotFoundError:
            book_texts.append("")

    book_data = list(zip(book_titles, book_texts))

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(book_data)):
        train_text = "\n\n".join(book_data[i][1] for i in train_idx if book_data[i][1])
        test_text = "\n\n".join(book_data[i][1] for i in test_idx if book_data[i][1])

        with open(os.path.join(user_folder, f"train_fold{fold_idx}.txt"), "w", encoding="utf-8") as f_train:
            f_train.write(train_text)

        with open(os.path.join(user_folder, f"test_fold{fold_idx}.txt"), "w", encoding="utf-8") as f_test:
            f_test.write(test_text)

print("✓ K-fold user data created.")
