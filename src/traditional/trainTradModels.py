import os
import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD, KNNBasic

# === CONFIG ===
KFOLD_PATH = os.path.join("..", "..", "data", "kfold_data")
FINAL_BOOKS_PATH = os.path.join("..", "..", "data", "final_books")
MATCHED_RATINGS_PATH = os.path.join("..", "..", "data", "processed", "BooksFiltered.csv")
OUTPUT_MODEL_PATH = os.path.join("..", "..", "results", "traditional")
Path(OUTPUT_MODEL_PATH).mkdir(parents=True, exist_ok=True)

# === Load book content ===
def load_book_content(book_title):
    path = os.path.join(FINAL_BOOKS_PATH, book_title.replace(" ", "_").lower() + ".txt")
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            return file.read()
    except FileNotFoundError:
        return None

# === Calculate similarity and assign ratings ===
def assign_ratings_from_text_similarity(train_folds, matched_ratings):
    ratings_data = []
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    for fold_path in train_folds:
        if not os.path.exists(fold_path):
            continue

        with open(fold_path, "r", encoding="utf-8", errors="ignore") as f:
            fold_content = f.read()

        book_contents = []
        book_info = []
        for _, row in matched_ratings.iterrows():
            content = load_book_content(row["Book-Title"])
            if content:
                book_contents.append(content)
                book_info.append(row)

        if not book_contents:
            continue

        combined = [fold_content] + book_contents
        tfidf_matrix = tfidf_vectorizer.fit_transform(combined)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        for idx, sim in enumerate(similarities):
            if sim > 0.1:
                row = book_info[idx]
                ratings_data.append({
                    "User-ID": row["User-ID"],
                    "Book-Title": row["Book-Title"],
                    "Book-Rating": row["Book-Rating"] * sim
                })

    return pd.DataFrame(ratings_data)

# === Prepare training data ===
def prepare_training_data(user_id):
    user_dir = os.path.join(KFOLD_PATH, str(user_id))
    train_folds = [os.path.join(user_dir, f"train_fold{i}.txt") for i in range(5)]

    matched_ratings = pd.read_csv(MATCHED_RATINGS_PATH)
    matched_ratings["User-ID"] = matched_ratings["User-ID"].astype(str)
    matched_ratings["Book-Title"] = matched_ratings["Book-Title"].astype(str)

    train_data = assign_ratings_from_text_similarity(train_folds, matched_ratings)
    reader = Reader(rating_scale=(1, 10))
    train_dataset = Dataset.load_from_df(train_data[["User-ID", "Book-Title", "Book-Rating"]], reader)
    return train_dataset

# === Train models ===
def train_models_for_user(user_id):
    try:
        train_data = prepare_training_data(user_id)

        # CF model
        cf_model = KNNBasic(sim_options={"user_based": True})
        cf_model.fit(train_data.build_full_trainset())
        pickle.dump(cf_model, open(os.path.join(OUTPUT_MODEL_PATH, f"cf_model_{user_id}.pkl"), "wb"))

        # MF model
        mf_model = SVD()
        mf_model.fit(train_data.build_full_trainset())
        pickle.dump(mf_model, open(os.path.join(OUTPUT_MODEL_PATH, f"mf_model_{user_id}.pkl"), "wb"))

        print(f"✓ Trained models for user {user_id}")
    except Exception as e:
        print(f"❌ Error training user {user_id}: {e}")

# === Main ===
if __name__ == "__main__":
    user_ids = [uid for uid in os.listdir(KFOLD_PATH) if os.path.isdir(os.path.join(KFOLD_PATH, uid))]
    for user_id in user_ids:
        train_models_for_user(user_id)
