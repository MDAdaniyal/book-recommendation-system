import pandas as pd
import os
from difflib import get_close_matches

# === CONFIG ===
RAW_DATA_PATH = os.path.join("..", "..", "data", "raw")
PROCESSED_DATA_PATH = os.path.join("..", "..", "data", "processed")
GUTENBERG_INDEX_FILE = os.path.join(RAW_DATA_PATH, "GUTINDEX.ALL.txt")
RATINGS_FILE = os.path.join(RAW_DATA_PATH, "ratings.csv")
BOOKS_FILE = os.path.join(RAW_DATA_PATH, "books.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DATA_PATH, "BooksFiltered.csv")

# === Load data ===
print("Loading data...")
ratings_df = pd.read_csv(RATINGS_FILE)
books_df = pd.read_csv(BOOKS_FILE)

# === Merge and filter ===
print("Merging and filtering data...")
merged_df = pd.merge(ratings_df, books_df, on="ISBN")
high_rated_df = merged_df[merged_df["Book-Rating"] >= 7]

# === Filter users with 5-9 rated books ===
user_book_counts = high_rated_df.groupby("User-ID")["Book-Title"].nunique()
eligible_users = user_book_counts[(user_book_counts >= 5) & (user_book_counts <= 9)].index
filtered_df = high_rated_df[high_rated_df["User-ID"].isin(eligible_users)]

# === Match with Gutenberg Index ===
print("Matching titles with Project Gutenberg index...")
with open(GUTENBERG_INDEX_FILE, "r", encoding="utf-8", errors="ignore") as f:
    gutenberg_lines = f.readlines()

gutenberg_titles = set()
for line in gutenberg_lines:
    parts = line.strip().split()
    if len(parts) > 1 and parts[-1].isdigit():
        title = " ".join(parts[:-1]).strip().lower()
        gutenberg_titles.add(title)

def match_title(title):
    matches = get_close_matches(title.lower(), gutenberg_titles, n=1, cutoff=0.9)
    return bool(matches)

filtered_df["Is_Gutenberg"] = filtered_df["Book-Title"].apply(match_title)
gutenberg_df = filtered_df[filtered_df["Is_Gutenberg"] == True]

# === Output filtered books ===
print(f"Saving filtered data to {OUTPUT_FILE}...")
gutenberg_df = gutenberg_df.drop(columns=["Is_Gutenberg"])
gutenberg_df.to_csv(OUTPUT_FILE, index=False)

print("✓ BooksFiltered.csv created successfully.")
