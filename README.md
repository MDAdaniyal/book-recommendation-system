# 📚 Book Recommendation System

This project implements a hybrid book recommendation system combining:
- **Content-based modeling** using raw book texts
- **TAWA toolkit** (compression-based modeling)
- **Matrix Factorization & Collaborative Filtering** using Surprise

It includes full preprocessing, k-fold evaluation, model training, and result comparison.

---

## 🗂️ Project Structure
```
book-recommendation-system/
├── data/
│   ├── raw/                  # ratings.csv, books.csv, GUTINDEX.ALL.txt
│   ├── processed/            # BooksFiltered.csv
│   ├── final_books/          # Cleaned text files
│   └── kfold_data/           # User-specific 5-fold data
│
├── src/
│   ├── preprocessing/        # create_books_filtered.py
│   ├── kfold/                # create_kfold.py
│   ├── tawa/                 # train_tawa_models.py, run_ident.sh
│   ├── traditional/          # train_trad_models.py
│   └── evaluation/           # trad_results.py
│
├── results/                  # tawa_evaluation_metrics.csv, TRAD_results.csv
├── notebooks/                # book_recommendation_demo.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Setup Instructions

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run scripts step-by-step**:
```bash
# Step 1: Filter relevant books from ratings + Gutenberg
python src/preprocessing/create_books_filtered.py

# Step 2: Create 5-fold text data for each user
python src/kfold/create_kfold.py

# Step 3a: Train TAWA models (Linux/WSL only)
python src/tawa/train_tawa_models.py

# Step 3b: Train traditional models (Surprise)
python src/traditional/train_trad_models.py

# Step 4: Evaluate traditional model results
python src/evaluation/trad_results.py
```

3. **Run the Jupyter Notebook**:
```bash
jupyter notebook notebooks/book_recommendation_demo.ipynb
```

---

## 📊 Output
- Results saved in `results/`
- Evaluation results: `tawa_evaluation_metrics.csv`, `TRAD_results.csv`

---

## 📌 Requirements
- Python 3.8+
- Linux/WSL for TAWA
- Surprise, Pandas, Scikit-learn

---

## 🧠 Author
Mohammad Daniyal Ayyub  
MSc Advanced Data Science, Bangor University

---

## ✅ License
MIT License (optional)
