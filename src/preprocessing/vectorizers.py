# src/preprocessing/vectorizers.py
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os


class SmishingTfidfVectorizer:
    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features
        )

    def fit_transform(self, train_text):
        """Fits vocabulary on training text and transforms it."""
        return self.vectorizer.fit_transform(train_text)

    def transform(self, test_text):
        """Transforms unseen test or deployment text strings based on learned vocabulary."""
        return self.vectorizer.transform(test_text)

    def save_vectorizer(self, filepath="saved_models/tfidf_vectorizer.pkl"):
        """Saves the fitted vectorizer matrix state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)
        print(f"📦 TF-IDF Vectorizer saved to {filepath}")
