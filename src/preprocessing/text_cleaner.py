# src/preprocessing/text_cleaner.py
from config.threat_keywords import URGENCY_WORDS
import re
import pandas as pd


class SmishingFeatureExtractor:
    def __init__(self):
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        # self.urgency_words = ['voucher', 'congratulations', 'locked',
        #                       'winner', 'urgent', 'account', 'verify', 'nanalo']

    def extract_meta_features(self, df):
        """Extracts engineered metadata signals from the text and sender columns."""
        df = df.copy()

        # 1. Text Length Feature
        df['text_len'] = df['text'].apply(len)

        # 2. URL Structural Presence Flag
        df['has_url'] = df['text'].apply(
            lambda x: 1 if self.url_pattern.search(str(x)) else 0)

        # 3. Urgency Phrasing Density Scores
        df["urgency_score"] = df["text"].apply(
            lambda x: sum(
                1 for word in URGENCY_WORDS if word in str(x).lower())
        )

        # 4. Temporal Hour Extraction
        df['date-received'] = pd.to_datetime(
            df['date-received'], format='mixed')
        df['hour_received'] = df['date-received'].dt.hour

        # 5. One-Hot Encode Sender Categorization
        df = pd.get_dummies(
            df, columns=['sender_category'], prefix='sc', dtype=int)

        return df
