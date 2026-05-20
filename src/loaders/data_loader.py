# src/loaders/data_loader.py
import os
import re
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter


def transform_sender_to_category(sender):
    """Categorizes the sender based on strings, privacy markers, and lengths."""
    s = str(sender).strip().lower()
    masked_and_digits = re.sub(r'[^0-9*]', '', s)

    if "redacted_individual" in s or "redacted_business" in s:
        return "Personal_Mobile"
    if s in ["unknown", "hidden", "private"]:
        return "Anonymized_Sender"
    if len(masked_and_digits) >= 10:
        return "Personal_Mobile"
    if masked_and_digits and len(masked_and_digits) < 10:
        return "Short_Code"
    if any(char.isalpha() for char in s):
        return "Verified_Brand"
    return "Unknown"


def load_and_combine_ph_data():
    """Downloads, merges, prunes redactions, and labels the PH datasets."""
    print("📥 Loading Dataset 1 (bwandowando)...")
    df1 = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "bwandowando/philippine-spam-sms-messages",
        "SPAM_SMS.csv"
    )
    # Align and label df1
    df1['category'] = 'smishing'
    df1 = df1.rename(columns={'date': 'date-received',
                     'masked_celphone_number': 'sender'})

    print("📥 Loading Dataset 2 (scottleechua)...")
    df2 = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "scottleechua/ph-spam-marketing-sms-w-timestamps",
        "text-messages.csv"
    )

    # 1. Filter columns to ensure structural matching before stacking
    keep_cols = ['text', 'category', 'date-received', 'sender']
    df1_filtered = df1[keep_cols].copy()
    df2_filtered = df2[keep_cols].copy()

    # 2. Combine DataFrames
    combined_df = pd.concat([df2_filtered, df1_filtered], ignore_index=True)

    # 3. Prune complete redactions that leak no textual signal
    df_filtered = combined_df[combined_df['text'] != '<REDACTED>'].copy()
    df_filtered = df_filtered[df_filtered['text'].notna()]

    # 4. Standardize text categories ('spam' -> 'smishing')
    df_filtered['category'] = df_filtered['category'].replace(
        'spam', 'smishing')

    # 5. Transform Senders
    df_filtered['sender_category'] = df_filtered['sender'].apply(
        transform_sender_to_category)

    # 6. Map target variables to strict binary integers
    # 1 = Smishing, 0 = Safe (Ads, Gov, OTP, Personal)
    mapping = {'smishing': 1, 'ads': 0, 'gov': 0, 'otp': 0, 'personal': 0}
    df_filtered['label'] = df_filtered['category'].str.lower(
    ).str.strip().map(mapping).fillna(0).astype(int)

    # Save a local cache snapshot to interim data registry
    os.makedirs("data/2_interim", exist_ok=True)
    df_filtered.to_csv("data/2_interim/combined_raw_sms.csv", index=False)
    print(f"🏁 Data Ingestion Complete! Merged shape: {df_filtered.shape}")

    return df_filtered
