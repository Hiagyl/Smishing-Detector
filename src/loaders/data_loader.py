# src/loaders/data_loader.py
import os
import shutil
import kagglehub
import pandas as pd


def download_and_organize_raw_data():
    """
    Downloads the two PH-specific smishing datasets using kagglehub
    and organizes them into data/1_raw/ for Git tracking.
    """
    raw_dir = "data/1_raw"
    os.makedirs(raw_dir, exist_ok=True)

    # --- DATASET 1: Philippine Spam SMS Messages ---
    print("📥 Fetching Dataset 1 (bwandowando/philippine-spam-sms-messages)...")
    path_1 = kagglehub.dataset_download(
        "bwandowando/philippine-spam-sms-messages")

    for file in os.listdir(path_1):
        if file.endswith('.csv'):
            shutil.copy(os.path.join(path_1, file), os.path.join(
                raw_dir, "ph_spam_dataset_1.csv"))
            print("✅ Dataset 1 saved as 'ph_spam_dataset_1.csv'")

    # --- DATASET 2: PH Spam & Marketing SMS ---
    print("📥 Fetching Dataset 2 (scottleechua/ph-spam-marketing-sms-w-timestamps)...")
    path_2 = kagglehub.dataset_download(
        "scottleechua/ph-spam-marketing-sms-w-timestamps")

    for file in os.listdir(path_2):
        if file.endswith('.csv'):
            shutil.copy(os.path.join(path_2, file), os.path.join(
                raw_dir, "ph_spam_dataset_2.csv"))
            print("✅ Dataset 2 saved as 'ph_spam_dataset_2.csv'")

    print(f"\n🎉 Raw storage preparation complete inside: {raw_dir}")


def combine_and_save_datasets():
    """
    Reads the two raw local CSV files, unifies their columns/schemas,
    combines them, drops duplicates, and saves the file to data/2_interim/.
    """
    print("\n🔄 Initializing data merging pipeline...")

    # 1. Process Dataset One
    # (Note: bwandowando's dataset generally uses columns like 'text' and 'label')
    df1 = pd.read_csv("data/1_raw/ph_spam_dataset_1.csv",
                      encoding='utf-8', errors='ignore')
    # Standardize column selection. Swap names below if your columns vary!
    df1 = df1[['text', 'label']]

    # 2. Process Dataset Two
    # (Note: scottleechua's dataset has columns like 'text', 'label', 'timestamp')
    df2 = pd.read_csv("data/1_raw/ph_spam_dataset_2.csv",
                      encoding='utf-8', errors='ignore')
    df2 = df2[['text', 'label']]  # Drop timestamp to match df1 schema

    # 3. Concatenate datasets
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # 4. Standardize labels to integers: 0 = Ham, 1 = Spam/Smishing
    # Handles strings like 'spam', 'ham', 'Spam', 'Ham'
    combined_df['label'] = combined_df['label'].astype(
        str).str.lower().str.strip()
    combined_df['label'] = combined_df['label'].map(
        {'ham': 0, 'spam': 1, 'smishing': 1})

    # Clean up rows that failed mapping or have missing values
    combined_df = combined_df.dropna(subset=['text', 'label'])
    combined_df['label'] = combined_df['label'].astype(int)

    # 5. Drop exact message duplicates across datasets
    initial_len = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['text'])
    print(
        f"✂️ Removed {initial_len - len(combined_df)} duplicate cross-over rows.")

    # 6. Save combined output
    interim_path = "data/2_interim/combined_raw_sms.csv"
    os.makedirs(os.path.dirname(interim_path), exist_ok=True)
    combined_df.to_csv(interim_path, index=False)

    print(
        f"🏁 Clean merge successful! Dataset saved to {interim_path} ({len(combined_df)} total rows).")
    return combined_df
