# src/models/ensemble.py
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import os
import joblib


def evaluate_all_stacked_combinations(base_models, y_train, y_test):
    """
    Generates cross-validated out-of-fold probability meta-features,
    evaluates all 31 non-empty model combinations using an XGBoost meta-classifier,
    and returns a ranked performance DataFrame.
    """
    train_probs = {}
    test_probs = {}

    print("🔄 Generating out-of-fold base model probabilities...")
    for name, (model, train_data, test_data) in base_models.items():
        # Generate 5-fold cross-validated training probability matrices to avoid data leakage
        train_probs[name] = cross_val_predict(
            model, train_data, y_train, cv=5, method="predict_proba"
        )[:, 1]

        # Train full model on the entire training subset for inference testing
        model.fit(train_data, y_train)
        test_probs[name] = model.predict_proba(test_data)[:, 1]
        print(f"  ↳ {name}: Probabilities generated.")

    ensemble_results = []
    model_names = list(base_models.keys())

    print("\n⚔️ Evaluating all 31 stacked ensemble combinations...")
    for r in range(1, len(model_names) + 1):
        for combo in itertools.combinations(model_names, r):
            combo_name = " + ".join(combo)

            # Stack predicted probabilities horizontally as your new feature space
            meta_X_train = np.column_stack([train_probs[m] for m in combo])
            meta_X_test = np.column_stack([test_probs[m] for m in combo])

            # Initialize meta-classifier engine
            meta_clf = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42
            )
            meta_clf.fit(meta_X_train, y_train)
            y_pred = meta_clf.predict(meta_X_test)

            # Record metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            ensemble_results.append({
                "Combination": combo_name,
                "Model_Count": len(combo),
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
            })

    # Convert, sort, and format results
    stacked_results_df = pd.DataFrame(ensemble_results)
    stacked_results_df = stacked_results_df.sort_values(
        by="F1-Score", ascending=False
    ).reset_index(drop=True)

    print("\n🎉 All combinations evaluated successfully!")
    return stacked_results_df, train_probs, test_probs


def train_and_save_champion(stacked_results_df, train_probs, test_probs, y_train, filepath="saved_models/champion_ensemble.pkl"):
    """
    Identifies the highest-performing model combination from the grid search,
    re-trains its specific XGBoost meta-classifier layer, and pickles it to disk.
    """
    # 1. Identify the champion configuration names from the top row
    champion_row = stacked_results_df.iloc[0]
    champion_combo_name = champion_row["Combination"]
    combo_models = champion_combo_name.split(" + ")

    print(f"\n🏆 Champion Combination Identified: {champion_combo_name}")
    print(f"🥇 Validation F1-Score: {champion_row['F1-Score']:.4f}")

    # 2. Re-stack the exact subset of probabilities that won
    meta_X_train_sub = np.column_stack([train_probs[m] for m in combo_models])

    # 3. Train the final production model instance
    champion_meta_clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )
    print("🏋️ Freezing final meta-classifier parameters...")
    champion_meta_clf.fit(meta_X_train_sub, y_train)

    # 4. Export the artifact using joblib
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(champion_meta_clf, filepath)
    print(
        f"📦 Success! Champion ensemble model pickled directly to: {filepath}")

    # Return the models list so the frontend knows which base models to run
    return combo_models
