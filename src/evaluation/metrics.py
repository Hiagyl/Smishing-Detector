# src/evaluation/metrics.py
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

# ... keep your existing evaluate_and_plot_models function here ...


def plot_all_ensemble_matrices(stacked_results_df, train_probs, test_probs, y_train, y_test):
    """
    Generates a dense grid layout containing the confusion matrices 
    for all 31 ensemble permutations.
    """
    n_combos = len(stacked_results_df)
    cols = 4
    rows = math.ceil(n_combos / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4))
    axes = axes.flatten()

    print(
        f"🔮 Generating all {n_combos} hybrid ensemble confusion matrices (Purples Spectrum)...")

    for i, (index, row) in enumerate(stacked_results_df.iterrows()):
        combo_name = row["Combination"]
        combo_models = combo_name.split(" + ")

        # Re-stack the chosen sub-probabilities horizontally
        meta_X_train_sub = np.column_stack(
            [train_probs[m] for m in combo_models])
        meta_X_test_sub = np.column_stack(
            [test_probs[m] for m in combo_models])

        # Instantiate a temporary meta-classifier for matrix extraction
        temp_meta_clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        )
        temp_meta_clf.fit(meta_X_train_sub, y_train)

        y_pred_combo = temp_meta_clf.predict(meta_X_test_sub)
        cm = confusion_matrix(y_test, y_pred_combo)

        # Draw specific subplot
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Purples",
            cbar=False,
            ax=axes[i]
        )

        axes[i].set_title(
            f"Rank {i+1}: {combo_name}\n"
            f"Acc: {row['Accuracy']:.4f} | F1: {row['F1-Score']:.4f}",
            fontsize=9
        )
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    # Clean up empty subplots at the bottom of the grid
    for j in range(n_combos, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
