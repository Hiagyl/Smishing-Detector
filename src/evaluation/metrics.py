# src/evaluation/metrics.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_and_plot_models(models_dict, y_train, y_test):
    """
    Trains specified models, calculates core metrics, 
    and generates consolidated confusion matrix heatmaps.
    """
    results = []
    plt.figure(figsize=(18, 12))

    print("--- Individual Model Performance ---")

    for i, (name, (model, train_data, test_data)) in enumerate(models_dict.items(), 1):
        # 1. Fit and Predict
        model.fit(train_data, y_train)
        preds = model.predict(test_data)

        # 2. Extract Mathematical Metrics
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)

        # 3. Store in metrics log registry
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

        # 4. Generate Plot Grid Layouts
        plt.subplot(2, 3, i)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False
        )
        plt.title(
            f"{name}\n"
            f"Acc={acc:.3f} | Prec={precision:.3f}\n"
            f"Recall={recall:.3f} | F1={f1:.3f}",
            fontsize=11
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        print(f"✅ {name}: Complete")

    plt.tight_layout()
    plt.show()

    # 5. Format results summary table
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="F1-Score", ascending=False).reset_index(drop=True)

    return results_df
