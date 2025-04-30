import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

def plot_feature_importance(df, column_name, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_df = df.sort_values(by=column_name, ascending=False)

    sorted_df.plot(kind="bar", x="Feature", y=column_name, ax=ax, legend=False, color="skyblue")
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Feature", fontsize=12, fontweight='bold')
    plt.ylabel(column_name, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')

    highest_feature = sorted_df.iloc[0]
    lowest_feature = sorted_df.iloc[-1]
    y_padding_high = max(0.1 * abs(highest_feature[column_name]), 1)

    ax.annotate(f"Highest: {highest_feature['Feature']}",
                xy=(0, highest_feature[column_name]),
                xytext=(0, highest_feature[column_name] + y_padding_high),
                arrowprops=dict(facecolor='green', shrink=0.05),
                ha='center', fontsize=10, fontweight='bold')

    ax.annotate(f"Lowest: {lowest_feature['Feature']}",
                xy=(len(sorted_df)-1, lowest_feature[column_name]),
                xytext=(len(sorted_df)-1, lowest_feature[column_name] + y_padding_high),
                arrowprops=dict(facecolor='red', shrink=0.05),
                ha='center', fontsize=10, fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def plot_logistic_coefficients(coef_df, title="Logistic Regression Coefficients"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_df = coef_df.sort_values(by="Weight", ascending=False)

    bars = ax.barh(sorted_df["Feature"], sorted_df["Weight"], color="skyblue")
    ax.set_xlabel("Coefficient Weight", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    for bar in bars:
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.2f}',
                va='center', ha='left' if bar.get_width() >= 0 else 'right',
                fontsize=9, fontweight='bold')

    plt.tight_layout()
    return fig

def plot_confusion_matrix_with_accuracy(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)

    ax.set_xlabel("Predicted", fontsize=12, fontweight='bold')
    ax.set_ylabel("True", fontsize=12, fontweight='bold')
    ax.set_title(f"Confusion Matrix (Accuracy = {acc:.1f}%)", fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig
