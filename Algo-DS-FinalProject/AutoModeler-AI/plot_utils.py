import matplotlib.pyplot as plt

def plot_feature_importance(df, column_name, title):
    """
    Plots feature importance or feature weights.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Feature' and 'Weight' or 'Importance' columns.
        column_name (str): Column to plot (e.g., 'Weight' or 'Importance')
        title (str): Title of the chart
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_df = df.sort_values(by=column_name, ascending=False)

    # Plot bar chart
    sorted_df.plot(kind="bar", x="Feature", y=column_name, ax=ax, legend=False, color="skyblue")

    # Set x and y ticks: Normal font (no bold)
    plt.xticks(rotation=30, ha="right", fontsize=10, fontweight='normal')
    plt.yticks(fontsize=10, fontweight='normal')

    # Set axis labels and title: Bold
    plt.xlabel("Feature", fontsize=12, fontweight='bold')
    plt.ylabel(column_name, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')

    # Find highest and lowest features
    highest_feature = sorted_df.iloc[0]
    lowest_feature = sorted_df.iloc[-1]

    # Calculate padding
    y_padding_high = max(0.1 * abs(highest_feature[column_name]), 1)

    # ✅ Move both annotations UPWARD above the bar
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

    # Add light gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    return fig

def plot_logistic_coefficients(coef_df, title="Logistic Regression Coefficients"):
    """
    Plots logistic regression coefficients.
    
    Args:
        coef_df (pd.DataFrame): DataFrame with 'Feature' and 'Weight' columns.
        title (str): Title for the plot.
    """
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
