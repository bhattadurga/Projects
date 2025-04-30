def evaluate_model(model_dict):
    metrics = model_dict.get("metrics", {})
    suggestions = ""

    # ✅ Classification Evaluation
    if "Accuracy" in metrics:
        accuracy = metrics["Accuracy"]
        suggestions += f"📊 Classification Accuracy: {accuracy:.2%}. "

        if accuracy >= 0.85:
            suggestions += "🌟 Excellent classifier performance!"
        elif accuracy >= 0.70:
            suggestions += "✅ Good performance. Maybe improve with better features or boosting."
        else:
            suggestions += "⚠️ Accuracy is low. Consider feature engineering or using different classification models."

        return {
            "score": accuracy,
            "adjusted_R2": None,
            "suggestions": suggestions
        }

    # ✅ Regression Evaluation
    r2 = metrics.get("R2", 0)
    n_features = metrics.get("n_features", 1)
    n_samples = metrics.get("n_samples", 1)
    adjusted_r2 = None

    if n_samples > n_features + 1:
        adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

    if r2 > 0.8:
        suggestions += "🌟 Outstanding regression model fit!"
    elif r2 > 0.5:
        suggestions += "👍 Good model. Consider regularization or boosting."
    elif r2 > 0.2:
        suggestions += "⚠️ Limited predictive power. Try feature engineering or different models."
    else:
        suggestions += "❌ Poor performance. Try scaling features, reducing outliers, or switching model."

    if adjusted_r2 is not None:
        suggestions += f" Adjusted R² is {adjusted_r2:.3f}, which accounts for model complexity."

    # Add p-value info if available
    ols_info = model_dict.get("ols_summary")
    if ols_info:
        pvals = ols_info.get("p_values", {})
        insignificant = [k for k, v in pvals.items() if v > 0.05 and k != "const"]
        if insignificant:
            suggestions += f" ⚠️ The following features are not statistically significant: {', '.join(insignificant)}."

    return {
        "score": r2,
        "adjusted_R2": adjusted_r2,
        "suggestions": suggestions
    }
