def evaluate_model(model_dict):
    score = model_dict.get("metrics", {}).get("R2", 0)
    n_features = model_dict.get("metrics", {}).get("n_features", 1)
    n_samples = model_dict.get("metrics", {}).get("n_samples", 1)
    adjusted_r2 = None
    feedback = ""

    if n_samples > n_features + 1:
        adjusted_r2 = 1 - (1 - score) * (n_samples - 1) / (n_samples - n_features - 1)

    if score > 0.8:
        feedback += "🌟 Outstanding model fit!"
    elif score > 0.5:
        feedback += "👍 Good model. You could still try boosting or regularization."
    elif score > 0.2:
        feedback += "⚠️ Model has limited predictive power. Try feature engineering, adding interactions, or using tree-based models."
    else:
        feedback += "❌ Poor performance. Consider different algorithm or feature engineering. MAE is quite high — maybe scale features or reduce outliers."

    if adjusted_r2 is not None:
        feedback += f" Adjusted R² is {adjusted_r2:.3f}, which accounts for model complexity."

    # Add p-value analysis if available
    ols_info = model_dict.get("ols_summary")
    if ols_info:
        pvals = ols_info.get("p_values", {})
        insignificant = [k for k, v in pvals.items() if v > 0.05 and k != "const"]
        if insignificant:
            feedback += f" ⚠️ The following features are not statistically significant: {', '.join(insignificant)}."

    return {
        "score": score,
        "adjusted_R2": adjusted_r2,
        "suggestions": feedback
    }
