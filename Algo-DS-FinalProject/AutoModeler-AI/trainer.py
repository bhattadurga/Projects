import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    confusion_matrix
)
import statsmodels.api as sm

def engineer_features(df):
    df["Income_per_room"] = df["Avg. Area Income"] / df["Avg. Area Number of Rooms"]
    df["Rooms_per_bedroom"] = df["Avg. Area Number of Rooms"] / df["Avg. Area Number of Bedrooms"]
    df["House_Age_Squared"] = df["Avg. Area House Age"] ** 2
    return df

def remove_outliers(df):
    if "Price" in df.columns:
        q_low = df["Price"].quantile(0.01)
        q_high = df["Price"].quantile(0.99)
        df = df[(df["Price"] >= q_low) & (df["Price"] <= q_high)]
    return df

def train(data: dict, model_type: str = "linear", bin_target: bool = False):
    df = pd.DataFrame(data)
    df = df.select_dtypes(include=[float, int])
    df = engineer_features(df)
    df = remove_outliers(df)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if model_type == "logistic" and bin_target:
        y_binned = pd.qcut(y, q=3, labels=[0, 1, 2])
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y_binned)
        y_pred = model.predict(X)

        metrics = {
            "Accuracy": accuracy_score(y_binned, y_pred),
            "n_samples": len(y_binned),
            "n_features": X.shape[1]
        }

        return model, model_type, metrics, None, y_binned.tolist(), y_pred.tolist()

    # Regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ols_summary = None

    if model_type == "linear":
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        X_ols = sm.add_constant(X_train)
        ols_model = sm.OLS(y_train, X_ols).fit()
        ols_summary = {
            "coefficients": ols_model.params.to_dict(),
            "t_values": ols_model.tvalues.to_dict(),
            "p_values": ols_model.pvalues.to_dict(),
            "summary": ols_model.summary2().as_text()
        }

    elif model_type == "gradient_boost":
        model = GradientBoostingRegressor()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    else:
        raise ValueError("Unsupported model type.")

    metrics = {
        "R2": model.score(X_test_scaled, y_test),
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "n_samples": len(y),
        "n_features": X.shape[1]
    }

    return model, model_type, metrics, ols_summary, None, None

def save_model(model, filename="saved_model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
