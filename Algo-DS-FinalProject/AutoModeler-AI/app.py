import streamlit as st
import pandas as pd
import requests
import pickle
from io import StringIO
import matplotlib.pyplot as plt
from plot_utils import plot_feature_importance, plot_logistic_coefficients, plot_confusion_matrix_with_accuracy

st.set_page_config(page_title="AutoModeler AI", layout="wide")
st.title("AutoModeler AI 🤖")

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
model_type = st.sidebar.selectbox("Choose Model Type", ["Auto", "Regression", "Classification"])
api_base_url = st.sidebar.text_input("API URL", value="http://localhost:8000")
enable_ai_judge = st.sidebar.checkbox("Enable AI Judge", value=True)

bin_target = False
user_clarification = None

if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    df = pd.read_csv(stringio)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.select_dtypes(include=[float, int])
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    task_description = st.text_input("📝 Describe your task (e.g., 'Predict housing prices')")

    if task_description:
        st.write("### Chatbot: Analyzing your task...")

        if model_type == "Regression":
            model_type = "linear"
        elif model_type == "Classification":
            model_type = "logistic"
        else:
            model_type = "auto"

        if "price" in task_description.lower() and "logistic" in task_description.lower():
            st.write("🤖 You mentioned logistic regression for housing price.")
            st.write("Housing price is a continuous variable — logistic regression might not be suitable.")
            st.write("Would you like me to bin the prices and use logistic regression, or switch to linear regression?")
            user_clarification = st.text_input("Your reply (e.g., 'Use linear regression'):")

            if "linear" in user_clarification.lower():
                model_type = "linear"
                st.success("✅ Using linear regression.")
            elif "logistic" in user_clarification.lower() or "bin" in user_clarification.lower():
                model_type = "logistic"
                bin_target = True
                st.success("✅ Using logistic regression with binning.")
            elif user_clarification:
                st.warning("🤖 Please reply with 'linear' or 'logistic'.")
                st.stop()
            else:
                st.stop()

        st.session_state.df = df
        data_dict = df.to_dict(orient="list")
        payload = {
            "data": data_dict,
            "model_type": model_type
        }
        if bin_target:
            payload["bin"] = True

        st.write("📦 Payload being sent to backend:")
        st.json(payload)

        response = requests.post(f"{api_base_url}/train_model", json=payload)

        if response.status_code == 200:
            model_info = response.json()
            st.success("✅ Model Trained Successfully!")
            st.write(f"**Model Type:** {model_info.get('model_type')}")
            st.write(f"**Metrics:** {model_info.get('metrics')}")
            st.write(f"**Performance Summary:** {model_info.get('performance_summary')}")

            # 🎯 Confusion matrix for logistic model
            if model_info.get("model_type") == "logistic" and model_info.get("true_labels") and model_info.get("predicted_labels"):
                y_true = model_info["true_labels"]
                y_pred = model_info["predicted_labels"]
                fig = plot_confusion_matrix_with_accuracy(y_true, y_pred)
                st.pyplot(fig)

            if enable_ai_judge:
                st.write("### 🧑‍⚖️ AI Judge Evaluation")
                judge_response = requests.post(f"{api_base_url}/judge", json=model_info)
                if judge_response.status_code == 200:
                    judge = judge_response.json()
                    st.write(f"**Judge Score:** {judge.get('score')}")
                    if judge.get("adjusted_R2") is not None:
                        st.write(f"**Adjusted R²:** {judge.get('adjusted_R2'):.3f}")
                    st.write(f"**Suggestions:** {judge.get('suggestions')}")
        else:
            st.error("❌ Model training failed.")
            st.stop()

        st.markdown("---")

        command = st.text_input("\n💬 Type your assistant command (e.g., 'add features'):")

        if "add" in command.lower():
            st.info("📃 Adding engineered features...")
            df["Income_per_room"] = df["Avg. Area Income"] / df["Avg. Area Number of Rooms"]
            df["Rooms_per_bedroom"] = df["Avg. Area Number of Rooms"] / df["Avg. Area Number of Bedrooms"]
            df["House_Age_Squared"] = df["Avg. Area House Age"] ** 2
            st.success("✅ Features added.")
            st.write("### 🆕 Engineered Dataset")
            st.dataframe(df.head())

            st.info("🔁 Re-training model...")
            payload = {"data": df.to_dict(orient="list"), "model_type": model_type}
            if bin_target:
                payload["bin"] = True

            response = requests.post(f"{api_base_url}/train_model", json=payload)
            if response.status_code == 200:
                model_info = response.json()
                st.success("📅 Model retrained successfully!")
                st.session_state.df = df
                st.write(f"**Model Type:** {model_info.get('model_type')}")
                st.write(f"**Metrics:** {model_info.get('metrics')}")
                st.write(f"**Performance Summary:** {model_info.get('performance_summary')}")

                user_action = st.radio("📋 What next?", [
                    "Try boosting",
                    "Show features",
                    "Explain prediction",
                    "Download model"])

                if user_action == "Try boosting":
                    st.info("🚀 Retrying with Gradient Boosting...")
                    payload = {"data": df.to_dict(orient="list"), "model_type": "gradient_boost"}
                    response = requests.post(f"{api_base_url}/train_model", json=payload)
                    if response.status_code == 200:
                        model_info = response.json()
                        st.success("🌟 Boosting completed!")
                        st.write(f"**Model Type:** {model_info.get('model_type')}")
                        st.write(f"**Metrics:** {model_info.get('metrics')}")
                        st.write(f"**Performance Summary:** {model_info.get('performance_summary')}")

                elif user_action == "Show features":
                    st.write("📊 Current Features:")
                    st.dataframe(df.head())

                elif user_action == "Explain prediction":
                    idx = st.number_input("Enter row index (0-based):", min_value=0, max_value=len(df)-1)
                    if st.button("Explain"):
                        row = df.iloc[[int(idx)]]
                        st.write("�� Row Data")
                        st.dataframe(row)
                        for col in row.columns:
                            st.write(f"• {col}: {row[col].values[0]:.2f}")

                elif user_action == "Download model":
                    st.info("📦 Preparing model download...")
                    try:
                        with open("saved_model.pkl", "rb") as f:
                            st.download_button(
                                "📅 Download Trained Model (.pkl)",
                                f,
                                file_name="trained_model.pkl",
                                mime="application/octet-stream"
                            )
                            st.success("🎉 Model download ready!")

                        with open("saved_model.pkl", "rb") as model_file:
                            model = pickle.load(model_file)

                        st.write("### 🔍 Loaded Model Summary")
                        st.write(model)

                        feature_names = df.drop(columns=[df.columns[-1]]).columns
                        if hasattr(model, "coef_"):
                            st.write("### 📈 Feature Weights (Coefficients)")
                            weights_df = pd.DataFrame({
                                "Feature": feature_names,
                                "Weight": model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                            })
                            st.dataframe(weights_df)

                            # Choose plot function based on model type
                            if model_type == "logistic":
                                fig = plot_logistic_coefficients(weights_df)
                            else:
                                fig = plot_feature_importance(weights_df, "Weight", "Feature Weights (Linear Model)")

                            st.pyplot(fig)

                            csv = weights_df.to_csv(index=False).encode("utf-8")
                            st.download_button("📥 Download Feature Weights", csv, "feature_weights.csv", "text/csv")

                        elif hasattr(model, "feature_importances_"):
                            st.write("### 📊 Feature Importances")
                            fi_df = pd.DataFrame({
                                "Feature": feature_names,
                                "Importance": model.feature_importances_
                            })
                            st.dataframe(fi_df)

                            fig = plot_feature_importance(fi_df, "Importance", "Feature Importances")
                            st.pyplot(fig)

                            csv = fi_df.to_csv(index=False).encode("utf-8")
                            st.download_button("📥 Download Feature Importances", csv, "feature_importances.csv", "text/csv")

                        else:
                            st.info("ℹ️ Model does not expose coefficients or importances.")

                    except FileNotFoundError:
                        st.error("❌ saved_model.pkl not found.")
else:
    st.info("📄 Upload a CSV file to begin.")
