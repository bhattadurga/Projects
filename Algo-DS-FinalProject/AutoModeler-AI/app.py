import streamlit as st
import pandas as pd
import requests
from io import StringIO
import pickle
import matplotlib.pyplot as plt

# ✅ Import plotting function from separate file
from plot_utils import plot_feature_importance

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
            st.info("📃 Adding engineered features to your dataset...")
            df["Income_per_room"] = df["Avg. Area Income"] / df["Avg. Area Number of Rooms"]
            df["Rooms_per_bedroom"] = df["Avg. Area Number of Rooms"] / df["Avg. Area Number of Bedrooms"]
            df["House_Age_Squared"] = df["Avg. Area House Age"] ** 2
            st.success("✅ New features added.")

            st.write("### 🆕 Updated Dataset with Engineered Features")
            st.dataframe(df.head())

            st.info("\n🔁 Re-training model with new features...")
            data_dict = df.to_dict(orient="list")
            payload = {"data": data_dict, "model_type": model_type}
            if bin_target:
                payload["bin"] = True

            response = requests.post(f"{api_base_url}/train_model", json=payload)
            if response.status_code == 200:
                model_info = response.json()
                st.success("\n📅 Model retrained with new features!")
                st.write(f"**Model Type:** {model_info.get('model_type')}")
                st.write(f"**Metrics:** {model_info.get('metrics')}")
                st.write(f"**Performance Summary:** {model_info.get('performance_summary')}")
                st.session_state.df = df

                user_action = st.radio("\n\n📋 What would you like to do next?", [
                    "Try boosting",
                    "Show features",
                    "Explain prediction",
                    "Download model"])

                if user_action == "Try boosting":
                    st.info("\n🚀 Retrying with Gradient Boosting...")
                    payload = {"data": df.to_dict(orient="list"), "model_type": "gradient_boost"}
                    response = requests.post(f"{api_base_url}/train_model", json=payload)
                    if response.status_code == 200:
                        model_info = response.json()
                        st.success("\n🌟 Gradient Boosting model trained!")
                        st.write(f"**Model Type:** {model_info.get('model_type')}")
                        st.write(f"**Metrics:** {model_info.get('metrics')}")
                        st.write(f"**Performance Summary:** {model_info.get('performance_summary')}")

                elif user_action == "Show features":
                    st.write("\n📊 Current Features:")
                    st.dataframe(df.head())

                elif user_action == "Explain prediction":
                    row_idx = st.number_input("Enter the row index to explain (0-based):", min_value=0, max_value=len(df)-1, step=1)
                    if st.button("Explain"):
                        selected = df.iloc[[int(row_idx)]]
                        st.write("\n🔎 Input to explain:")
                        st.dataframe(selected)
                        st.write("\n📊 Feature impact (approximate):")
                        for col in selected.columns[:-1]:
                            st.write(f"• {col}: {selected[col].values[0]:.2f}")

                elif user_action == "Download model":
                    st.info("\n📦 Preparing model download...")
                    try:
                        with open("saved_model.pkl", "rb") as f:
                            st.download_button(
                                label="📅 Download Trained Model (.pkl)",
                                data=f,
                                file_name="trained_model.pkl",
                                mime="application/octet-stream"
                            )
                            st.success("\n🎉 Model download ready!")
                        
                        st.write("### 🔍 Loaded Model Information")
                        with open("saved_model.pkl", "rb") as model_file:
                            loaded_model = pickle.load(model_file)
                        
                        st.write(loaded_model)

                        feature_names = st.session_state.df.drop(columns=[st.session_state.df.columns[-1]]).columns

                        if hasattr(loaded_model, "coef_"):
                            st.write("### 📈 Feature Weights (Coefficients)")
                            coef_df = pd.DataFrame({
                                "Feature": feature_names,
                                "Weight": loaded_model.coef_
                            })
                            st.dataframe(coef_df)

                            st.write("### 📊 Feature Weights Chart")
                            fig = plot_feature_importance(coef_df, "Weight", "Feature Weights (Linear Model)")
                            st.pyplot(fig)

                            csv = coef_df.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Download Feature Weights as CSV", csv, "feature_weights.csv", "text/csv")

                        elif hasattr(loaded_model, "feature_importances_"):
                            st.write("### 📊 Feature Importances (Tree-based Model)")
                            importances_df = pd.DataFrame({
                                "Feature": feature_names,
                                "Importance": loaded_model.feature_importances_
                            })
                            st.dataframe(importances_df)

                            st.write("### 📊 Feature Importances Chart")
                            fig = plot_feature_importance(importances_df, "Importance", "Feature Importances (Boosted Model)")
                            st.pyplot(fig)

                            csv = importances_df.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Download Feature Importances as CSV", csv, "feature_importances.csv", "text/csv")

                        else:
                            st.info("ℹ️ Model type does not expose coefficients or feature importances.")

                    except FileNotFoundError:
                        st.error("\n❌ Model file not found.")
else:
    st.info("📄 Upload a CSV file to get started.")
