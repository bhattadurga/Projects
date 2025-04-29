# AutoModeler AI 🤖

## Overview
AutoModeler AI is an intelligent end-to-end chatbot application that guides users through uploading datasets, training machine learning models, evaluating them and building machine learning models based on their data and goals.It features a simple UI for CSV upload, task description, model building, feature importance visualization, and model download.
It supports dynamic model selection between regression and classification tasks, feature engineering, retraining, and judging model quality — all integrated into a seamless pipeline powered by Streamlit + FastAPI.

## Features
- **Conversational UI** : Friendly Streamlit app guides users through model training.
- **Dynamic Model Type Selection**: Automatically suggests Regression, Classification, or Boosting.
- **Model Building Pipeline**: Cleans data, engineers new features, and trains machine learning models.
- **Evaluation Summary**: Outputs performance metrics like R², MSE, MAE, Adjusted R².
- **AI Judge**: Evaluates trained models and provides suggestions.
- **Feature Visualization**: Displays feature importance using beautiful charts.
- **Model Download**: Download trained models (.pkl) and feature weights (CSV).

## Tech Stack
- **Frontend**: Streamlit (for user interface)
- **Backend**: FastAPI (Python 3)
- **ML Libraries**: Scikit-learn, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model Persistence**: joblib

**Deployment (Bonus):** AWS Lambda + API Gateway

## Installation
## Prerequisites
Make sure you have the following installed:
- Python 3.10+
- pip
- (Optional) Virtual Environment

### Steps
1.Clone the repository:
  ```sh
  git clone https://github.com/bhattadurga/Projects/tree/main/Algo-DS-FinalProject/AutoModeler-AI
  cd AutoModeler-AI
  ```
2.Create and activate a virtual environment:
  ```sh
  python -m venv venv
  source venv/bin/activate  # macOS/Linux
  venv\Scripts\activate     # Windows
  ```
3.Install dependencies:
  ```sh
  pip install -r requirements.txt
  ```
4. Running the Application
   Start the FastAPI Backend - This runs the backend server on: http://localhost:8000
   ```sh
   uvicorn model_api:app --host 0.0.0.0 --port 8000 --reload
   ```  
5.Running AutoModeler AI
  Start the Streamlit Frontend - This opens the frontend on: http://localhost:8501
  Launch the application by running:
  ```sh
  streamlit run app.py
  ```
Now you can upload your CSV file, select model types, and train!

## Configuration
Adjust these parameters in `app.py` sidebar to customize behavior:
```python
model_type = st.selectbox("Choose Model Type", ["Auto", "Regression", "Classification"])
api_base_url = st.text_input("API URL", value="http://localhost:8000")
enable_ai_judge = st.checkbox("Enable AI Judge", value=True)
```

## Usage
## User Journey
**Scenario 1 (Regression):**
1.User: “I want to predict housing prices.”
AutoModeler: “That sounds like a continuous variable. Would you like to use linear regression?”
→ Backend trains a linear regression model and returns R² and coefficients.

**Scenario 2 (Classification):**
2.User: “I want to classify housing prices into 'low', 'medium', 'high'.”
AutoModeler: “Great, I’ll bin the target variable and apply logistic regression. Shall we proceed?”
→ Backend bins the target and trains logistic regression, returning a confusion matrix and accuracy.

## Code Structure
```
AutoModeler-AI/

│── app.py                  # Streamlit frontend app
│── model_api.py            # FastAPI backend server
│── trainer.py              # Model training logic
│── judge.py                # Model judging logic
│── plot_utils.py           # Feature importance plotting
│── utils/                  # (optional helpers)
│── requirements.txt        # Required dependencies
│── saved_model.pkl         # Generated after training
│── README.md               # Project documentation

```

## API and AI Integration

- **POST /train_model**: Train and return metrics.

- **POST /judge**: Evaluate model and return scoring.

- Initially, saved_model.pkl will not exist.
It will be automatically created after the first model training.

- All model files and weights can be downloaded through the Streamlit app.

**Sample FastAPI Endpoint:**
```sh
@app.post("/train_model")
def train_model(file: UploadFile, model_type: str):
    ...
```

## AWS Deployment (Optional)
You can deploy the FastAPI backend using:
- Frameworks: serverless, zappa
- Cloud Tools: AWS Lambda, API Gateway, boto3
- Testing Tools: Postman, cURL for verifying endpoints

## Troubleshooting
- Streamlit fails to load:
  ```sh
  pip install streamlit --upgrade
  ```
  FastAPI not running:
  ```sh
  uvicorn api.model_api:app --reload
  CORS issues (for frontend-backend communication): Add CORSMiddleware in model_api.py.
  ```

## Contribution
- Fork the repository.
- Create a feature branch (`git checkout -b feature-name`).
- Commit your changes (`git commit -m "Add new feature"`).
- Push to GitHub (`git push origin feature-name`).
- Submit a Pull Request.

## Contact
For questions or contributions, reach out to: vb46901n@pace.edu
