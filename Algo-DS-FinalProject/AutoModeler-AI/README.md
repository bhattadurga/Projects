AutoModeler AI 🤖
Overview
AutoModeler AI is a complete intelligent platform that guides users through uploading datasets, training machine learning models, and evaluating them — without any coding. It features a simple UI for CSV upload, task description, model building, feature importance visualization, and model download.

It supports dynamic model selection between regression and classification tasks, feature engineering, retraining, and judging model quality — all integrated into a seamless pipeline powered by Streamlit + FastAPI.

Features
Conversational UI: Friendly Streamlit app guides users through model training.

Dynamic Model Type Selection: Automatically suggests Regression, Classification, or Boosting.

Model Building Pipeline: Cleans data, engineers new features, and trains machine learning models.

Evaluation Summary: Outputs performance metrics like R², MSE, MAE, Adjusted R².

AI Judge: Evaluates trained models and provides suggestions.

Feature Visualization: Displays feature importance using beautiful charts.

Model Download: Download trained models (.pkl) and feature weights (CSV).

Tech Stack
Frontend: Streamlit (for user interface)

Backend: FastAPI (Python 3)

ML Libraries: Scikit-learn, Pandas, NumPy

Visualization: Matplotlib, Seaborn

Model Persistence: joblib

Installation
Prerequisites
Make sure you have:

Python 3.8+

pip

(Optional) virtualenv for isolated environments

Steps
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/AutoModeler-AI.git
cd AutoModeler-AI
(Optional) Create and activate a virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
Install all dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Running the Application
Start the FastAPI Backend
bash
Copy
Edit
uvicorn model_api:app --host 0.0.0.0 --port 8000 --reload
This runs the backend server on: http://localhost:8000

Start the Streamlit Frontend
bash
Copy
Edit
streamlit run app.py
This opens the frontend on: http://localhost:8501

✅ Now you can upload your CSV file, select model types, and train!

Configuration
Adjust the parameters inside the Streamlit sidebar:

python
Copy
Edit
model_type = st.selectbox("Choose Model Type", ["Auto", "Regression", "Classification"])
api_base_url = st.text_input("API URL", value="http://localhost:8000")
enable_ai_judge = st.checkbox("Enable AI Judge", value=True)
Usage
Scenario 1 (Regression):
User: "I want to predict house prices."

AutoModeler: "Suggests Linear Regression."

Result: Trains model, shows R² and feature weights.

Scenario 2 (Classification):
User: "I want to classify houses into 'cheap' or 'expensive'."

AutoModeler: "Suggests Logistic Regression with binning."

Result: Trains logistic model, displays evaluation metrics.

Bonus:
Add engineered features like Income_per_room, Rooms_per_bedroom, House_Age_Squared.

Try boosting techniques.

Code Structure
bash
Copy
Edit
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
API Endpoints
POST /train_model: Train and return metrics.

POST /judge: Evaluate model and return scoring.

Sample Endpoint:

python
Copy
Edit
@app.post("/train_model")
def train_model(payload: dict):
    ...
Notes
Initially, saved_model.pkl will not exist.
It will be automatically created after the first model training.

All model files and weights can be downloaded through the Streamlit app.

Troubleshooting
Streamlit fails to run:

bash
Copy
Edit
pip install streamlit --upgrade
Backend server not running:

bash
Copy
Edit
uvicorn model_api:app --reload
CORS issues (frontend-backend communication):
Add CORSMiddleware to model_api.py if needed.

Contribution
Fork this repo 🍴

Create a new branch git checkout -b feature-name

Commit your changes

Push your changes

Create a Pull Request 📬

Contact
For queries and contributions, contact: venkatadurgakavya.bhatta@pace.edu
