from fastapi import FastAPI
from pydantic import BaseModel
from trainer import train, save_model
from judge import evaluate_model  # ✅ make sure judge.py is imported

app = FastAPI()

class TrainRequest(BaseModel):
    data: dict
    model_type: str
    bin: bool | None = False

@app.post("/train_model")
def train_model(request: TrainRequest):
    try:
        model, model_type, metrics, ols_summary = train(
            request.data,
            request.model_type,
            request.bin
        )
        save_model(model)
        return {
            "message": "Model trained successfully.",
            "model_type": model_type,
            "metrics": metrics,
            "performance_summary": f"Model performed well with R² of {metrics.get('R2', 'N/A')}",
            "ols_summary": ols_summary
        }
    except Exception as e:
        return {"error": str(e)}

# ✅ ADD THIS NEW ROUTE BELOW
@app.post("/judge")
def judge_model(model_dict: dict):
    try:
        result = evaluate_model(model_dict)
        return result
    except Exception as e:
        return {"error": f"Judge failed: {str(e)}"}
