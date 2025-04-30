from fastapi import FastAPI
from pydantic import BaseModel
from trainer import train, save_model
from judge import evaluate_model

app = FastAPI()

class TrainRequest(BaseModel):
    data: dict
    model_type: str
    bin: bool | None = False

@app.post("/train_model")
def train_model_api(request: TrainRequest):
    try:
        model, model_type, metrics, ols_summary, y_true, y_pred = train(
            request.data,
            request.model_type,
            request.bin
        )
        save_model(model)
        return {
            "message": "Model trained successfully.",
            "model_type": model_type,
            "metrics": metrics,
            "performance_summary": f"Model trained successfully",
            "ols_summary": ols_summary,
            "true_labels": y_true,
            "predicted_labels": y_pred
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/judge")
def judge_model(model_dict: dict):
    try:
        return evaluate_model(model_dict)
    except Exception as e:
        return {"error": f"Judge failed: {str(e)}"}
