from fastapi import FastAPI
from pydantic import BaseModel
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.artifacts import load_artifacts
app = FastAPI(title="Documents classification API")

model, vectorizer = load_artifacts()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    X = vectorizer.transform([req.text])
    pred = model.predict(X)[0]
    return {"label": str(pred)}