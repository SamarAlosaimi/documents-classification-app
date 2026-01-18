from fastapi import FastAPI
from pydantic import BaseModel

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