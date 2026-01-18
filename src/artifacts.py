from pathlib import Path
import joblib

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

def save_artifacts(model, vectorizer):
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "model.pkl")
    joblib.dump(vectorizer, ARTIFACTS_DIR / "vectorizer.pkl")

def load_artifacts():
    model = joblib.load(ARTIFACTS_DIR / "model.pkl")
    vectorizer = joblib.load(ARTIFACTS_DIR / "vectorizer.pkl")
    return model, vectorizer