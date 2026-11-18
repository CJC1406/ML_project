"""Small prediction API boundary; labels remain server-side."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="PathSnap API", version="0.1.0")


class PredictionRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, str]:
    # Deliberately fail closed until real data and a trained artifact are installed.
    raise HTTPException(503, detail=f"No trained model is installed for trip {request.trip_id}")
