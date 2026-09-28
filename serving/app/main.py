import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Response

from app.schemas import BatchRequest, BatchResponse, CustomerFeatures, Prediction
from common.features import FEATURES

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "model"))

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
log = logging.getLogger("churn-api")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model is baked into the image at build time: no registry call, no credentials at runtime.
    state["meta"] = json.loads((MODEL_DIR / "meta.json").read_text())
    state["model"] = mlflow.sklearn.load_model(str(MODEL_DIR))
    log.info(json.dumps({"event": "model_loaded", **state["meta"]}))
    yield
    state.clear()


app = FastAPI(title="Telco Churn API", lifespan=lifespan)


@app.get("/health")            # liveness: process is up
def health():
    return {"status": "ok"}


@app.get("/ready")             # readiness: only route traffic once the model is loaded
def ready(response: Response):
    if "model" not in state:
        response.status_code = 503
        return {"ready": False}
    m = state["meta"]
    return {"ready": True, "model": m["model_name"], "version": m["version"], "threshold": m["threshold"]}


def _score(customers: List[CustomerFeatures]) -> List[Prediction]:
    meta = state["meta"]
    threshold = meta["threshold"]
    df = pd.DataFrame([c.model_dump() for c in customers])[FEATURES]   # enforce training column order
    start = time.perf_counter()
    proba = state["model"].predict_proba(df)[:, 1]
    latency_ms = (time.perf_counter() - start) * 1000

    preds = [Prediction(churn_probability=round(float(p), 6), churn_prediction=int(p >= threshold),
                        threshold=threshold, model_version=str(meta["version"])) for p in proba]

    # One structured line per prediction -> collected by Container Insights; phase 9 drift job reads these.
    for row, pr in zip(df.to_dict(orient="records"), preds):
        log.info(json.dumps({"event": "prediction", "request_id": str(uuid.uuid4()),
                             "model_version": pr.model_version, "features": row,
                             "churn_probability": pr.churn_probability,
                             "latency_ms": round(latency_ms / len(preds), 3)}))
    return preds


@app.post("/predict", response_model=Prediction)
def predict(customer: CustomerFeatures):
    return _score([customer])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    return BatchResponse(predictions=_score(req.customers))
