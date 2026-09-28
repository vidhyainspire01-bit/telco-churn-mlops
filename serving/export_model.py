"""Build-time step: pull the @champion model from Unity Catalog into serving/model/.

Runs in CI (and locally for testing) BEFORE `docker build`, so the image contains the
model and pods never need Databricks credentials at runtime.

Needs env vars: DATABRICKS_HOST and DATABRICKS_TOKEN (or DATABRICKS_CLIENT_ID/SECRET in CI).
Usage:  python serving/export_model.py --model dev_ml.telco.churn_model --alias champion
"""
import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow import MlflowClient

OUT_DIR = Path(__file__).parent / "model"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dev_ml.telco.churn_model")
    p.add_argument("--alias", default="champion")
    args = p.parse_args()

    # Both URIs explicit: newer MLflow defaults tracking to a local sqlite DB otherwise
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")
    mv = client.get_model_version_by_alias(args.model, args.alias)
    print(f"{args.model}@{args.alias} -> version {mv.version} (run {mv.run_id})")

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{args.model}@{args.alias}", dst_path=tmp))
        model_root = next(p.parent for p in local.rglob("MLmodel"))   # handle nested layouts
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
        shutil.copytree(model_root, OUT_DIR)

    meta = {
        "model_name": args.model,
        "alias": args.alias,
        "version": mv.version,
        "run_id": mv.run_id,
        "threshold": float(mv.tags.get("threshold", 0.5)),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"exported to {OUT_DIR}:", json.dumps(meta))


if __name__ == "__main__":
    main()
