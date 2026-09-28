# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Train: compare candidate models, log to MLflow
# MAGIC Each candidate is ONE sklearn Pipeline (encoding + model), so the logged model
# MAGIC accepts raw values like `Contract = "Two year"`. Serving needs no encoding logic.

# COMMAND ----------
%pip install -q lightgbm==4.5.0
%restart_python

# COMMAND ----------
 
import os, sys
sys.path.append(os.path.abspath(".."))   # repo root, so `common` is importable from the Git folder
 
import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
 
from common.features import CATEGORICAL, FEATURES, NUMERIC, TARGET, split
 
# COMMAND ----------
 
dbutils.widgets.text("catalog", "dev_ml")
dbutils.widgets.text("schema", "telco")
dbutils.widgets.text("target_recall", "0.75")   # business choice: catch at least 75% of churners
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
target_recall = float(dbutils.widgets.get("target_recall"))
 
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/telco_churn_{catalog}")
mlflow.set_registry_uri("databricks-uc")
 
df = spark.table(f"{catalog}.{schema}.silver_customers").toPandas()
train_df, test_df = split(df)
X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]
print(f"train={len(train_df)} test={len(test_df)} "
      f"churn train={y_train.mean():.3f} test={y_test.mean():.3f}")
 
# COMMAND ----------
 
def preprocessor(scale_numeric: bool) -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler() if scale_numeric else "passthrough", NUMERIC),
    ])
 
candidates = {
    "logreg_baseline": Pipeline([
        ("prep", preprocessor(scale_numeric=True)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]),
    "lightgbm": Pipeline([
        ("prep", preprocessor(scale_numeric=False)),
        ("model", LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, random_state=42, verbose=-1)),
    ]),
    "lightgbm_balanced": Pipeline([
        ("prep", preprocessor(scale_numeric=False)),
        ("model", LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, class_weight="balanced",
                                 random_state=42, verbose=-1)),
    ]),
}
 
# COMMAND ----------
 
def choose_threshold(pipe, X, y, target_recall: float) -> float:
    """Pick the decision threshold on TRAINING data only (5-fold out-of-fold predictions).
 
    Highest threshold that still reaches target_recall -> best precision at that recall.
    Never tune on the test set: that would leak and inflate the test metrics.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, oof)
    ok = recall[:-1] >= target_recall          # recall has one more element than thresholds
    return float(thresholds[ok].max()) if ok.any() else 0.5
 
 
def evaluate(model, X, y, threshold: float) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "test_pr_auc": average_precision_score(y, proba),   # threshold-free ranking quality
        "test_roc_auc": roc_auc_score(y, proba),
        "test_recall": recall_score(y, pred),              # at the chosen threshold
        "test_precision": precision_score(y, pred),
        "test_f1": f1_score(y, pred),
    }
 
results = []
for name, pipe in candidates.items():
    with mlflow.start_run(run_name=name) as run:
        threshold = choose_threshold(pipe, X_train, y_train, target_recall)
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test, threshold)
 
        mlflow.log_params({"candidate": name, "catalog": catalog, "n_train": len(X_train),
                           "n_test": len(X_test), "target_recall": target_recall,
                           "threshold": round(threshold, 4)})
        mlflow.log_params({f"model__{k}": v for k, v in pipe.named_steps["model"].get_params().items()
                           if isinstance(v, (int, float, str, bool)) or v is None})
        mlflow.log_metrics(metrics)
 
        signature = infer_signature(X_train, pipe.predict_proba(X_train)[:, 1])
        mlflow.sklearn.log_model(
            pipe, artifact_path="model",
            signature=signature,
            input_example=X_train.head(3),
            pyfunc_predict_fn="predict_proba",
        )
        results.append({"candidate": name, "run_id": run.info.run_id,
                        "threshold": round(threshold, 4), **metrics})
 
leaderboard = pd.DataFrame(results).sort_values("test_pr_auc", ascending=False)
display(leaderboard)
 
# COMMAND ----------
 
best = leaderboard.iloc[0]
with mlflow.start_run(run_id=best["run_id"]):
    mlflow.set_tag("best_candidate", "true")
 
print(f"best: {best['candidate']}  pr_auc={best['test_pr_auc']:.4f}  run_id={best['run_id']}")
 
try:   # hands the run_id to the evaluate/promote task when run as a job
    dbutils.jobs.taskValues.set(key="best_run_id", value=best["run_id"])
except Exception:
    pass
 