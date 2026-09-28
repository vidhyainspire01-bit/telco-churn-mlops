# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Evaluate & promote (the gate)
# MAGIC 1. Register the best training run in Unity Catalog -> alias `@challenger`
# MAGIC 2. Re-score challenger AND current `@champion` on the SAME holdout (current data)
# MAGIC 3. Gate: absolute floor + recall guardrail + must beat champion by `min_improvement`
# MAGIC 4. Pass -> move `@champion` to the new version (old one kept as `@previous` for rollback)

# COMMAND ----------

%pip install -q lightgbm==4.5.0
%restart_python

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath(".."))

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from common.features import FEATURES, TARGET, split
from common.gate import GateConfig, decide, evaluate

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_ml")
dbutils.widgets.text("schema", "telco")
dbutils.widgets.text("model_name", "churn_model")
dbutils.widgets.text("best_run_id", "")            # empty -> taken from train task / latest tagged run
dbutils.widgets.text("fail_on_regression", "false")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_fqn = f"{catalog}.{schema}.{dbutils.widgets.get('model_name')}"
fail_on_regression = dbutils.widgets.get("fail_on_regression").lower() == "true"

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]
experiment_path = f"/Users/{user}/telco_churn_{catalog}"

# COMMAND ----------

# MAGIC %md ## 1. Find the best run and register it as challenger

# COMMAND ----------

run_id = dbutils.widgets.get("best_run_id")
if not run_id:
    try:
        run_id = dbutils.jobs.taskValues.get(taskKey="train", key="best_run_id", debugValue="")
    except Exception:
        run_id = ""
if not run_id:   # interactive fallback: latest run tagged by 02_train
    runs = mlflow.search_runs(experiment_names=[experiment_path],
                              filter_string="tags.best_candidate = 'true'",
                              order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        raise ValueError("No run tagged best_candidate=true. Run 02_train first.")
    run_id = runs.iloc[0]["run_id"]

run = client.get_run(run_id)
threshold = float(run.data.params["threshold"])
candidate = run.data.params["candidate"]
print(f"run_id={run_id} candidate={candidate} threshold={threshold}")

mv = mlflow.register_model(f"runs:/{run_id}/model", model_fqn)
version = mv.version
client.set_model_version_tag(model_fqn, version, "threshold", str(threshold))
client.set_model_version_tag(model_fqn, version, "candidate", candidate)
client.set_registered_model_alias(model_fqn, "challenger", version)
print(f"registered {model_fqn} v{version} as @challenger")

# COMMAND ----------

# MAGIC %md ## 2. Score challenger and champion on the same holdout

# COMMAND ----------

df = spark.table(f"{catalog}.{schema}.silver_customers").toPandas()
_, test_df = split(df)
X_test, y_test = test_df[FEATURES], test_df[TARGET]

challenger_model = mlflow.sklearn.load_model(f"models:/{model_fqn}@challenger")
challenger_metrics = evaluate(challenger_model, X_test, y_test, threshold)

try:
    champ_mv = client.get_model_version_by_alias(model_fqn, "champion")
    champ_threshold = float(champ_mv.tags.get("threshold", 0.5))
    champion_model = mlflow.sklearn.load_model(f"models:/{model_fqn}@champion")
    champion_metrics = evaluate(champion_model, X_test, y_test, champ_threshold)
    champion_version = champ_mv.version
except MlflowException:
    champ_mv, champion_metrics, champion_version = None, None, None

print("challenger:", {k: round(v, 4) for k, v in challenger_metrics.items()})
print(f"champion (v{champion_version}):",
      None if champion_metrics is None else {k: round(v, 4) for k, v in champion_metrics.items()})

# COMMAND ----------

# MAGIC %md ## 3. Gate decision and alias move

# COMMAND ----------

promote, reason = decide(challenger_metrics, champion_metrics, GateConfig())

client.set_model_version_tag(model_fqn, version, "gate_decision", "promoted" if promote else "rejected")
client.set_model_version_tag(model_fqn, version, "gate_reason", reason)
for k, v in challenger_metrics.items():
    client.set_model_version_tag(model_fqn, version, f"holdout_{k}", f"{v:.4f}")

if promote:
    if champion_version is not None:
        client.set_registered_model_alias(model_fqn, "previous", champion_version)   # rollback target
    client.set_registered_model_alias(model_fqn, "champion", version)
    print(f"PROMOTED v{version} -> @champion. {reason}")
else:
    print(f"REJECTED v{version}. {reason}. @champion stays on v{champion_version}")
    if fail_on_regression:
        raise RuntimeError(f"Gate rejected challenger: {reason}")

try:
    dbutils.jobs.taskValues.set(key="promoted", value=promote)
except Exception:
    pass