# Build from the REPO ROOT so common/ is in the context:
#   python serving/export_model.py          (downloads @champion into serving/model/)
#   docker build -f serving/Dockerfile -t telco-churn-api:local .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 MODEL_DIR=/srv/model

# LightGBM needs the OpenMP runtime, which slim images don't include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Dependencies first (cached layer): exact training versions + API libs
COPY serving/model/requirements.txt /tmp/model-requirements.txt
COPY serving/requirements.txt /tmp/api-requirements.txt
RUN pip install -r /tmp/model-requirements.txt -r /tmp/api-requirements.txt

COPY common/ ./common/
COPY serving/app/ ./app/
COPY serving/model/ ./model/

RUN useradd --uid 10001 --no-create-home appuser && chown -R appuser /srv
USER 10001

EXPOSE 8000
# 1 worker per pod: scale with Kubernetes replicas, not uvicorn workers
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]