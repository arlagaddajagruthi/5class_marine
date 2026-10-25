FROM python:3.12-slim
ENV NUMBA_CACHE_DIR=/tmp/numba PYTHONUNBUFFERED=1 ONE_MODEL_IN_MEMORY=1 CACHE_SIZE=2
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# one worker: the analysed-scene cache lives in process memory; Render sets $PORT
CMD gunicorn -w 1 --threads 4 -t 300 -b 0.0.0.0:${PORT:-10000} app:app
