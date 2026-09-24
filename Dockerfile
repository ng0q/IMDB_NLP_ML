FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/

RUN mkdir -p data model

RUN KAGGLEHUB_CACHE=/app/data python -c "\
import kagglehub, shutil; \
from pathlib import Path; \
p = kagglehub.dataset_download('lakshmi25npathi/imdb-dataset-of-50k-movie-reviews'); \
shutil.copy(Path(p) / 'IMDB Dataset.csv', Path('data') / 'IMDB Dataset.csv')" \
    && python -m src.train

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
