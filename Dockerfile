FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY sql ./sql
COPY config ./config
COPY scripts ./scripts
COPY powerbi ./powerbi

CMD ["python", "-m", "src.data.loader"]
