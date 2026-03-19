FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

WORKDIR /app

# instala dependência do LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.prod.txt .

RUN pip install --no-cache-dir -r requirements.prod.txt

COPY app/ ./app/
COPY ml/ ./ml/

EXPOSE 8080

CMD ["uvicorn", "app.api:main", "--host", "0.0.0.0", "--port", "8080"]