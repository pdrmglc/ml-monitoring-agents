FROM python:3.11-slim

RUN pip install mlflow psycopg2-binary boto3 google-cloud-storage

WORKDIR /app

COPY start.sh .

RUN chmod +x start.sh

CMD ["./start.sh"]