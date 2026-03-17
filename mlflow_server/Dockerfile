FROM python:3.11-slim

WORKDIR /app

RUN pip install mlflow google-cloud-storage

COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]