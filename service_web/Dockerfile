FROM python:3.9-slim
WORKDIR /app
COPY service_web/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY service_web/ ./service_web/
COPY service_model/ ./service_model/
WORKDIR /app/service_web
EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]