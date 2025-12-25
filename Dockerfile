FROM rafeeeda/german_credit_model:latest

RUN pip install --no-cache-dir fastapi uvicorn

WORKDIR /app
COPY app.py /app/app.py

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
