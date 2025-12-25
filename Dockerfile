FROM german-credit-best-model

RUN pip install fastapi uvicorn

COPY app.py /app/app.py
WORKDIR /app

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]