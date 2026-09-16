FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 ANONYMIZED_TELEMETRY=False
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples
RUN pip install --no-cache-dir .
EXPOSE 8000
CMD ["uvicorn", "rag_workbench.api:app", "--host", "0.0.0.0", "--port", "8000"]
