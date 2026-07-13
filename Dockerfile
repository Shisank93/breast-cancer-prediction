FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and config files
COPY . .

# Install package locally
RUN pip install -e .

# Run pipeline execution to bake the model and scaler into the image
RUN python verify_pipeline_phase3.py

# Expose port for FastAPI
EXPOSE 8000

# Start uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
