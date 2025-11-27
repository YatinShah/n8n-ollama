# Use a production-ready FastAPI image provided by Tiangolo
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11-slim

WORKDIR /app

# Copy API requirements and install them
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code (main.py) into the container
COPY api/main.py main.py

# The base image already sets up uvicorn command correctly. 
# We don't need to specify CMD here unless we change the file names.
# Default command: uvicorn main:app --host 0.0.0.0 --port 80
