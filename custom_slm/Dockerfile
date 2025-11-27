# Use a slim official Python runtime base image
FROM python:3.11-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the necessary Python packages (CPU only)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main training script into the container
COPY train_slm.py .

# We only need the input pdfs directory structure for the trainer service
# The 'run_training.sh' script ensures this directory exists on the host.
# The volume mount in docker-compose handles the actual data transfer during runtime.
