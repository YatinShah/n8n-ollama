# Use a slim official Python runtime base image
FROM python:3.11-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the necessary Python packages (CPU only)
# We ensure no GPU dependencies are installed implicitly.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main training script into the container
COPY train_slm.py .
COPY sample_data/ input_pdfs/

# Command to run the application when the container starts
CMD ["python", "train_slm.py"]
