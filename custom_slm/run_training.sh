#!/bin/bash

set -e

echo "Starting automated SLM training and API deployment process..."

# 1. Create necessary directories
echo "Ensuring input_pdfs/ and output_models/ directories exist..."
mkdir -p input_pdfs
mkdir -p output_models
mkdir -p api # Create the directory for the API files

# 2. Build the Docker images
echo "Building Docker images for trainer and api services..."
docker compose build

# 3. Run the training process (trainer service)
echo "Running the training container..."
# We only run the 'trainer' service using 'docker-compose up', stopping immediately after it finishes
docker compose up --abort-on-container-exit trainer

# 4. Verify model existence
if [ -f output_models/custom_slm_model_cpu.pth ]; then
    echo "Model successfully exported to output_models/custom_slm_model_cpu.pth"
else
    echo "ERROR: Model file not found in output_models/ after training."
    exit 1
fi

# 5. Start the API service in the background
echo "Starting the API container in detached mode (-d)..."
docker compose up -d api

echo "API is running. Check documentation at http://localhost:8000/docs"
echo "Health check at http://localhost:8000/health"
echo "To stop the API, run: docker-compose down"
echo "Automated process completed."