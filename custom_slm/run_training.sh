#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting automated SLM training process..."

# 1. Create necessary directories if they don't exist
echo "Ensuring input_pdfs/ and output_models/ directories exist..."
# mkdir -p input_pdfs
mkdir -p output_models

# Note: You must place at least one .pdf file into input_pdfs/ before running this script.

# 2. Build the Docker image using docker-compose
echo "Building Docker image 'custom-slm-trainer'..."
# The 'build' command builds or rebuilds services defined in compose.
docker compose build slm-trainer

# 3. Run the training process
echo "Running the training container (this might take a while)..."
# 'up' creates and starts containers. We use --abort-on-container-exit to stop once training finishes.
docker compose up --abort-on-container-exit slm-trainer

# 4. Success Message and verification
echo "Training process finished successfully."
echo "Checking for exported model file in output_models/..."

if [ -f output_models/custom_slm_model_cpu.pth ]; then
    echo "Model successfully exported to output_models/custom_slm_model_cpu.pth"
    ls -lh output_models/
else
    echo "ERROR: Model file not found in output_models/ after execution."
    exit 1
fi

echo "Automation script complete."
