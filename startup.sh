#!/bin/bash

# startup.sh - N8N + Ollama Docker Setup Script
# This script ensures proper permissions for n8n data directory and starts the services

set -e  # Exit on any error

echo "🚀 Starting N8N + Ollama Docker Setup..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker compose down 2>/dev/null || true

# Create data directories if they don't exist
echo "📂 Creating data directories..."
mkdir -p data/n8n-data
mkdir -p data/ollama

# Get the yatin user ID and group ID (the user that should own n8n data)
YATIN_UID=$(id -u yatin 2>/dev/null || echo "1000")
YATIN_GID=$(id -g yatin 2>/dev/null || echo "1000")

echo "👤 Setting up permissions for user yatin (UID: $YATIN_UID, GID: $YATIN_GID)..."

# Fix ownership of n8n-data directory to match the container user
if [ -d "data/n8n-data" ]; then
    echo "🔧 Setting ownership of data/n8n-data to $YATIN_UID:$YATIN_GID..."
    
    # Check if we need sudo for changing ownership
    if [ "$(stat -c %u data/n8n-data)" != "$YATIN_UID" ] || [ "$(stat -c %g data/n8n-data)" != "$YATIN_GID" ]; then
        if [ "$EUID" -eq 0 ]; then
            # Running as root
            chown -R "$YATIN_UID:$YATIN_GID" data/n8n-data
        else
            # Try without sudo first, fallback to sudo if needed
            if ! chown -R "$YATIN_UID:$YATIN_GID" data/n8n-data 2>/dev/null; then
                echo "⚠️  Need elevated privileges to change ownership..."
                sudo chown -R "$YATIN_UID:$YATIN_GID" data/n8n-data
            fi
        fi
        echo "✅ Ownership updated successfully"
    else
        echo "✅ Ownership already correct"
    fi
fi

# Set proper permissions
echo "🔒 Setting directory permissions..."
chmod -R 755 data/n8n-data 2>/dev/null || sudo chmod -R 755 data/n8n-data

# Verify the docker-compose.yml has the correct user configuration
echo "🔍 Verifying docker-compose.yml configuration..."
if grep -q "user: \"$YATIN_UID:$YATIN_GID\"" docker-compose.yml; then
    echo "✅ Docker compose user configuration is correct"
else
    echo "⚠️  Updating docker-compose.yml with correct user ID..."
    # Update the user line in docker-compose.yml
    sed -i "s/user: \"[0-9]*:[0-9]*\"/user: \"$YATIN_UID:$YATIN_GID\"/" docker-compose.yml
    echo "✅ Updated user configuration to $YATIN_UID:$YATIN_GID"
fi

# Start the services
echo "🐳 Starting Docker containers..."
docker compose up -d

# Wait a moment for containers to start
echo "⏳ Waiting for services to initialize..."
sleep 10

# Check container status
echo "📊 Container status:"
docker compose ps

# Check if n8n is accessible
echo "🔗 Checking n8n accessibility..."
for i in {1..30}; do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:5678 | grep -q "200"; then
        echo "✅ N8N is accessible at http://localhost:5678"
        break
    else
        echo "⏳ Waiting for n8n to be ready... (attempt $i/30)"
        sleep 2
    fi
done

# Check if ollama is accessible
echo "🤖 Checking Ollama accessibility..."
for i in {1..30}; do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:11434 | grep -q "200"; then
        echo "✅ Ollama is accessible at http://localhost:11434"
        break
    else
        echo "⏳ Waiting for Ollama to be ready... (attempt $i/30)"
        sleep 2
    fi
done

echo ""
echo "🎉 Setup complete!"
echo "📱 N8N Web UI: http://localhost:5678"
echo "🤖 Ollama API: http://localhost:11434"
echo ""
echo "💡 To view logs: docker compose logs -f"
echo "🛑 To stop: docker compose down"
echo "🔄 To restart: ./startup.sh"