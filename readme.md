# N8N + Ollama Docker Setup

This repository contains a Docker Compose setup for running N8N workflow automation platform alongside Ollama AI language models.

## Services

- **N8N**: Workflow automation platform accessible at http://localhost:5678
- **Ollama**: Local AI language model server accessible at http://localhost:11434

## Quick Start

### Automated Setup (Recommended)

Use the provided startup script that automatically handles permissions and starts all services:

```bash
./startup.sh
```

The script will:
- ✅ Stop any existing containers
- ✅ Create necessary data directories
- ✅ Set correct permissions for n8n data
- ✅ Verify docker-compose.yml configuration
- ✅ Start all services
- ✅ Check service accessibility

### Manual Setup

If you prefer manual setup:

1. Create data directories:
   ```bash
   mkdir -p data/n8n-data data/ollama
   ```

2. Set correct permissions:
   ```bash
   sudo chown -R 1000:1000 data/n8n-data
   ```

3. Start services:
   ```bash
   docker compose up -d
   ```

## Configuration

### N8N
- Data is persisted in `./data/n8n-data/` (local volume)
- Runs with user ID 1000:1000 for proper file permissions
- SQLite database for data storage
- No basic authentication enabled by default

### Ollama
- Models are stored in `./data/ollama/`
- Automatically pulls the Gemma 7B model on startup
- Configured for CPU execution with memory limits (4-10GB)
- Health check ensures model is loaded before marking as healthy

## Usage

### Start Services
```bash
./startup.sh              # Recommended - handles permissions
# OR
docker compose up -d       # Manual start
```

### Stop Services
```bash
docker compose down
```

### View Logs
```bash
docker compose logs -f
```

### Restart Services
```bash
./startup.sh              # Recommended - ensures permissions are correct
```

## Integration Notes

- **N8N ↔ Ollama**: Use `http://ollama_model:11434` as the Ollama URL within N8N workflows (internal Docker network)
- **External Access**: Use `http://localhost:11434` for external access to Ollama API
- **N8N Access**: Access N8N directly at `http://localhost:5678` (no Firefox container needed)

## Troubleshooting

### Permission Issues
The startup script automatically handles permission issues. If you encounter "EACCES: permission denied" errors:

1. Use the startup script: `./startup.sh`
2. Or manually fix permissions:
   ```bash
   sudo chown -R 1000:1000 data/n8n-data
   chmod 755 data/n8n-data
   ```

### Ollama Model Issues
- Ollama container may take time to pull the Gemma 7B model
- If model pulling fails, manually run:
  ```bash
  docker exec -it ollama_model ollama pull gemma:7b
  docker compose restart ollama
  ```

### Container Issues
- Check container status: `docker compose ps`
- View logs: `docker compose logs [service_name]`
- Restart specific service: `docker compose restart [service_name]`

### Clean Start
To start completely fresh:
```bash
docker compose down
sudo rm -rf data/n8n-data/*
./startup.sh
```

## File Structure
```
.
├── docker-compose.yml    # Docker services configuration
├── startup.sh           # Automated setup script
├── readme.md            # This file
└── data/
    ├── n8n-data/        # N8N data (workflows, database, etc.)
    └── ollama/          # Ollama models and data
```

## References

- [N8N + Ollama Tutorial Video](https://www.youtube.com/watch?v=mnV-lXxaFhk)
- [N8N Self-hosted AI Starter Kit](https://github.com/n8n-io/self-hosted-ai-starter-kit)

## Next time
- Use smaller models such as 
    - Tinyllama 1.1B
    - Gemma 2.2B
    - Phi3 3.8B
    - Qwen3 0.6B, 1.5B

- Import the `n8n_sample_ollama_workflow.json` file into the n8n workflow to get a test workflow ready

<hr/>
- the docker-compose-fun.yml file has some fun containers, not particularly usful for learning.
- https://www.howtogeek.com/self-hosted-alternatives-to-google-photos/

<hr/>
#Fun contaiers
- https://www.marktechpost.com/2025/10/25/how-to-build-a-fully-functional-computer-use-agent-that-thinks-plans-and-executes-virtual-actions-using-local-ai-models/
- https://www.marktechpost.com/2025/10/20/meet-langchains-deepagents-library-and-a-practical-example-to-see-how-deepagents-actually-work-in-action/
- https://dev.to/tyaga001/5-must-know-open-source-repositories-to-build-cool-ai-apps-3pn7


#langchain:
- Use Langchain folder for langchain related tuts
- Use LangSmith for any debugging needs
   - | set following 3 env vars to debug
         LANGSMITH_TRACING=true
         LANGSMITH_API_KEY # Create a new one or use one from the local.env file.
         LANGSMITH_PROJECT #its optional but get it from langsmith login y..72@hot...

The repo also contains code for small language model generation and hosting it.
put pdf files under input_pdfs folder, and run the run_training.sh file.
the apis can be then accessed at http://localhost:8000/docs

The model is not accurate at-all.
Once model is generated, we only need to run the API container! (bcos the model generated is used in the api container)