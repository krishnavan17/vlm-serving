# vlm-serving
vlm serving using ollama backend

## Run Ollama with GPU (Docker Compose)

### Prerequisites
- NVIDIA GPU with recent drivers installed
- Docker Engine and Docker Compose plugin
- NVIDIA Container Toolkit installed and configured for Docker
- `jq` installed (used in cURL examples)

Quick check:
- `nvidia-smi`
- `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`

### Start Ollama
- `docker compose up -d`
- `docker compose logs -f ollama`

Ollama API is available at `http://localhost:11434`.

### Pull and test a model
- `docker exec -it ollama sh -lc 'OLLAMA_HOST=http://127.0.0.1:11434 ollama pull llama3.2'`
- `curl -s http://localhost:11434/api/generate -d '{"model":"llama3.2","prompt":"Hello","stream":false}' | jq .`

### Helper script (pull any model)
- `./utils/pull_model.sh <model-name>`
- Example: `./utils/pull_model.sh qwen2.5:7b`

The script checks that Docker is available, verifies `docker-compose.yml` exists, ensures the `ollama` service is running, and then pulls the model inside the container.

### Stop
- `docker compose down`

## Run full stack with Docker Compose (Ollama + VLM API)

Build and start both services:
- `docker compose up -d --build`

Check service status:
- `docker compose ps`
- `docker compose logs -f vlm-serving`

API base URL: `http://localhost:8000`

Quick checks:
- `curl -s http://localhost:8000/health | jq .`
- `curl -s -X POST http://localhost:8000/describe -H "Content-Type: application/json" -d '{"image_path":"/home/user/screenshot.png"}' | jq .`
- `curl -s -X POST http://localhost:8000/describe -F "image=@/home/user/screenshot.png" | jq .`

Notes:
- `vlm-serving` mounts `/home/user` from host as read-only, so paths like `/home/user/screenshot.png` work in API requests.
- If you change compose config, recreate containers: `docker compose up -d --build --force-recreate`
- If dependencies change (for example `python-multipart` for upload endpoints), rebuild containers: `docker compose up -d --build --force-recreate`

Stop all services:
- `docker compose down`

## Run FastAPI app

From the repo root:
- `set -a; source /etc/environment; set +a`
- `python3 -m pip install -r vlm-serving/requirement.txt`
- `uvicorn vlm-serving.main:app --host 0.0.0.0 --port 8000 --reload`

API base URL: `http://localhost:8000`

Set a reusable image path for examples:
- `export IMG_PATH="/home/user/screenshot.png"`

### cURL examples

Health:
- `curl -s http://localhost:8000/health | jq .`

Describe image:
- `curl -s -X POST http://localhost:8000/describe -H "Content-Type: application/json" -d '{"image_path":"'"$IMG_PATH"'"}' | jq .`

Describe image (upload file directly on same API):
- `curl -s -X POST http://localhost:8000/describe -F "image=@$IMG_PATH" | jq .`

Locate reference in image:
- `curl -s -X POST http://localhost:8000/locate-reference -H "Content-Type: application/json" -d '{"image_path":"'"$IMG_PATH"'","reference":"Intel"}' | jq .`

Locate reference in image (upload file directly on same API):
- `curl -s -X POST http://localhost:8000/locate-reference -F "image=@$IMG_PATH" -F "reference=Intel" | jq .`

Locate-reference response format:
- Returns model output as JSON text containing bounding box fields: `reference`, `found`, and `bbox` with `x`, `y`, `width`, `height` (normalized `0..1`).

Extract text (OCR):
- `curl -s -X POST http://localhost:8000/extract-text -H "Content-Type: application/json" -d '{"image_path":"'"$IMG_PATH"'"}' | jq .`

Extract text (upload file directly on same API):
- `curl -s -X POST http://localhost:8000/extract-text -F "image=@$IMG_PATH" | jq .`

Summarize document image:
- `curl -s -X POST http://localhost:8000/summarize -H "Content-Type: application/json" -d '{"image_path":"'"$IMG_PATH"'"}' | jq .`

Summarize document image (upload file directly on same API):
- `curl -s -X POST http://localhost:8000/summarize -F "image=@$IMG_PATH" | jq .`

Answer question about image:
- `curl -s -X POST http://localhost:8000/answer-question -H "Content-Type: application/json" -d '{"image_path":"'"$IMG_PATH"'","question":"What is this image about?"}' | jq .`

Answer question about image (upload file directly on same API):
- `curl -s -X POST http://localhost:8000/answer-question -F "image=@$IMG_PATH" -F "question=What is this image about?" | jq .`

Extract structured fields:
- `curl -s -X POST http://localhost:8000/extract-fields -H "Content-Type: application/json" -d '{"image_path":"'"$IMG_PATH"'","fields":["title","date","total"]}' | jq .`

Extract structured fields (upload file directly on same API):
- `curl -s -X POST http://localhost:8000/extract-fields -F "image=@$IMG_PATH" -F "fields=title,date,total" | jq .`

Detailed caption:
- `curl -s -X POST http://localhost:8000/caption-detailed -H "Content-Type: application/json" -d '{"image_path":"'"$IMG_PATH"'"}' | jq .`

Detailed caption (upload file directly on same API):
- `curl -s -X POST http://localhost:8000/caption-detailed -F "image=@$IMG_PATH" | jq .`
