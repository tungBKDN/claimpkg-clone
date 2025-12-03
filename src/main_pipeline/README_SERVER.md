# ClaimPKG FastAPI Server

FastAPI server for the ClaimPKG claim verification pipeline with singleton pattern for efficient resource management.

## Features

- ✅ **Singleton Pattern**: Heavy resources (KG connector, LLMs, embeddings) loaded once on startup
- ✅ **Health Check**: Monitor server status and embedding availability
- ✅ **Claim Verification**: POST endpoint to verify claims using the full pipeline
- ✅ **Auto Documentation**: Interactive API docs at `/docs` and `/redoc`

## Installation

```bash
pip install fastapi uvicorn
```

## Quick Start

### 1. Start the Server

```bash
# From src/main_pipeline directory
cd src/main_pipeline
python server.py
```

Or with uvicorn directly:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 2. Access the API

- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### GET /health

Health check endpoint showing server status and embedding information.

**Response Example**:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-03T10:30:00",
  "initialized": true,
  "init_timestamp": "2025-12-03T10:25:00",
  "embeddings": {
    "relations": {
      "available": true,
      "count": 1234,
      "file_path": "D:\\path\\to\\kg_relations.txt"
    },
    "entities": {
      "available": true,
      "count": 1591850,
      "file_path": "D:\\path\\to\\kg_entities.txt"
    }
  }
}
```

### POST /verify

Verify a claim using the ClaimPKG pipeline.

**Request Body**:
```json
{
  "claim": "Hòa Bình Province is in the southwest of Hanoi.",
  "specialize_mode": "FEWSHOT",
  "retry": 3
}
```

**Parameters**:
- `claim` (required): The claim text to verify
- `specialize_mode` (optional): "FEWSHOT" (default) or "FINETUNE"
- `retry` (optional): Number of retries for LLM generation (1-10, default: 3)

**Response Example**:
```json
{
  "claim": "Hòa Bình Province is in the southwest of Hanoi.",
  "retrieved_triplets": "<e>Hòa Bình Province</e>||southwest||<e>Hanoi</e>\n<e>Hanoi</e>||capital||<e>Vietnam</e>\n",
  "final_answer": "The claim is SUPPORTED. Based on the knowledge graph...",
  "specialize_mode": "FEWSHOT",
  "processing_time_seconds": 5.23
}
```

## Usage Examples

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Verify claim
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "Hòa Bình Province is southwest of Hanoi.",
    "specialize_mode": "FEWSHOT",
    "retry": 3
  }'
```

### Using Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Verify claim
claim_data = {
    "claim": "Hòa Bình Province is southwest of Hanoi.",
    "specialize_mode": "FEWSHOT",
    "retry": 3
}
response = requests.post("http://localhost:8000/verify", json=claim_data)
result = response.json()
print(f"Answer: {result['final_answer']}")
```

### Using JavaScript fetch

```javascript
// Health check
fetch('http://localhost:8000/health')
  .then(res => res.json())
  .then(data => console.log(data));

// Verify claim
fetch('http://localhost:8000/verify', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    claim: "Hòa Bình Province is southwest of Hanoi.",
    specialize_mode: "FEWSHOT",
    retry: 3
  })
})
  .then(res => res.json())
  .then(data => console.log(data.final_answer));
```

## Configuration

Set environment variables in `.env`:

```env
# Knowledge Graph
KG_URI=bolt://localhost:7687
KG_USERNAME=neo4j
KG_PASSWORD=your_password
KG_NAME=neo4j

# Embeddings
EMBEDDING_STORAGE_PATH=D:\path\to\embeddings
EMBEDDING_FILENAME=kg_relations
ENTITY_EMBEDDING_FILENAME=kg_entities

# LLM API Keys
GENERAL_LLM_API_KEY=your_openai_key
SPECIALIZED_LLM_API_KEY=your_key
```

## Performance Notes

- **First Request**: May be slow while resources initialize (~30s - 5min depending on embedding size)
- **Subsequent Requests**: Fast (<10s) as all resources are cached in memory
- **Concurrent Requests**: Shares singleton resources efficiently

## Development

### Run in Development Mode (with auto-reload)

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Test Endpoints

```bash
# Install httpie for easier testing
pip install httpie

# Health check
http GET localhost:8000/health

# Verify claim
http POST localhost:8000/verify \
  claim="Hanoi is the capital of Vietnam." \
  specialize_mode="FEWSHOT" \
  retry:=3
```

## Troubleshooting

### Server stuck on startup
- Check if embeddings are being generated (can take 5-30 min for large KGs)
- Monitor console output for progress

### 503 Service Unavailable
- Server is still initializing, wait a moment and retry

### 500 Internal Server Error
- Check logs for detailed error message
- Verify all environment variables are set correctly
- Ensure Neo4j is running and accessible

## License

Same as ClaimPKG project.
