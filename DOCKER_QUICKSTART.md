# 🚀 Docker Quick Start

**Deploy the entire arXiv Paper Curator in 3 commands!**

## Prerequisites

- Docker Desktop installed
- Groq API key ([Get free key](https://console.groq.com/))

## Quick Start

```bash
# 1. Configure environment
cp env.example .env
# Edit .env and set: GROQ_API_KEY=gsk_your_key_here

# 2. Start everything
docker-compose up

# Wait for: "✅ RAG pipeline initialized successfully!"
# First run: 10-15 minutes (downloads + indexes papers)
```

## Access Services

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:3000 | Main chat interface |
| **Backend API** | http://localhost:8000/docs | API documentation |
| **Qdrant** | http://localhost:6333/dashboard | Vector database |

## Common Commands

```bash
# Start in background
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop all services
docker-compose down

# Rebuild after changes
docker-compose up --build

# Re-index papers
docker-compose exec backend python scripts/fetch_and_index_papers.py
```

## Troubleshooting

### "Port already in use"
Change port in `docker-compose.yml`:
```yaml
frontend:
  ports:
    - "3001:3000"  # Use port 3001 instead
```

### Backend shows "Disconnected"
Wait 30-60 seconds for initialization. Check logs:
```bash
docker-compose logs backend | grep "initialized"
```

### No papers displayed
Check Qdrant has data:
```bash
# Visit: http://localhost:6333/dashboard
# Or manually re-index:
docker-compose exec backend python scripts/fetch_and_index_papers.py
```

## What Happens on First Run?

1. ✅ Downloads Docker images (2GB)
2. ✅ Builds backend and frontend containers
3. ✅ Starts Qdrant vector database
4. ✅ Backend waits for Qdrant to be ready
5. ✅ Automatically fetches 10 research papers from arXiv
6. ✅ Processes PDFs and creates embeddings
7. ✅ Indexes papers in vector database
8. ✅ Starts FastAPI server
9. ✅ Launches React frontend
10. ✅ Ready to use! 🎉

## Data Persistence

Your data survives container restarts:
- **Papers:** `./data/` directory
- **Vector DB:** `./qdrant_storage/` directory
- **Conversations:** Browser localStorage

## Optional: Langfuse Observability

Add to `.env` for query tracking:
```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

View traces at: https://cloud.langfuse.com

## Full Documentation

📖 [Complete Docker Guide](docs/DOCKER_DEPLOYMENT.md)

---

**That's it!** You now have a fully functional RAG research assistant. 🎓✨

