#!/bin/bash
set -e

echo "============================================================"
echo "🚀 Starting arXiv Paper Curator Backend"
echo "============================================================"

# Wait for Qdrant to be ready
echo ""
echo "⏳ Waiting for Qdrant to be ready..."
QDRANT_HOST=${QDRANT_HOST:-localhost}
QDRANT_PORT=${QDRANT_PORT:-6333}
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://${QDRANT_HOST}:${QDRANT_PORT}/healthz" > /dev/null 2>&1; then
        echo "✅ Qdrant is ready!"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "   Attempt $RETRY_COUNT/$MAX_RETRIES - Qdrant not ready yet, waiting..."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "❌ Failed to connect to Qdrant after $MAX_RETRIES attempts"
    exit 1
fi

# Check if collection exists and has papers
echo ""
echo "🔍 Checking if papers are indexed..."

set +e  # Temporarily disable exit on error
python3 << EOF
import os
import sys
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

try:
    client = QdrantClient(
        host=os.getenv('QDRANT_HOST', 'localhost'),
        port=int(os.getenv('QDRANT_PORT', '6333'))
    )
    
    collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'arxiv_papers')
    
    try:
        collection_info = client.get_collection(collection_name)
        points_count = collection_info.points_count
        
        if points_count == 0:
            print(f"⚠️  Collection '{collection_name}' exists but is empty")
            sys.exit(2)  # Exit code 2 = needs indexing
        else:
            print(f"✅ Found {points_count} indexed papers in collection '{collection_name}'")
            sys.exit(0)  # Exit code 0 = all good
            
    except UnexpectedResponse:
        print(f"⚠️  Collection '{collection_name}' does not exist")
        sys.exit(2)  # Exit code 2 = needs indexing
        
except Exception as e:
    print(f"❌ Error checking collection: {e}")
    sys.exit(1)  # Exit code 1 = error
EOF

CHECK_RESULT=$?
set -e  # Re-enable exit on error

if [ $CHECK_RESULT -eq 2 ]; then
    echo ""
    echo "📥 No papers found. Starting automatic indexing..."
    echo "   This may take 5-10 minutes on first run..."
    echo ""
    
    # Run indexing script
    set +e  # Don't exit if indexing fails
    python3 scripts/fetch_and_index_papers.py
    INDEXING_RESULT=$?
    set -e
    
    if [ $INDEXING_RESULT -eq 0 ]; then
        echo ""
        echo "✅ Papers indexed successfully!"
    else
        echo ""
        echo "⚠️  Indexing failed, but continuing to start server..."
        echo "   You can manually index papers later with:"
        echo "   docker-compose exec backend python scripts/fetch_and_index_papers.py"
    fi
elif [ $CHECK_RESULT -eq 0 ]; then
    echo "✅ Papers already indexed, skipping auto-indexing"
else
    echo "⚠️  Could not verify collection status, starting server anyway..."
fi

# Start FastAPI server
echo ""
echo "============================================================"
echo "🚀 Starting FastAPI Server"
echo "============================================================"
echo ""

exec python3 -m src.api.server

