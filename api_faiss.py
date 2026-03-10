#!/usr/bin/env python3
"""
ArchaeoFinder Backend API - Mit FAISS Integration
Schneller Endpunkt für Ähnlichkeitssuche
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import os

# FAISS Search importieren
from faiss_search import FAISSearch

app = FastAPI(title="ArchaeoFinder API", version="2.0.0")

# Globaler FAISS Index (wird einmalig geladen)
searcher = None

@app.on_event("startup")
async def load_faiss_index():
    """Lädt FAISS Index beim Server-Start"""
    global searcher
    print("🚀 Lade FAISS Index...")
    searcher = FAISSearch()
    searcher.build_index()  # Lägt bestehenden oder baut neu
    print(f"✅ FAISS bereit: {searcher.index.ntotal} Vektoren")

# Request/Response Models
class SearchRequest(BaseModel):
    embedding: List[float]  # 768-d oder 512-d Vektor vom Frontend
    top_k: int = 50
    threshold: float = 0.60
    filters: Optional[dict] = None  # Optional: Epoche, Region, etc.

class SearchResult(BaseModel):
    id: str
    similarity: float
    image_url: str
    title: str
    source: str
    metadata: dict

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    query_time_ms: float

@app.post("/api/search", response_model=SearchResponse)
async def search_similar(request: SearchRequest):
    """
    Ultraschnelle Ähnlichkeitssuche
    
    Beispiel-Request:
    {
        "embedding": [0.23, -0.15, 0.88, ...],  // 768 Zahlen
        "top_k": 50,
        "threshold": 0.60
    }
    """
    import time
    start_time = time.time()
    
    if searcher is None:
        raise HTTPException(status_code=503, detail="Search index not ready")
    
    # 1. FAISS Suche (ultraschnell: ~10ms)
    faiss_results = searcher.search(
        query_embedding=request.embedding,
        top_k=request.top_k * 2,  // Mehr holen für Filterung
        threshold=request.threshold
    )
    
    if not faiss_results:
        return SearchResponse(results=[], total=0, query_time_ms=0)
    
    # 2. IDs extrahieren
    ids = [r[0] for r in faiss_results]
    
    # 3. Vollständige Daten aus Supabase holen
    # (nur für die gefundenen IDs - viel schneller als alles zu laden)
    from supabase import create_client
    
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY")
    )
    
    response = supabase.table("fibula_embeddings")\
        .select("id, image_url, title, source, period, region, material")\
        .in_("id", ids[:100])\  # Limit für Performance
        .execute()
    
    # 4. Optional: Filter anwenden (Epoche, Region...)
    results = []
    for item in response.data:
        # Finde zugehörige Similarity-Score
        similarity = next((s for i, s in faiss_results if i == item['id']), 0)
        
        # Filter prüfen
        if request.filters:
            if request.filters.get('period') and item.get('period') != request.filters['period']:
                continue
            if request.filters.get('region') and item.get('region') != request.filters['region']:
                continue
        
        results.append(SearchResult(
            id=item['id'],
            similarity=similarity,
            image_url=item['image_url'],
            title=item['title'],
            source=item['source'],
            metadata={
                'period': item.get('period'),
                'region': item.get('region'),
                'material': item.get('material')
            }
        ))
    
    # Nach Similarity sortieren
    results.sort(key=lambda x: x.similarity, reverse=True)
    results = results[:request.top_k]  // Auf gewünschte Anzahl begrenzen
    
    query_time = (time.time() - start_time) * 1000
    
    return SearchResponse(
        results=results,
        total=len(results),
        query_time_ms=round(query_time, 2)
    )

@app.post("/api/search-by-image-id")
async def search_by_image_id(image_id: str, top_k: int = 10):
    """Finde ähnliche Bilder zu einem bestehenden Eintrag"""
    if searcher is None:
        raise HTTPException(status_code=503, detail="Search index not ready")
    
    results = searcher.search_similar_images(image_id, top_k=top_k)
    return {"results": results}

@app.get("/api/search/stats")
async def search_stats():
    """Statistiken über den Index"""
    if searcher is None:
        return {"status": "not_ready"}
    
    return {
        "status": "ready",
        "total_vectors": searcher.index.ntotal,
        "dimension": searcher.dimension,
        "index_size_mb": os.path.getsize(searcher.index_path) / (1024*1024)
    }

@app.post("/api/index/rebuild")
async def rebuild_index():
    """Rebuild FAISS Index (nach vielen neuen Einträgen)"""
    global searcher
    searcher.build_index(force_rebuild=True)
    return {"status": "rebuilt", "total_vectors": searcher.index.ntotal}

# Health Check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "faiss_ready": searcher is not None,
        "vectors": searcher.index.ntotal if searcher else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
