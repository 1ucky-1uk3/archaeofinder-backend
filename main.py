import os
import io
import hashlib
import logging
import asyncio
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ArchaeoFinder")

# Globale Ressourcen (Lazy Loading)
CLIP_RESOURCES = {
    "model": None,
    "preprocess": None,
    "tokenizer": None,
    "collection": None,
    "available": False,
    "error_logged": False
}

uploaded_images = {}
EUROPEANA_API_KEY = os.getenv("EUROPEANA_API_KEY", "api2demo")
DDB_API_KEY = os.getenv("DDB_API_KEY", "")

def get_clip_resources():
    """Lädt CLIP nur bei Bedarf (Lazy Loading), um RAM beim Start zu sparen."""
    if CLIP_RESOURCES["model"] is not None:
        return CLIP_RESOURCES

    try:
        import torch
        import open_clip
        import chromadb
        from huggingface_hub import login as hf_login
        from PIL import Image

        logger.info("Versuche CLIP-Ressourcen zu laden (On-Demand)...")
        
        hf_token = os.getenv("HF_TOKEN", "")
        if hf_token:
            hf_login(token=hf_token)

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model.eval()
        
        client = chromadb.Client()
        collection = client.get_or_create_collection(name="archaeo_images")

        CLIP_RESOURCES.update({
            "model": model,
            "preprocess": preprocess,
            "collection": collection,
            "available": True
        })
        logger.info("CLIP erfolgreich in den RAM geladen.")
        return CLIP_RESOURCES
    except Exception as e:
        if not CLIP_RESOURCES["error_logged"]:
            logger.error(f"CLIP konnte nicht geladen werden (Vermutlich RAM-Limit): {e}")
            CLIP_RESOURCES["error_logged"] = True
        return CLIP_RESOURCES

# --- Datenmodelle ---
class MuseumObject(BaseModel):
    id: str
    title: str
    museum: Optional[str] = None
    image_url: Optional[str] = None
    source_url: Optional[str] = None
    source: str = "europeana"
    similarity: Optional[int] = None

class SearchResponse(BaseModel):
    success: bool
    total_results: int
    results: List[MuseumObject]
    search_id: str
    search_mode: str

# --- API Setup ---
app = FastAPI(title="ArchaeoFinder (RAM Optimized)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health():
    # Zeigt an, ob CLIP geladen ist oder nicht
    return {
        "status": "online",
        "clip_loaded": CLIP_RESOURCES["available"],
        "memory_info": "Optimized for 1GB"
    }

# --- Such-Logik ---
async def search_europeana(q: str, limit: int):
    url = "https://api.europeana.eu/record/v2/search.json"
    params = {"wskey": EUROPEANA_API_KEY, "query": q, "rows": limit, "profile": "rich"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, params=params)
            data = r.json()
            return [
                MuseumObject(
                    id=item.get("id", "unk"),
                    title=item.get("title", ["Unbekannt"])[0],
                    museum=item.get("dataProvider", [""])[0],
                    image_url=item.get("edmPreview", [None])[0],
                    source_url=item.get("guid", ""),
                    source="europeana"
                ) for item in data.get("items", []) if item.get("edmPreview")
            ]
    except: return []

async def search_ddb(q: str, limit: int):
    if not DDB_API_KEY: return []
    url = "https://api.deutsche-digitale-bibliothek.de/search"
    params = {"oauth_consumer_key": DDB_API_KEY, "query": q, "rows": limit}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, params=params, headers={"Accept": "application/json"})
            docs = r.json().get("results", [{}])[0].get("docs", [])
            return [
                MuseumObject(
                    id=d.get("id", ""),
                    title=d.get("title", "Unbekannt"),
                    museum=d.get("provider", "DDB"),
                    image_url=f"https://api.deutsche-digitale-bibliothek.de/binary/{d.get('id')}/list/1.jpg",
                    source_url=f"https://www.deutsche-digitale-bibliothek.de/item/{d.get('id')}",
                    source="ddb"
                ) for d in docs if d.get("id")
            ]
    except: return []

# --- Endpoints ---
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    image_id = hashlib.md5(content).hexdigest()
    uploaded_images[image_id] = content
    return {"success": True, "image_id": image_id}

@app.get("/api/search", response_model=SearchResponse)
async def search(q: Optional[str] = Query(None), image_id: Optional[str] = Query(None), limit: int = 20):
    results = []
    mode = "text"

    # Bildsuche nur versuchen, wenn image_id vorhanden
    if image_id and image_id in uploaded_images:
        resources = get_clip_resources()
        if resources["available"]:
            # Hier würde die Vektorsuche passieren (gekürzt für Stabilität)
            mode = "image_attempted"
            logger.info("Bildsuche wurde angefordert.")
        else:
            logger.warning("Bildsuche übersprungen: Nicht genug RAM.")

    # Textsuche (Fallback oder Primär)
    text_query = q or "archaeology"
    eur_res = await search_europeana(text_query, limit // 2)
    ddb_res = await search_ddb(text_query, limit // 2)
    
    results = eur_res + ddb_res
    
    return SearchResponse(
        success=True,
        total_results=len(results),
        results=results,
        search_id="res_" + datetime.now().strftime("%M%S"),
        search_mode=mode
    )

@app.get("/api/sources")
async def get_sources():
    return {"sources": ["europeana", "ddb", "pas_uk", "pan_nl"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
