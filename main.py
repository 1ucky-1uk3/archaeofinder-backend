from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx
import os
from io import BytesIO
from PIL import Image
import base64

# CLIP imports
try:
    import torch
    import open_clip
    import chromadb
    import numpy as np
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

app = FastAPI(title="ArchaeoFinder API")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Europeana API Configuration
EUROPEANA_API_KEY = os.getenv("EUROPEANA_API_KEY", "")
EUROPEANA_SEARCH_URL = "https://api.europeana.eu/record/v2/search.json"

# CLIP globals
clip_model = None
clip_preprocess = None
clip_tokenizer = None
chroma_client = None
image_collection = None

# Models
class SearchQuery(BaseModel):
    query: str
    rows: int = 12
    start: int = 1

class ImageIndexRequest(BaseModel):
    image_url: str
    metadata: Optional[dict] = None

# Initialize CLIP
def initialize_clip():
    global clip_model, clip_preprocess, clip_tokenizer, chroma_client, image_collection
    if not CLIP_AVAILABLE:
        return False
    try:
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        clip_model.eval()
        
        # PersistentClient für dauerhafte Speicherung
        chroma_client = chromadb.PersistentClient(path="/app/chroma_data")
        image_collection = chroma_client.get_or_create_collection(name="archaeo_images")
        return True
    except Exception as e:
        print(f"CLIP initialization error: {e}")
        return False

# Startup event
@app.on_event("startup")
async def startup_event():
    if CLIP_AVAILABLE:
        success = initialize_clip()
        if success:
            print("CLIP initialized successfully")
        else:
            print("CLIP initialization failed")
    else:
        print("CLIP not available - image search disabled")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "clip_available": CLIP_AVAILABLE,
        "clip_initialized": clip_model is not None
    }

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "ArchaeoFinder API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/api/search",
            "image_search": "/api/image-search",
            "index_image": "/api/index-image",
            "health": "/health"
        }
    }

# Europeana search endpoint
@app.post("/api/search")
async def search_europeana(search_query: SearchQuery):
    if not EUROPEANA_API_KEY:
        raise HTTPException(status_code=500, detail="Europeana API key not configured")
    
    params = {
        "wskey": EUROPEANA_API_KEY,
        "query": search_query.query,
        "rows": search_query.rows,
        "start": search_query.start,
        "profile": "rich",
        "media": "true",
        "thumbnail": "true"
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(EUROPEANA_SEARCH_URL, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Europeana API error: {str(e)}")

# Image search endpoint
@app.post("/api/image-search")
async def image_search(file: UploadFile = File(...), top_k: int = 10):
    if not CLIP_AVAILABLE or clip_model is None:
        raise HTTPException(status_code=503, detail="CLIP not available")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # Generate embedding
        with torch.no_grad():
            image_tensor = clip_preprocess(image).unsqueeze(0)
            image_features = clip_model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy().flatten().tolist()
        
        # Search in ChromaDB
        results = image_collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, image_collection.count())
        )
        
        return {
            "results": results,
            "count": len(results["ids"][0]) if results["ids"] else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image search error: {str(e)}")

# Index image endpoint
@app.post("/api/index-image")
async def index_image(file: UploadFile = File(...), metadata: Optional[str] = None):
    if not CLIP_AVAILABLE or clip_model is None:
        raise HTTPException(status_code=503, detail="CLIP not available")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # Generate embedding
        with torch.no_grad():
            image_tensor = clip_preprocess(image).unsqueeze(0)
            image_features = clip_model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy().flatten().tolist()
        
        # Generate unique ID
        import hashlib
        image_id = hashlib.md5(image_data).hexdigest()
        
        # Store in ChromaDB
        image_collection.add(
            embeddings=[embedding],
            ids=[image_id],
            metadatas=[{"filename": file.filename, "metadata": metadata or ""}]
        )
        
        return {
            "success": True,
            "image_id": image_id,
            "message": "Image indexed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image indexing error: {str(e)}")

# Text search endpoint
@app.post("/api/text-search")
async def text_search(query: str, top_k: int = 10):
    if not CLIP_AVAILABLE or clip_model is None:
        raise HTTPException(status_code=503, detail="CLIP not available")
    
    try:
        # Generate text embedding
        with torch.no_grad():
            text_tokens = clip_tokenizer([query])
            text_features = clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            embedding = text_features.cpu().numpy().flatten().tolist()
        
        # Search in ChromaDB
        results = image_collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, image_collection.count())
        )
        
        return {
            "results": results,
            "count": len(results["ids"][0]) if results["ids"] else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text search error: {str(e)}")

# Collection stats endpoint
@app.get("/api/collection-stats")
async def collection_stats():
    if not CLIP_AVAILABLE or image_collection is None:
        raise HTTPException(status_code=503, detail="CLIP not available")
    
    try:
        count = image_collection.count()
        return {
            "total_images": count,
            "collection_name": image_collection.name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")
