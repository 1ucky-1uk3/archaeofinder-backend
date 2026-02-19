from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import os
from io import BytesIO
from PIL import Image
import hashlib
import json
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# CLIP imports - lazy loading
CLIP_AVAILABLE = False
clip_model = None
clip_preprocess = None
clip_tokenizer = None
chroma_client = None
image_collection = None
torch = None
open_clip = None
chromadb = None

try:
    import torch as torch_module
    import open_clip as open_clip_module
    import chromadb as chromadb_module
    torch = torch_module
    open_clip = open_clip_module
    chromadb = chromadb_module
    CLIP_AVAILABLE = True
    print("✓ CLIP libraries imported successfully", flush=True)
except ImportError as e:
    print(f"✗ CLIP Import Error: {e}", flush=True)
    CLIP_AVAILABLE = False

app = FastAPI(title="ArchaeoFinder API", version="1.0.0")

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

# Models
class SearchQuery(BaseModel):
    query: str
    rows: int = 12
    start: int = 1

class TextSearchQuery(BaseModel):
    query: str
    top_k: int = 10

# Lazy load CLIP model
def ensure_clip_loaded():
    global clip_model, clip_preprocess, clip_tokenizer, chroma_client, image_collection
    
    if not CLIP_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="CLIP libraries not available"
        )
    
    # Load CLIP model on first request
    if clip_model is None:
        try:
            print("Loading CLIP model (first request)...", flush=True)
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32-quickgelu",
                pretrained="laion400m_e32"
            )
            tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
            model.eval()
            clip_model = model
            clip_preprocess = preprocess
            clip_tokenizer = tokenizer
            print("✓ CLIP model loaded successfully", flush=True)
        except Exception as e:
            print(f"✗ CLIP loading error: {e}", flush=True)
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load CLIP model: {str(e)}"
            )
    
    # Initialize ChromaDB on first request
    if chroma_client is None:
        try:
            print("Initializing ChromaDB...", flush=True)
            chroma_client = chromadb.PersistentClient(path="/app/chroma_data")
            image_collection = chroma_client.get_or_create_collection(
                name="archaeo_images",
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✓ ChromaDB initialized. Collection has {image_collection.count()} images", flush=True)
        except Exception as e:
            print(f"✗ ChromaDB error: {e}", flush=True)
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize ChromaDB: {str(e)}"
            )

# Startup event
@app.on_event("startup")
async def startup_event():
    print("=" * 60, flush=True)
    print("ArchaeoFinder API Started", flush=True)
    print("CLIP will load on first use (lazy loading)", flush=True)
    print("=" * 60, flush=True)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check that always returns 200 if server is running"""
    collection_count = 0
    if image_collection is not None:
        try:
            collection_count = image_collection.count()
        except:
            pass
    
    return {
        "status": "healthy",
        "clip_available": CLIP_AVAILABLE,
        "clip_loaded": clip_model is not None,
        "indexed_images": collection_count
    }

# Liveness probe
@app.get("/healthz")
async def liveness():
    """Kubernetes-style liveness probe"""
    return {"status": "alive"}

# Readiness check
@app.get("/ready")
async def readiness_check():
    """Check if app is ready to serve requests"""
    return {
        "status": "ready",
        "clip_available": CLIP_AVAILABLE,
        "clip_loaded": clip_model is not None
    }

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "ArchaeoFinder API",
        "version": "1.0.0",
        "status": "running",
        "clip_available": CLIP_AVAILABLE,
        "clip_loaded": clip_model is not None,
        "endpoints": {
            "health": "/health",
            "healthz": "/healthz",
            "ready": "/ready",
            "search_europeana": "/api/search",
            "image_search": "/api/image-search",
            "text_search": "/api/text-search",
            "index_image": "/api/index-image",
            "collection_stats": "/api/collection-stats",
            "list_images": "/api/list-images",
            "delete_image": "/api/delete-image/{image_id}",
            "clear_collection": "/api/clear-collection"
        }
    }

# Europeana search endpoint
@app.post("/api/search")
async def search_europeana(search_query: SearchQuery):
    if not EUROPEANA_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Europeana API key not configured. Set EUROPEANA_API_KEY environment variable."
        )
    
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
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Europeana API timeout")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Europeana API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Image search endpoint
@app.post("/api/image-search")
async def image_search(file: UploadFile = File(...), top_k: int = Form(10)):
    # Ensure CLIP is loaded
    ensure_clip_loaded()
    
    if image_collection.count() == 0:
        raise HTTPException(
            status_code=404,
            detail="No images indexed yet. Please index images first using /api/index-image"
        )
    
    try:
        # Read and validate image
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        try:
            image = Image.open(BytesIO(image_data)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Generate embedding
        with torch.no_grad():
            image_tensor = clip_preprocess(image).unsqueeze(0)
            image_features = clip_model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy().flatten().tolist()
        
        # Search in ChromaDB
        n_results = min(top_k, image_collection.count())
        results = image_collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
        
        return {
            "success": True,
            "results": results,
            "count": len(results["ids"][0]) if results["ids"] else 0,
            "total_indexed": image_collection.count()
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image search error: {str(e)}")

# Index image endpoint
@app.post("/api/index-image")
async def index_image(
    file: UploadFile = File(...), 
    metadata: Optional[str] = Form(None)
):
    # Ensure CLIP is loaded
    ensure_clip_loaded()
    
    try:
        # Read and validate image
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        try:
            image = Image.open(BytesIO(image_data)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Generate embedding
        with torch.no_grad():
            image_tensor = clip_preprocess(image).unsqueeze(0)
            image_features = clip_model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy().flatten().tolist()
        
        # Generate unique ID
        image_id = hashlib.md5(image_data).hexdigest()
        
        # Parse metadata if provided
        meta_dict = {}
        if metadata:
            try:
                meta_dict = json.loads(metadata)
            except:
                meta_dict = {"raw_metadata": metadata}
        
        meta_dict["filename"] = file.filename
        meta_dict["content_type"] = file.content_type or "image/unknown"
        
        # Check if image already exists
        try:
            existing = image_collection.get(ids=[image_id])
            if existing and existing["ids"]:
                return {
                    "success": True,
                    "image_id": image_id,
                    "message": "Image already indexed",
                    "already_exists": True,
                    "total_indexed": image_collection.count()
                }
        except:
            pass
        
        # Store in ChromaDB
        image_collection.add(
            embeddings=[embedding],
            ids=[image_id],
            metadatas=[meta_dict]
        )
        
        return {
            "success": True,
            "image_id": image_id,
            "message": "Image indexed successfully",
            "already_exists": False,
            "total_indexed": image_collection.count()
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image indexing error: {str(e)}")

# Text search endpoint
@app.post("/api/text-search")
async def text_search(search_query: TextSearchQuery):
    # Ensure CLIP is loaded
    ensure_clip_loaded()
    
    if image_collection.count() == 0:
        raise HTTPException(
            status_code=404,
            detail="No images indexed yet. Please index images first using /api/index-image"
        )
    
    try:
        # Generate text embedding
        with torch.no_grad():
            text_tokens = clip_tokenizer([search_query.query])
            text_features = clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            embedding = text_features.cpu().numpy().flatten().tolist()
        
        # Search in ChromaDB
        n_results = min(search_query.top_k, image_collection.count())
        results = image_collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
        
        return {
            "success": True,
            "query": search_query.query,
            "results": results,
            "count": len(results["ids"][0]) if results["ids"] else 0,
            "total_indexed": image_collection.count()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Text search error: {str(e)}")

# Collection stats endpoint
@app.get("/api/collection-stats")
async def collection_stats():
    ensure_clip_loaded()
    
    try:
        count = image_collection.count()
        return {
            "success": True,
            "total_images": count,
            "collection_name": image_collection.name,
            "collection_metadata": image_collection.metadata
        }
except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

# Delete image endpoint
@app.delete("/api/delete-image/{image_id}")
async def delete_image(image_id: str):
    ensure_clip_loaded()
    
    try:
        image_collection.delete(ids=[image_id])
        return {
            "success": True,
            "message": f"Image {image_id} deleted successfully",
            "total_indexed": image_collection.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")

# Clear collection endpoint
@app.delete("/api/clear-collection")
async def clear_collection():
    ensure_clip_loaded()
    
    try:
        # Get all IDs and delete them
        all_items = image_collection.get()
        if all_items and all_items["ids"]:
            image_collection.delete(ids=all_items["ids"])
            deleted_count = len(all_items["ids"])
        else:
            deleted_count = 0
        
        return {
            "success": True,
            "message": "Collection cleared successfully",
            "deleted_count": deleted_count,
            "total_indexed": image_collection.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear collection error: {str(e)}")

# Get all indexed images endpoint
@app.get("/api/list-images")
async def list_images(limit: int = 100, offset: int = 0):
    ensure_clip_loaded()
    
    try:
        all_items = image_collection.get(
            limit=limit,
            offset=offset,
            include=["metadatas"]
        )
        
        return {
            "success": True,
            "images": all_items,
            "count": len(all_items["ids"]) if all_items["ids"] else 0,
            "total_indexed": image_collection.count(),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List images error: {str(e)}")
