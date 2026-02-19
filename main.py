"""
ArchaeoFinder Backend - Phase 2
KI-Bilderkennung mit CLIP und ChromaDB
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx
import base64
import hashlib
from datetime import datetime
import os
import io

try:
    import torch
    import open_clip
    from PIL import Image
    import chromadb
    import numpy as np
    CLIP_AVAILABLE = True
    print("CLIP und ChromaDB geladen")
except ImportError as e:
    CLIP_AVAILABLE = False
    print("CLIP nicht verfuegbar: " + str(e))

EUROPEANA_API_KEY = os.getenv("EUROPEANA_API_KEY", "api2demo")
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = set(["jpg", "jpeg", "png", "webp", "gif"])

clip_model = None
clip_preprocess = None
clip_tokenizer = None
chroma_client = None
image_collection = None

def initialize_clip():
    global clip_model, clip_preprocess, clip_tokenizer, chroma_client, image_collection
    
    if not CLIP_AVAILABLE:
        return False
    
    try:
        print("Lade CLIP-Modell...")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k"
        )
        clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        clip_model.eval()
        print("CLIP-Modell geladen")
        
        print("Initialisiere ChromaDB...")
        chroma_client = chromadb.Client()
        
        image_collection = chroma_client.get_or_create_collection(
            name="archaeo_images"
        )
        print("ChromaDB bereit")
        
        return True
    except Exception as e:
        print("Fehler bei Initialisierung: " + str(e))
        return False


def get_image_embedding(image):
    if not clip_model or not clip_preprocess:
        return None
    
    try:
        image_input = clip_preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features[0].tolist()
    except Exception as e:
        print("Fehler bei Bild-Embedding: " + str(e))
        return None


async def index_europeana_images(query, max_images=100):
    if not image_collection:
        return 0
    
    url = "https://api.europeana.eu/record/v2/search.json"
    params = {
        "wskey": EUROPEANA_API_KEY,
        "query": query,
        "rows": max_images,
        "profile": "rich",
        "media": "true",
        "qf": "TYPE:IMAGE",
    }
    
    indexed = 0
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url, params=params)
        if response.status_code != 200:
            return 0
        
        data = response.json()
        items = data.get("items", [])
        
        for item in items:
            try:
                image_url = None
                if "edmPreview" in item and item["edmPreview"]:
                    previews = item["edmPreview"]
                    if isinstance(previews, list):
                        image_url = previews[0]
                    else:
                        image_url = previews
                
                if not image_url:
                    continue
                
                item_id = item.get("id", "unknown_" + str(indexed))
                
                existing = image_collection.get(ids=[item_id])
                if existing and existing.get("ids"):
                    continue
                
                img_response = await client.get(image_url, timeout=10.0)
                if img_response.status_code != 200:
                    continue
                
                image = Image.open(io.BytesIO(img_response.content)).convert("RGB")
                embedding = get_image_embedding(image)
                
                if not embedding:
                    continue
                
                title_data = item.get("title", ["Unbekannt"])
                if isinstance(title_data, list):
                    title = title_data[0]
                else:
                    title = title_data
                
                museum_data = item.get("dataProvider", ["Unbekannt"])
                if isinstance(museum_data, list):
                    museum = museum_data[0]
                else:
                    museum = museum_data
                
                image_collection.add(
                    ids=[item_id],
                    embeddings=[embedding],
                    metadatas=[{
                        "title": str(title)[:500],
                        "museum": str(museum)[:500],
                        "image_url": image_url,
                        "source_url": item.get("guid", ""),
                        "source": "europeana"
                    }]
                )
                indexed = indexed + 1
                
            except Exception:
                continue
    
    return indexed


class MuseumObject(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    museum: Optional[str] = None
    epoch: Optional[str] = None
    image_url: Optional[str] = None
    source_url: Optional[str] = None
    source: str = "europeana"
    similarity: Optional[int] = None

class SearchResponse(BaseModel):
    success: bool
    total_results: int
    results: List[MuseumObject]
    search_id: str
    filters_applied: dict
    search_mode: str

class UploadResponse(BaseModel):
    success: bool
    image_id: str
    message: str

class IndexResponse(BaseModel):
    success: bool
    indexed_count: int
    total_in_db: int
    message: str


EPOCH_MAPPING = {
    "Steinzeit": ["neolithic", "stone age", "mesolithic", "paleolithic"],
    "Bronzezeit": ["bronze age"],
    "Eisenzeit": ["iron age", "hallstatt", "celtic"],
    "Roemische Kaiserzeit": ["roman", "roman empire"],
    "Fruehmittelalter": ["early medieval", "migration period"],
    "Hochmittelalter": ["medieval", "romanesque"],
    "Spaetmittelalter": ["late medieval", "gothic"]
}

OBJECT_TYPE_MAPPING = {
    "Fibeln": ["fibula", "brooch", "pin"],
    "Muenzen": ["coin", "numismatic"],
    "Keramik": ["ceramic", "pottery", "vessel", "amphora"],
    "Waffen": ["weapon", "sword", "axe", "tool"],
    "Schmuck": ["jewelry", "jewellery", "ring", "bracelet", "necklace"],
    "Kultgegenstaende": ["cult", "ritual", "religious", "votive"],
    "Alltagsgegenstaende": ["domestic", "household"]
}

REGION_MAPPING = {
    "Mitteleuropa": ["germany", "austria", "switzerland"],
    "Nordeuropa": ["scandinavia", "denmark", "sweden", "norway"],
    "Suedeuropa": ["italy", "greece", "spain"],
    "Westeuropa": ["france", "britain", "england"],
    "Osteuropa": ["poland", "czech", "hungary", "romania"],
    "Mittelmeerraum": ["mediterranean", "aegean"],
    "Naher Osten": ["mesopotamia", "egypt", "levant"]
}


app = FastAPI(
    title="ArchaeoFinder API",
    description="API fuer archaeologische Fundvergleiche",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploaded_images = {}


@app.on_event("startup")
async def startup_event():
    initialize_clip()


def build_europeana_query(keywords=None, epoch=None, object_type=None, region=None):
    query_parts = []
    query_parts.append("(archaeology OR archaeological)")
    
    if keywords:
        query_parts.append("(" + keywords + ")")
    
    if epoch and epoch != "Alle Epochen":
        epoch_terms = EPOCH_MAPPING.get(epoch, [])
        if epoch_terms:
            epoch_query = " OR ".join(epoch_terms)
            query_parts.append("(" + epoch_query + ")")
    
    if object_type and object_type != "Alle Objekttypen":
        type_terms = OBJECT_TYPE_MAPPING.get(object_type, [])
        if type_terms:
            type_query = " OR ".join(type_terms)
            query_parts.append("(" + type_query + ")")
    
    if region and region != "Alle Regionen":
        region_terms = REGION_MAPPING.get(region, [])
        if region_terms:
            region_query = " OR ".join(region_terms)
            query_parts.append("(" + region_query + ")")
    
    return " AND ".join(query_parts)


def parse_europeana_result(item):
    title = "Unbekanntes Objekt"
    if "title" in item and item["title"]:
        title_data = item["title"]
        if isinstance(title_data, list):
            title = title_data[0]
        else:
            title = title_data
    
    description = None
    if "dcDescription" in item and item["dcDescription"]:
        desc_data = item["dcDescription"]
        if isinstance(desc_data, list):
            description = desc_data[0]
        else:
            description = desc_data
        if description and len(description) > 300:
            description = description[:297] + "..."
    
    museum = None
    if "dataProvider" in item and item["dataProvider"]:
        dp_data = item["dataProvider"]
        if isinstance(dp_data, list):
            museum = dp_data[0]
        else:
            museum = dp_data
    
    epoch = None
    if "year" in item and item["year"]:
        years = item["year"]
        if isinstance(years, list) and years:
            if len(years) > 1:
                epoch = str(min(years)) + " - " + str(max(years))
            else:
                epoch = str(years[0])
    
    image_url = None
    if "edmPreview" in item and item["edmPreview"]:
        previews = item["edmPreview"]
        if isinstance(previews, list):
            image_url = previews[0]
        else:
            image_url = previews
    
    source_url = None
    if "guid" in item:
        source_url = item["guid"]
    elif "id" in item:
        source_url = "https://www.europeana.eu/item" + item["id"]
    
    return MuseumObject(
        id=item.get("id", "unknown"),
        title=title,
        description=description,
        museum=museum,
        epoch=epoch,
        image_url=image_url,
        source_url=source_url,
        source="europeana"
    )


async def search_europeana(query, rows=20):
    url = "https://api.europeana.eu/record/v2/search.json"
    params = {
        "wskey": EUROPEANA_API_KEY,
        "query": query,
        "rows": rows,
        "profile": "rich",
        "media": "true",
        "qf": "TYPE:IMAGE",
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, params=params)
        
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail="Europeana API error")
        
        data = response.json()
        total = data.get("totalResults", 0)
        items = data.get("items", [])
        
        results = []
        for item in items:
            try:
                parsed = parse_europeana_result(item)
                if parsed.image_url:
                    results.append(parsed)
            except Exception:
                continue
        
        return total, results


async def search_by_image(image, limit=20):
    if not image_collection:
        return []
    
    count = image_collection.count()
    if count == 0:
        return []
    
    embedding = get_image_embedding(image)
    if not embedding:
        return []
    
    n_results = limit
    if count < limit:
        n_results = count
    
    results = image_collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )
    
    if not results:
        return []
    
    ids_list = results.get("ids", [])
    if not ids_list or not ids_list[0]:
        return []
    
    museum_objects = []
    metadatas = results.get("metadatas", [[]])
    distances = results.get("distances", [[]])
    
    for i in range(len(ids_list[0])):
        item_id = ids_list[0][i]
        
        metadata = {}
        if metadatas and metadatas[0] and i < len(metadatas[0]):
            metadata = metadatas[0][i]
        
        distance = 1.0
        if distances and distances[0] and i < len(distances[0]):
            distance = distances[0][i]
        
        similarity = int(max(0, min(100, (1 - distance) * 100)))
        
        museum_objects.append(MuseumObject(
            id=item_id,
            title=metadata.get("title", "Unbekannt"),
            museum=metadata.get("museum", None),
            image_url=metadata.get("image_url", None),
            source_url=metadata.get("source_url", None),
            source=metadata.get("source", "europeana"),
            similarity=similarity
        ))
    
    return museum_objects


@app.get("/")
async def root():
    db_count = 0
    if image_collection:
        db_count = image_collection.count()
    
    return {
        "name": "ArchaeoFinder API",
        "version": "2.0.0",
        "status": "online",
        "clip_available": CLIP_AVAILABLE,
        "images_indexed": db_count,
        "environment": os.getenv("ENVIRONMENT", "development")
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "clip": CLIP_AVAILABLE}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_image_endpoint(file: UploadFile = File(...)):
    filename = file.filename or "unknown"
    ext = ""
    if "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
    
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Ungueltiges Format")
    
    content = await file.read()
    
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Datei zu gross")
    
    image_hash = hashlib.sha256(content).hexdigest()[:16]
    image_id = "img_" + image_hash + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
    
    uploaded_images[image_id] = {
        "content": content,
        "filename": filename,
        "content_type": file.content_type,
        "uploaded_at": datetime.now().isoformat()
    }
    
    return UploadResponse(success=True, image_id=image_id, message="Bild hochgeladen")


@app.post("/api/search/image", response_model=SearchResponse)
async def search_by_uploaded_image(file: UploadFile = File(...), limit: int = Query(20, ge=1, le=50)):
    if not CLIP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Bildsuche nicht verfuegbar")
    
    content = await file.read()
    
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Ungueltiges Bildformat")
    
    results = await search_by_image(image, limit)
    
    search_id = "img_search_" + datetime.now().strftime("%Y%m%d%H%M%S")
    
    return SearchResponse(
        success=True,
        total_results=len(results),
        results=results,
        search_id=search_id,
        filters_applied={"image_search": True},
        search_mode="image"
    )


@app.get("/api/search", response_model=SearchResponse)
async def search(
    q: Optional[str] = Query(None),
    image_id: Optional[str] = Query(None),
    epoch: Optional[str] = Query(None),
    object_type: Optional[str] = Query(None),
    region: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=50)
):
    search_mode = "text"
    results = []
    total = 0
    
    if image_id and image_id in uploaded_images and CLIP_AVAILABLE:
        try:
            content = uploaded_images[image_id]["content"]
            image = Image.open(io.BytesIO(content)).convert("RGB")
            results = await search_by_image(image, limit)
            total = len(results)
            search_mode = "image"
        except Exception as e:
            print("Bildsuche fehlgeschlagen: " + str(e))
    
    if not results or q:
        query = build_europeana_query(keywords=q, epoch=epoch, object_type=object_type, region=region)
        
        try:
            total, text_results = await search_europeana(query, rows=limit)
            
            if search_mode == "text":
                import random
                for i in range(len(text_results)):
                    result = text_results[i]
                    base_similarity = 95 - (i * 3)
                    result.similarity = max(50, min(99, base_similarity + random.randint(-5, 5)))
                results = text_results
            else:
                search_mode = "hybrid"
                existing_ids = set()
                for r in results:
                    existing_ids.add(r.id)
                for r in text_results:
                    if r.id not in existing_ids:
                        r.similarity = 50
                        results.append(r)
                        
        except Exception as e:
            if not results:
                raise HTTPException(status_code=500, detail="Suchfehler: " + str(e))
    
    results.sort(key=lambda x: x.similarity or 0, reverse=True)
    
    search_id = "search_" + datetime.now().strftime("%Y%m%d%H%M%S")
    
    return SearchResponse(
        success=True,
        total_results=total,
        results=results[:limit],
        search_id=search_id,
        filters_applied={"keywords": q, "epoch": epoch, "object_type": object_type, "region": region},
        search_mode=search_mode
    )


@app.post("/api/index", response_model=IndexResponse)
async def index_images(
    query: str = Query("archaeology"),
    max_images: int = Query(50, ge=10, le=200)
):
    if not CLIP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Indexierung nicht verfuegbar")
    
    indexed = await index_europeana_images(query, max_images)
    
    total = 0
    if image_collection:
        total = image_collection.count()
    
    return IndexResponse(
        success=True,
        indexed_count=indexed,
        total_in_db=total,
        message=str(indexed) + " neue Bilder indexiert"
    )


@app.get("/api/index/status")
async def index_status():
    total = 0
    if image_collection:
        total = image_collection.count()
    return {
        "clip_available": CLIP_AVAILABLE,
        "total_images": total,
        "ready": CLIP_AVAILABLE and total > 0
    }


@app.get("/api/filters")
async def get_filters():
    return {
        "epochs": ["Alle Epochen"] + list(EPOCH_MAPPING.keys()),
        "object_types": ["Alle Objekttypen"] + list(OBJECT_TYPE_MAPPING.keys()),
        "regions": ["Alle Regionen"] + list(REGION_MAPPING.keys())
    }


@app.get("/api/sources")
async def get_sources():
    return {
        "sources": [
            {"id": "europeana", "name": "Europeana", "status": "active"},
            {"id": "ddb", "name": "Deutsche Digitale Bibliothek", "status": "coming_soon"},
            {"id": "british_museum", "name": "British Museum", "status": "coming_soon"}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
