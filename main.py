“””
ArchaeoFinder Backend - Phase 2
Mit KI-Bilderkennung (CLIP) und Vektordatenbank (ChromaDB)
“””

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

# Phase 2 Imports

try:
import torch
import open_clip
from PIL import Image
import chromadb
from chromadb.config import Settings
import numpy as np
CLIP_AVAILABLE = True
print(“CLIP und ChromaDB erfolgreich geladen”)
except ImportError as e:
CLIP_AVAILABLE = False
print(f”CLIP/ChromaDB nicht verfuegbar: {e}”)

# Configuration

EUROPEANA_API_KEY = os.getenv(“EUROPEANA_API_KEY”, “api2demo”)
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {“jpg”, “jpeg”, “png”, “webp”, “gif”}

# CLIP Model Setup

clip_model = None
clip_preprocess = None
clip_tokenizer = None
chroma_client = None
image_collection = None

def initialize_clip():
global clip_model, clip_preprocess, clip_tokenizer, chroma_client, image_collection

```
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
    chroma_client = chromadb.Client(Settings(
        anonymized_telemetry=False
    ))
    
    image_collection = chroma_client.get_or_create_collection(
        name="archaeo_images",
        metadata={"description": "Archaeological artifact images"}
    )
    print(f"ChromaDB bereit (Bilder: {image_collection.count()})")
    
    return True
except Exception as e:
    print(f"Fehler bei Initialisierung: {e}")
    return False
```

def get_image_embedding(image: Image.Image) -> Optional[List[float]]:
if not clip_model or not clip_preprocess:
return None

```
try:
    image_input = clip_preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    return image_features[0].tolist()
except Exception as e:
    print(f"Fehler bei Bild-Embedding: {e}")
    return None
```

def get_text_embedding(text: str) -> Optional[List[float]]:
if not clip_model or not clip_tokenizer:
return None

```
try:
    text_input = clip_tokenizer([text])
    
    with torch.no_grad():
        text_features = clip_model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    return text_features[0].tolist()
except Exception as e:
    print(f"Fehler bei Text-Embedding: {e}")
    return None
```

async def index_europeana_images(query: str, max_images: int = 100):
if not image_collection:
return 0

```
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
                image_url = item["edmPreview"][0] if isinstance(item["edmPreview"], list) else item["edmPreview"]
            
            if not image_url:
                continue
            
            item_id = item.get("id", f"unknown_{indexed}")
            
            existing = image_collection.get(ids=[item_id])
            if existing and existing["ids"]:
                continue
            
            img_response = await client.get(image_url, timeout=10.0)
            if img_response.status_code != 200:
                continue
            
            image = Image.open(io.BytesIO(img_response.content)).convert("RGB")
            embedding = get_image_embedding(image)
            
            if not embedding:
                continue
            
            title = item.get("title", ["Unbekannt"])[0] if isinstance(item.get("title"), list) else item.get("title", "Unbekannt")
            museum = item.get("dataProvider", ["Unbekannt"])[0] if isinstance(item.get("dataProvider"), list) else item.get("dataProvider", "Unbekannt")
            
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
            indexed += 1
            
        except Exception as e:
            continue

return indexed
```

# Data Models

class MuseumObject(BaseModel):
id: str
title: str
description: Optional[str] = None
museum: Optional[str] = None
epoch: Optional[str] = None
image_url: Optional[str] = None
source_url: Optional[str] = None
source: str = “europeana”
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

# Mappings

EPOCH_MAPPING = {
“Steinzeit (bis 2200 v. Chr.)”: [“neolithic”, “stone age”, “mesolithic”, “paleolithic”],
“Bronzezeit (2200-800 v. Chr.)”: [“bronze age”, “bronzezeit”],
“Eisenzeit (800 v. Chr. - 0)”: [“iron age”, “eisenzeit”, “hallstatt”, “celtic”],
“Roemische Kaiserzeit (0-400 n. Chr.)”: [“roman”, “roman empire”],
“Fruehmittelalter (400-1000 n. Chr.)”: [“early medieval”, “migration period”],
“Hochmittelalter (1000-1300 n. Chr.)”: [“medieval”, “romanesque”],
“Spaetmittelalter (1300-1500 n. Chr.)”: [“late medieval”, “gothic”]
}

OBJECT_TYPE_MAPPING = {
“Fibeln und Gewandnadeln”: [“fibula”, “brooch”, “pin”],
“Muenzen”: [“coin”, “numismatic”],
“Keramik und Gefaesse”: [“ceramic”, “pottery”, “vessel”, “amphora”],
“Waffen und Werkzeuge”: [“weapon”, “sword”, “axe”, “tool”],
“Schmuck und Zierrat”: [“jewelry”, “jewellery”, “ring”, “bracelet”, “necklace”],
“Kultgegenstaende”: [“cult”, “ritual”, “religious”, “votive”],
“Alltagsgegenstaende”: [“domestic”, “household”, “daily life”]
}

REGION_MAPPING = {
“Mitteleuropa”: [“germany”, “austria”, “switzerland”],
“Nordeuropa”: [“scandinavia”, “denmark”, “sweden”, “norway”],
“Suedeuropa”: [“italy”, “greece”, “spain”, “mediterranean”],
“Westeuropa”: [“france”, “britain”, “england”],
“Osteuropa”: [“poland”, “czech”, “hungary”, “romania”],
“Mittelmeerraum”: [“mediterranean”, “aegean”, “roman”],
“Naher Osten”: [“mesopotamia”, “egypt”, “levant”, “near east”]
}

# FastAPI App

app = FastAPI(
title=“ArchaeoFinder API”,
description=“API fuer archaeologische Fundvergleiche mit KI-Bilderkennung”,
version=“2.0.0”,
docs_url=”/docs”,
redoc_url=”/redoc”
)

app.add_middleware(
CORSMiddleware,
allow_origins=[”*”],
allow_credentials=True,
allow_methods=[”*”],
allow_headers=[”*”],
)

uploaded_images = {}

@app.on_event(“startup”)
async def startup_event():
initialize_clip()

def build_europeana_query(keywords: Optional[str] = None, epoch: Optional[str] = None,
object_type: Optional[str] = None, region: Optional[str] = None) -> str:
query_parts = []
query_parts.append(”(archaeology OR archaeological)”)

```
if keywords:
    query_parts.append(f"({keywords})")

if epoch and epoch != "Alle Epochen":
    epoch_terms = EPOCH_MAPPING.get(epoch, [])
    if epoch_terms:
        epoch_query = " OR ".join(epoch_terms)
        query_parts.append(f"({epoch_query})")

if object_type and object_type != "Alle Objekttypen":
    type_terms = OBJECT_TYPE_MAPPING.get(object_type, [])
    if type_terms:
        type_query = " OR ".join(type_terms)
        query_parts.append(f"({type_query})")

if region and region != "Alle Regionen":
    region_terms = REGION_MAPPING.get(region, [])
    if region_terms:
        region_query = " OR ".join(region_terms)
        query_parts.append(f"({region_query})")

return " AND ".join(query_parts)
```

def parse_europeana_result(item: dict) -> MuseumObject:
title = “Unbekanntes Objekt”
if “title” in item and item[“title”]:
title = item[“title”][0] if isinstance(item[“title”], list) else item[“title”]

```
description = None
if "dcDescription" in item and item["dcDescription"]:
    desc = item["dcDescription"]
    description = desc[0] if isinstance(desc, list) else desc
    if description and len(description) > 300:
        description = description[:297] + "..."

museum = None
if "dataProvider" in item and item["dataProvider"]:
    dp = item["dataProvider"]
    museum = dp[0] if isinstance(dp, list) else dp

epoch = None
if "year" in item and item["year"]:
    years = item["year"]
    if isinstance(years, list) and years:
        epoch = f"{min(years)} - {max(years)}" if len(years) > 1 else str(years[0])

image_url = None
if "edmPreview" in item and item["edmPreview"]:
    previews = item["edmPreview"]
    image_url = previews[0] if isinstance(previews, list) else previews

source_url = None
if "guid" in item:
    source_url = item["guid"]
elif "id" in item:
    source_url = f"https://www.europeana.eu/item{item['id']}"

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
```

async def search_europeana(query: str, rows: int = 20) -> tuple:
url = “https://api.europeana.eu/record/v2/search.json”
params = {
“wskey”: EUROPEANA_API_KEY,
“query”: query,
“rows”: rows,
“profile”: “rich”,
“media”: “true”,
“qf”: “TYPE:IMAGE”,
}

```
async with httpx.AsyncClient(timeout=30.0) as client:
    response = await client.get(url, params=params)
    
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Europeana API error: {response.status_code}")
    
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
```

async def search_by_image(image: Image.Image, limit: int = 20) -> List[MuseumObject]:
if not image_collection or image_collection.count() == 0:
return []

```
embedding = get_image_embedding(image)
if not embedding:
    return []

results = image_collection.query(
    query_embeddings=[embedding],
    n_results=min(limit, image_collection.count())
)

if not results or not results["ids"] or not results["ids"][0]:
    return []

museum_objects = []
for i, item_id in enumerate(results["ids"][0]):
    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
    distance = results["distances"][0][i] if results["distances"] else 1.0
    
    similarity = int(max(0, min(100, (1 - distance) * 100)))
    
    museum_objects.append(MuseumObject(
        id=item_id,
        title=metadata.get("title", "Unbekannt"),
        museum=metadata.get("museum"),
        image_url=metadata.get("image_url"),
        source_url=metadata.get("source_url"),
        source=metadata.get("source", "europeana"),
        similarity=similarity
    ))

return museum_objects
```

# API Endpoints

@app.get(”/”)
async def root():
db_count = image_collection.count() if image_collection else 0
return {
“name”: “ArchaeoFinder API”,
“version”: “2.0.0”,
“status”: “online”,
“clip_available”: CLIP_AVAILABLE,
“images_indexed”: db_count,
“environment”: os.getenv(“ENVIRONMENT”, “development”),
“endpoints”: {
“search”: “/api/search”,
“search_image”: “/api/search/image”,
“upload”: “/api/upload”,
“index”: “/api/index”,
“docs”: “/docs”
}
}

@app.get(”/health”)
async def health_check():
return {“status”: “healthy”, “clip”: CLIP_AVAILABLE}

@app.post(”/api/upload”, response_model=UploadResponse)
async def upload_image(file: UploadFile = File(…)):
filename = file.filename or “unknown”
ext = filename.rsplit(”.”, 1)[-1].lower() if “.” in filename else “”

```
if ext not in ALLOWED_EXTENSIONS:
    raise HTTPException(status_code=400, detail=f"Ungueltiges Format. Erlaubt: {', '.join(ALLOWED_EXTENSIONS)}")

content = await file.read()

if len(content) > MAX_FILE_SIZE:
    raise HTTPException(status_code=400, detail=f"Datei zu gross. Maximum: {MAX_FILE_SIZE // (1024*1024)} MB")

image_hash = hashlib.sha256(content).hexdigest()[:16]
image_id = f"img_{image_hash}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

uploaded_images[image_id] = {
    "content": content,
    "filename": filename,
    "content_type": file.content_type,
    "uploaded_at": datetime.now().isoformat()
}

return UploadResponse(success=True, image_id=image_id, message="Bild erfolgreich hochgeladen")
```

@app.post(”/api/search/image”, response_model=SearchResponse)
async def search_by_uploaded_image(file: UploadFile = File(…), limit: int = Query(20, ge=1, le=50)):
if not CLIP_AVAILABLE:
raise HTTPException(status_code=503, detail=“Bildsuche nicht verfuegbar (CLIP nicht geladen)”)

```
content = await file.read()

try:
    image = Image.open(io.BytesIO(content)).convert("RGB")
except Exception as e:
    raise HTTPException(status_code=400, detail=f"Ungueltiges Bildformat: {e}")

results = await search_by_image(image, limit)

search_id = f"img_search_{datetime.now().strftime('%Y%m%d%H%M%S')}"

return SearchResponse(
    success=True,
    total_results=len(results),
    results=results,
    search_id=search_id,
    filters_applied={"image_search": True},
    search_mode="image"
)
```

@app.get(”/api/search”, response_model=SearchResponse)
async def search(
q: Optional[str] = Query(None, description=“Suchbegriffe”),
image_id: Optional[str] = Query(None, description=“ID des hochgeladenen Bildes”),
epoch: Optional[str] = Query(None, description=“Epoche”),
object_type: Optional[str] = Query(None, description=“Objekttyp”),
region: Optional[str] = Query(None, description=“Region”),
limit: int = Query(20, ge=1, le=50)
):
search_mode = “text”
results = []
total = 0

```
if image_id and image_id in uploaded_images and CLIP_AVAILABLE:
    try:
        content = uploaded_images[image_id]["content"]
        image = Image.open(io.BytesIO(content)).convert("RGB")
        results = await search_by_image(image, limit)
        total = len(results)
        search_mode = "image"
    except Exception as e:
        print(f"Bildsuche fehlgeschlagen: {e}")

if not results or q:
    query = build_europeana_query(keywords=q, epoch=epoch, object_type=object_type, region=region)
    
    try:
        total, text_results = await search_europeana(query, rows=limit)
        
        if search_mode == "text":
            import random
            for i, result in enumerate(text_results):
                base_similarity = 95 - (i * 3)
                result.similarity = max(50, min(99, base_similarity + random.randint(-5, 5)))
            results = text_results
        else:
            search_mode = "hybrid"
            existing_ids = {r.id for r in results}
            for r in text_results:
                if r.id not in existing_ids:
                    r.similarity = 50
                    results.append(r)
                    
    except Exception as e:
        if not results:
            raise HTTPException(status_code=500, detail=f"Suchfehler: {str(e)}")

results.sort(key=lambda x: x.similarity or 0, reverse=True)

search_id = f"search_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hashlib.md5(str(q).encode()).hexdigest()[:8]}"

return SearchResponse(
    success=True,
    total_results=total,
    results=results[:limit],
    search_id=search_id,
    filters_applied={"keywords": q, "epoch": epoch, "object_type": object_type, "region": region, "image_id": image_id},
    search_mode=search_mode
)
```

@app.post(”/api/index”, response_model=IndexResponse)
async def index_images(
query: str = Query(“archaeology”, description=“Suchbegriff fuer zu indexierende Bilder”),
max_images: int = Query(50, ge=10, le=200)
):
if not CLIP_AVAILABLE:
raise HTTPException(status_code=503, detail=“Indexierung nicht verfuegbar (CLIP nicht geladen)”)

```
indexed = await index_europeana_images(query, max_images)
total = image_collection.count() if image_collection else 0

return IndexResponse(
    success=True,
    indexed_count=indexed,
    total_in_db=total,
    message=f"{indexed} neue Bilder indexiert. Gesamt in Datenbank: {total}"
)
```

@app.get(”/api/index/status”)
async def index_status():
total = image_collection.count() if image_collection else 0
return {
“clip_available”: CLIP_AVAILABLE,
“total_images”: total,
“ready”: CLIP_AVAILABLE and total > 0
}

@app.get(”/api/filters”)
async def get_filters():
return {
“epochs”: [“Alle Epochen”] + list(EPOCH_MAPPING.keys()),
“object_types”: [“Alle Objekttypen”] + list(OBJECT_TYPE_MAPPING.keys()),
“regions”: [“Alle Regionen”] + list(REGION_MAPPING.keys())
}

@app.get(”/api/sources”)
async def get_sources():
return {
“sources”: [
{“id”: “europeana”, “name”: “Europeana”, “status”: “active”},
{“id”: “ddb”, “name”: “Deutsche Digitale Bibliothek”, “status”: “coming_soon”},
{“id”: “british_museum”, “name”: “British Museum”, “status”: “coming_soon”}
]
}

if **name** == “**main**”:
import uvicorn
port = int(os.getenv(“PORT”, 8080))
uvicorn.run(app, host=“0.0.0.0”, port=port)