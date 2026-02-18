# “””
ArchaeoFinder Backend

FastAPI server for archaeological find comparison.
Connects to Europeana and other museum databases.

Optimiert für DigitalOcean App Platform.
“””

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import httpx
import base64
import hashlib
from datetime import datetime
import os

# ============================================================

# CONFIGURATION

# ============================================================

# Europeana API - Free tier, register at https://pro.europeana.eu/page/get-api

EUROPEANA_API_KEY = os.getenv(“EUROPEANA_API_KEY”, “api2demo”)
EUROPEANA_BASE_URL = “https://api.europeana.eu/record/v2/search.json”

# App configuration

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {“jpg”, “jpeg”, “png”, “webp”, “gif”}

# Frontend URL (für CORS) - Setze dies auf deine echte Domain

FRONTEND_URL = os.getenv(“FRONTEND_URL”, “*”)

# ============================================================

# DATA MODELS

# ============================================================

class SearchFilters(BaseModel):
epoch: Optional[str] = None
object_type: Optional[str] = None
region: Optional[str] = None

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

class UploadResponse(BaseModel):
success: bool
image_id: str
message: str

# ============================================================

# EPOCH & CATEGORY MAPPINGS

# ============================================================

EPOCH_MAPPING = {
“Steinzeit (bis 2200 v. Chr.)”: [“neolithic”, “stone age”, “mesolithic”, “paleolithic”],
“Bronzezeit (2200-800 v. Chr.)”: [“bronze age”, “bronzezeit”],
“Eisenzeit (800 v. Chr. - 0)”: [“iron age”, “eisenzeit”, “la tène”, “hallstatt”, “celtic”],
“Römische Kaiserzeit (0-400 n. Chr.)”: [“roman”, “römisch”, “roman empire”],
“Frühmittelalter (400-1000 n. Chr.)”: [“early medieval”, “frühmittelalter”, “migration period”],
“Hochmittelalter (1000-1300 n. Chr.)”: [“medieval”, “mittelalter”, “romanesque”],
“Spätmittelalter (1300-1500 n. Chr.)”: [“late medieval”, “gothic”, “spätmittelalter”]
}

OBJECT_TYPE_MAPPING = {
“Fibeln & Gewandnadeln”: [“fibula”, “brooch”, “fibel”, “gewandnadel”, “pin”],
“Münzen”: [“coin”, “münze”, “numismatic”],
“Keramik & Gefäße”: [“ceramic”, “pottery”, “keramik”, “vessel”, “gefäß”, “amphora”],
“Waffen & Werkzeuge”: [“weapon”, “sword”, “axe”, “tool”, “waffe”, “werkzeug”],
“Schmuck & Zierrat”: [“jewelry”, “jewellery”, “schmuck”, “ring”, “bracelet”, “necklace”],
“Kultgegenstände”: [“cult”, “ritual”, “religious”, “votive”],
“Alltagsgegenstände”: [“domestic”, “household”, “daily life”]
}

REGION_MAPPING = {
“Mitteleuropa”: [“germany”, “austria”, “switzerland”, “deutschland”, “österreich”],
“Nordeuropa”: [“scandinavia”, “denmark”, “sweden”, “norway”],
“Südeuropa”: [“italy”, “greece”, “spain”, “mediterranean”],
“Westeuropa”: [“france”, “britain”, “england”, “frankreich”],
“Osteuropa”: [“poland”, “czech”, “hungary”, “romania”],
“Mittelmeerraum”: [“mediterranean”, “aegean”, “roman”],
“Naher Osten”: [“mesopotamia”, “egypt”, “levant”, “near east”]
}

# ============================================================

# FASTAPI APP

# ============================================================

app = FastAPI(
title=“ArchaeoFinder API”,
description=“API für archäologische Fundvergleiche mit Museumsdatenbanken”,
version=“1.0.0”,
docs_url=”/docs”,
redoc_url=”/redoc”
)

# CORS Configuration

app.add_middleware(
CORSMiddleware,
allow_origins=[FRONTEND_URL] if FRONTEND_URL != “*” else [”*”],
allow_credentials=True,
allow_methods=[”*”],
allow_headers=[”*”],
)

# In-memory storage for uploaded images

uploaded_images = {}

# ============================================================

# HELPER FUNCTIONS

# ============================================================

def build_europeana_query(
keywords: Optional[str] = None,
filters: Optional[SearchFilters] = None
) -> str:
“”“Build Europeana search query from keywords and filters.”””

```
query_parts = []

# Base query for archaeological content
query_parts.append('(what:"archaeology" OR what:"archaeological" OR what:"archäologie")')

# Add user keywords
if keywords:
    query_parts.append(f'("{keywords}")')

# Add epoch filter
if filters and filters.epoch and filters.epoch != "Alle Epochen":
    epoch_terms = EPOCH_MAPPING.get(filters.epoch, [])
    if epoch_terms:
        epoch_query = " OR ".join([f'"{term}"' for term in epoch_terms])
        query_parts.append(f"({epoch_query})")

# Add object type filter
if filters and filters.object_type and filters.object_type != "Alle Objekttypen":
    type_terms = OBJECT_TYPE_MAPPING.get(filters.object_type, [])
    if type_terms:
        type_query = " OR ".join([f'what:"{term}"' for term in type_terms])
        query_parts.append(f"({type_query})")

# Add region filter
if filters and filters.region and filters.region != "Alle Regionen":
    region_terms = REGION_MAPPING.get(filters.region, [])
    if region_terms:
        region_query = " OR ".join([f'"{term}"' for term in region_terms])
        query_parts.append(f"({region_query})")

return " AND ".join(query_parts)
```

def parse_europeana_result(item: dict) -> MuseumObject:
“”“Parse a single Europeana API result into our MuseumObject format.”””

```
# Extract title
title = "Unbekanntes Objekt"
if "title" in item and item["title"]:
    title = item["title"][0] if isinstance(item["title"], list) else item["title"]

# Extract description
description = None
if "dcDescription" in item and item["dcDescription"]:
    desc = item["dcDescription"]
    description = desc[0] if isinstance(desc, list) else desc
    if description and len(description) > 300:
        description = description[:297] + "..."

# Extract museum/provider
museum = None
if "dataProvider" in item and item["dataProvider"]:
    dp = item["dataProvider"]
    museum = dp[0] if isinstance(dp, list) else dp

# Extract time period
epoch = None
if "dctermsCreated" in item and item["dctermsCreated"]:
    created = item["dctermsCreated"]
    epoch = created[0] if isinstance(created, list) else created
elif "year" in item and item["year"]:
    years = item["year"]
    if isinstance(years, list) and years:
        epoch = f"{min(years)} - {max(years)}" if len(years) > 1 else str(years[0])

# Extract image URL
image_url = None
if "edmPreview" in item and item["edmPreview"]:
    previews = item["edmPreview"]
    image_url = previews[0] if isinstance(previews, list) else previews
elif "edmIsShownBy" in item and item["edmIsShownBy"]:
    shown = item["edmIsShownBy"]
    image_url = shown[0] if isinstance(shown, list) else shown

# Build source URL
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

async def search_europeana(
query: str,
rows: int = 20
) -> tuple[int, List[MuseumObject]]:
“”“Search Europeana API and return parsed results.”””

```
params = {
    "wskey": EUROPEANA_API_KEY,
    "query": query,
    "rows": rows,
    "profile": "rich",
    "media": "true",
    "qf": "TYPE:IMAGE",
}

async with httpx.AsyncClient(timeout=30.0) as client:
    response = await client.get(EUROPEANA_BASE_URL, params=params)
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Europeana API error: {response.status_code}"
        )
    
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

# ============================================================

# API ENDPOINTS

# ============================================================

@app.get(”/”)
async def root():
“”“API health check.”””
return {
“name”: “ArchaeoFinder API”,
“version”: “1.0.0”,
“status”: “online”,
“environment”: os.getenv(“ENVIRONMENT”, “development”),
“endpoints”: {
“search”: “/api/search”,
“upload”: “/api/upload”,
“sources”: “/api/sources”,
“docs”: “/docs”
}
}

@app.get(”/health”)
async def health_check():
“”“Health check endpoint for DigitalOcean.”””
return {“status”: “healthy”}

@app.get(”/api/sources”)
async def get_sources():
“”“Get available museum data sources.”””
return {
“sources”: [
{
“id”: “europeana”,
“name”: “Europeana”,
“description”: “Europäische digitale Bibliothek mit über 50 Mio. Objekten”,
“status”: “active”,
“website”: “https://www.europeana.eu”
},
{
“id”: “ddb”,
“name”: “Deutsche Digitale Bibliothek”,
“description”: “Deutsche Museen und Sammlungen”,
“status”: “coming_soon”,
“website”: “https://www.deutsche-digitale-bibliothek.de”
},
{
“id”: “british_museum”,
“name”: “British Museum”,
“description”: “Eine der größten Antikensammlungen weltweit”,
“status”: “coming_soon”,
“website”: “https://www.britishmuseum.org”
}
]
}

@app.post(”/api/upload”, response_model=UploadResponse)
async def upload_image(file: UploadFile = File(…)):
“”“Upload an image for analysis.”””

```
filename = file.filename or "unknown"
ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

if ext not in ALLOWED_EXTENSIONS:
    raise HTTPException(
        status_code=400,
        detail=f"Ungültiges Dateiformat. Erlaubt: {', '.join(ALLOWED_EXTENSIONS)}"
    )

content = await file.read()

if len(content) > MAX_FILE_SIZE:
    raise HTTPException(
        status_code=400,
        detail=f"Datei zu groß. Maximum: {MAX_FILE_SIZE // (1024*1024)} MB"
    )

image_hash = hashlib.sha256(content).hexdigest()[:16]
image_id = f"img_{image_hash}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

uploaded_images[image_id] = {
    "content": base64.b64encode(content).decode(),
    "filename": filename,
    "content_type": file.content_type,
    "uploaded_at": datetime.now().isoformat()
}

return UploadResponse(
    success=True,
    image_id=image_id,
    message="Bild erfolgreich hochgeladen"
)
```

@app.get(”/api/search”, response_model=SearchResponse)
async def search(
q: Optional[str] = Query(None, description=“Suchbegriffe”),
image_id: Optional[str] = Query(None, description=“ID des hochgeladenen Bildes”),
epoch: Optional[str] = Query(None, description=“Epoche”),
object_type: Optional[str] = Query(None, description=“Objekttyp”),
region: Optional[str] = Query(None, description=“Region”),
limit: int = Query(20, ge=1, le=50, description=“Maximale Ergebnisse”)
):
“”“Search for similar archaeological objects.”””

```
filters = SearchFilters(
    epoch=epoch,
    object_type=object_type,
    region=region
)

query = build_europeana_query(keywords=q, filters=filters)

try:
    total, results = await search_europeana(query, rows=limit)
except HTTPException:
    raise
except Exception as e:
    raise HTTPException(
        status_code=500,
        detail=f"Suchfehler: {str(e)}"
    )

search_id = f"search_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hashlib.md5(query.encode()).hexdigest()[:8]}"

# Add similarity scores (will be real with CLIP in Phase 2)
import random
for i, result in enumerate(results):
    base_similarity = 95 - (i * 3)
    result.similarity = max(50, min(99, base_similarity + random.randint(-5, 5)))

results.sort(key=lambda x: x.similarity or 0, reverse=True)

return SearchResponse(
    success=True,
    total_results=total,
    results=results,
    search_id=search_id,
    filters_applied={
        "keywords": q,
        "epoch": epoch,
        "object_type": object_type,
        "region": region
    }
)
```

@app.get(”/api/filters”)
async def get_filters():
“”“Get available filter options.”””
return {
“epochs”: [
“Alle Epochen”,
“Steinzeit (bis 2200 v. Chr.)”,
“Bronzezeit (2200-800 v. Chr.)”,
“Eisenzeit (800 v. Chr. - 0)”,
“Römische Kaiserzeit (0-400 n. Chr.)”,
“Frühmittelalter (400-1000 n. Chr.)”,
“Hochmittelalter (1000-1300 n. Chr.)”,
“Spätmittelalter (1300-1500 n. Chr.)”
],
“object_types”: [
“Alle Objekttypen”,
“Fibeln & Gewandnadeln”,
“Münzen”,
“Keramik & Gefäße”,
“Waffen & Werkzeuge”,
“Schmuck & Zierrat”,
“Kultgegenstände”,
“Alltagsgegenstände”
],
“regions”: [
“Alle Regionen”,
“Mitteleuropa”,
“Nordeuropa”,
“Südeuropa”,
“Westeuropa”,
“Osteuropa”,
“Mittelmeerraum”,
“Naher Osten”
]
}

# ============================================================

# RUN SERVER

# ============================================================

if **name** == “**main**”:
import uvicorn
port = int(os.getenv(“PORT”, 8080))
uvicorn.run(app, host=“0.0.0.0”, port=port)
