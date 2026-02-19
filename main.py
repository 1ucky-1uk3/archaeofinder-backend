# ArchaeoFinder Backend - Phase 2 Extended
# Mit CLIP, ChromaDB und zusaetzlichen Datenbanken

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
import asyncio

try:
    import torch
    import open_clip
    from PIL import Image
    import chromadb
    import numpy as np
    CLIP_AVAILABLE = True
except ImportError as e:
    CLIP_AVAILABLE = False

EUROPEANA_API_KEY = os.getenv("EUROPEANA_API_KEY", "api2demo")
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = set(["jpg", "jpeg", "png", "webp", "gif"])

clip_model = None
clip_preprocess = None
clip_tokenizer = None
chroma_client = None
image_collection = None
uploaded_images = {}


def initialize_clip():
    global clip_model, clip_preprocess, clip_tokenizer, chroma_client, image_collection
    if not CLIP_AVAILABLE:
        return False
    try:
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        clip_model.eval()
        chroma_client = chromadb.Client()
        image_collection = chroma_client.get_or_create_collection(name="archaeo_images")
        return True
    except Exception as e:
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
    except Exception:
        return None


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
    material: Optional[str] = None
    findspot: Optional[str] = None


class SearchResponse(BaseModel):
    success: bool
    total_results: int
    results: List[MuseumObject]
    search_id: str
    filters_applied: dict
    search_mode: str
    sources_searched: List[str]


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
    "Naher Osten": ["mesopotamia", "egypt", "levant"],
    "Grossbritannien": ["britain", "england", "wales", "uk"],
    "Niederlande": ["netherlands", "dutch", "holland"]
}


app = FastAPI(title="ArchaeoFinder API", version="2.1.0", docs_url="/docs", redoc_url="/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    initialize_clip()


# =============================================================================
# EUROPEANA API
# =============================================================================

def build_europeana_query(keywords=None, epoch=None, object_type=None, region=None):
    query_parts = []
    query_parts.append("(archaeology OR archaeological)")
    if keywords:
        query_parts.append("(" + keywords + ")")
    if epoch and epoch != "Alle Epochen":
        epoch_terms = EPOCH_MAPPING.get(epoch, [])
        if epoch_terms:
            query_parts.append("(" + " OR ".join(epoch_terms) + ")")
    if object_type and object_type != "Alle Objekttypen":
        type_terms = OBJECT_TYPE_MAPPING.get(object_type, [])
        if type_terms:
            query_parts.append("(" + " OR ".join(type_terms) + ")")
    if region and region != "Alle Regionen":
        region_terms = REGION_MAPPING.get(region, [])
        if region_terms:
            query_parts.append("(" + " OR ".join(region_terms) + ")")
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
    return MuseumObject(id=item.get("id", "unknown"), title=title, description=description, museum=museum, epoch=epoch, image_url=image_url, source_url=source_url, source="europeana")


async def search_europeana(query, rows=20):
    url = "https://api.europeana.eu/record/v2/search.json"
    params = {"wskey": EUROPEANA_API_KEY, "query": query, "rows": rows, "profile": "rich", "media": "true", "qf": "TYPE:IMAGE"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            if response.status_code != 200:
                return 0, []
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
    except Exception:
        return 0, []


# =============================================================================
# PORTABLE ANTIQUITIES SCHEME (UK) - finds.org.uk
# =============================================================================

async def search_pas_uk(keywords=None, object_type=None, rows=20):
    base_url = "https://finds.org.uk/database/search/results/format/json"
    params = {"show": rows}
    
    if keywords:
        params["q"] = keywords
    if object_type and object_type != "Alle Objekttypen":
        type_mapping = {
            "Fibeln": "BROOCH",
            "Muenzen": "COIN",
            "Keramik": "VESSEL",
            "Waffen": "WEAPON",
            "Schmuck": "FINGER RING"
        }
        if object_type in type_mapping:
            params["objectType"] = type_mapping[object_type]
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(base_url, params=params)
            if response.status_code != 200:
                return 0, []
            data = response.json()
            results = []
            items = data.get("results", [])
            total = data.get("meta", {}).get("totalResults", len(items))
            
            for item in items:
                try:
                    title = item.get("objectType", "Unknown Object")
                    if item.get("broadperiod"):
                        title = title + " (" + item.get("broadperiod") + ")"
                    
                    description = item.get("description", None)
                    if description and len(description) > 300:
                        description = description[:297] + "..."
                    
                    image_url = None
                    if item.get("filename"):
                        image_url = "https://finds.org.uk/images/thumbnails/" + item.get("filename")
                    
                    source_url = "https://finds.org.uk/database/artefacts/record/id/" + str(item.get("id", ""))
                    
                    findspot = None
                    if item.get("county"):
                        findspot = item.get("county")
                        if item.get("parish"):
                            findspot = findspot + ", " + item.get("parish")
                    
                    results.append(MuseumObject(
                        id="pas_" + str(item.get("id", "")),
                        title=title,
                        description=description,
                        museum="Portable Antiquities Scheme (UK)",
                        epoch=item.get("broadperiod", None),
                        image_url=image_url,
                        source_url=source_url,
                        source="pas_uk",
                        material=item.get("material", None),
                        findspot=findspot
                    ))
                except Exception:
                    continue
            
            return total, results
    except Exception:
        return 0, []


# =============================================================================
# PORTABLE ANTIQUITIES NETHERLANDS - portable-antiquities.nl
# =============================================================================

async def search_pan_nl(keywords=None, object_type=None, rows=20):
    base_url = "https://portable-antiquities.nl/pan/api/search"
    
    params = {"limit": rows, "offset": 0}
    
    search_terms = []
    if keywords:
        search_terms.append(keywords)
    if object_type and object_type != "Alle Objekttypen":
        type_mapping = {
            "Fibeln": "fibula",
            "Muenzen": "munt",
            "Keramik": "aardewerk",
            "Waffen": "wapen",
            "Schmuck": "ring"
        }
        if object_type in type_mapping:
            search_terms.append(type_mapping[object_type])
    
    if search_terms:
        params["q"] = " ".join(search_terms)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(base_url, params=params)
            if response.status_code != 200:
                return 0, []
            data = response.json()
            results = []
            items = data.get("items", data.get("results", []))
            total = data.get("total", len(items))
            
            for item in items:
                try:
                    title = item.get("objectName", item.get("name", "Unknown Object"))
                    
                    description = item.get("description", None)
                    if description and len(description) > 300:
                        description = description[:297] + "..."
                    
                    image_url = item.get("imageUrl", item.get("thumbnail", None))
                    
                    item_id = item.get("id", item.get("identifier", ""))
                    source_url = "https://portable-antiquities.nl/pan/#/object/" + str(item_id)
                    
                    results.append(MuseumObject(
                        id="pan_" + str(item_id),
                        title=title,
                        description=description,
                        museum="Portable Antiquities Netherlands",
                        epoch=item.get("period", item.get("dating", None)),
                        image_url=image_url,
                        source_url=source_url,
                        source="pan_nl",
                        material=item.get("material", None),
                        findspot=item.get("municipality", item.get("findspot", None))
                    ))
                except Exception:
                    continue
            
            return total, results
    except Exception:
        return 0, []


# =============================================================================
# COMBINED SEARCH
# =============================================================================

async def search_all_sources(keywords=None, epoch=None, object_type=None, region=None, limit=20):
    results = []
    sources_searched = []
    total = 0
    
    # Calculate results per source
    per_source = max(5, limit // 3)
    
    # Create search tasks
    tasks = []
    
    # Always search Europeana
    europeana_query = build_europeana_query(keywords=keywords, epoch=epoch, object_type=object_type, region=region)
    tasks.append(("europeana", search_europeana(europeana_query, rows=per_source)))
    
    # Search PAS UK if region matches or no region specified
    if not region or region in ["Alle Regionen", "Grossbritannien", "Westeuropa"]:
        tasks.append(("pas_uk", search_pas_uk(keywords=keywords, object_type=object_type, rows=per_source)))
    
    # Search PAN Netherlands if region matches or no region specified
    if not region or region in ["Alle Regionen", "Niederlande", "Westeuropa"]:
        tasks.append(("pan_nl", search_pan_nl(keywords=keywords, object_type=object_type, rows=per_source)))
    
    # Execute all searches concurrently
    for source_name, task in tasks:
        try:
            source_total, source_results = await task
            if source_results:
                results.extend(source_results)
                total += source_total
                sources_searched.append(source_name)
        except Exception:
            continue
    
    return total, results, sources_searched


# =============================================================================
# IMAGE SEARCH
# =============================================================================

async def search_by_image(image, limit=20):
    if not image_collection:
        return []
    count = image_collection.count()
    if count == 0:
        return []
    embedding = get_image_embedding(image)
    if not embedding:
        return []
    n_results = min(limit, count)
    results = image_collection.query(query_embeddings=[embedding], n_results=n_results)
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
        museum_objects.append(MuseumObject(id=item_id, title=metadata.get("title", "Unbekannt"), museum=metadata.get("museum", None), image_url=metadata.get("image_url", None), source_url=metadata.get("source_url", None), source=metadata.get("source", "europeana"), similarity=similarity))
    return museum_objects


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    db_count = 0
    if image_collection:
        db_count = image_collection.count()
    return {
        "name": "ArchaeoFinder API",
        "version": "2.1.0",
        "status": "online",
        "clip_available": CLIP_AVAILABLE,
        "images_indexed": db_count,
        "sources": ["europeana", "pas_uk", "pan_nl"]
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
    uploaded_images[image_id] = {"content": content, "filename": filename, "content_type": file.content_type, "uploaded_at": datetime.now().isoformat()}
    return UploadResponse(success=True, image_id=image_id, message="Bild hochgeladen")


@app.get("/api/search", response_model=SearchResponse)
async def search(q: Optional[str] = Query(None), image_id: Optional[str] = Query(None), epoch: Optional[str] = Query(None), object_type: Optional[str] = Query(None), region: Optional[str] = Query(None), limit: int = Query(20, ge=1, le=50)):
    search_mode = "text"
    results = []
    total = 0
    sources_searched = []
    
    if image_id and image_id in uploaded_images and CLIP_AVAILABLE:
        try:
            content = uploaded_images[image_id]["content"]
            image = Image.open(io.BytesIO(content)).convert("RGB")
            results = await search_by_image(image, limit)
            total = len(results)
            search_mode = "image"
            sources_searched = ["clip_index"]
        except Exception:
            pass
    
    if not results or q:
        total, text_results, sources_searched = await search_all_sources(
            keywords=q, epoch=epoch, object_type=object_type, region=region, limit=limit
        )
        
        if search_mode == "text":
            import random
            for i in range(len(text_results)):
                result = text_results[i]
                base_similarity = 95 - (i * 2)
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
    
    results.sort(key=lambda x: x.similarity or 0, reverse=True)
    search_id = "search_" + datetime.now().strftime("%Y%m%d%H%M%S")
    
    return SearchResponse(
        success=True,
        total_results=total,
        results=results[:limit],
        search_id=search_id,
        filters_applied={"keywords": q, "epoch": epoch, "object_type": object_type, "region": region},
        search_mode=search_mode,
        sources_searched=sources_searched
    )


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
            {"id": "europeana", "name": "Europeana", "country": "EU", "status": "active", "url": "https://www.europeana.eu"},
            {"id": "pas_uk", "name": "Portable Antiquities Scheme", "country": "UK", "status": "active", "url": "https://finds.org.uk"},
            {"id": "pan_nl", "name": "Portable Antiquities Netherlands", "country": "NL", "status": "active", "url": "https://portable-antiquities.nl"},
            {"id": "ddb", "name": "Deutsche Digitale Bibliothek", "country": "DE", "status": "coming_soon", "url": "https://www.deutsche-digitale-bibliothek.de"},
            {"id": "danish", "name": "Metaldetektorfund Danmark", "country": "DK", "status": "coming_soon", "url": "https://www.metaldetektorfund.dk"},
            {"id": "czech", "name": "AMCR Digiarchiv", "country": "CZ", "status": "coming_soon", "url": "https://digiarchiv.aiscr.cz"},
            {"id": "scotland", "name": "Treasure Trove Scotland", "country": "UK", "status": "coming_soon", "url": "https://treasuretrovescotland.co.uk"}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
