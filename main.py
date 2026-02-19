# ArchaeoFinder Backend - Phase 2 Extended
# Mit CLIP, ChromaDB, Europeana, PAS, PAN und DDB

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
    from huggingface_hub import login as hf_login
    CLIP_AVAILABLE = True
except ImportError as e:
    CLIP_AVAILABLE = False

# API Keys aus den Umgebungsvariablen laden
EUROPEANA_API_KEY = os.getenv("EUROPEANA_API_KEY", "api2demo")
DDB_API_KEY = os.getenv("DDB_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

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
        
    # Hugging Face Login durchführen, falls Token vorhanden (verhindert Rate-Limits)
    if HF_TOKEN:
        try:
            hf_login(token=HF_TOKEN)
            print("Erfolgreich bei Hugging Face eingeloggt.")
        except Exception as e:
            print(f"Hugging Face Login fehlgeschlagen: {e}")

    try:
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        clip_model.eval()
        chroma_client = chromadb.Client()
        image_collection = chroma_client.get_or_create_collection(name="archaeo_images")
        return True
    except Exception as e:
        print(f"CLIP Initialisierung fehlgeschlagen: {e}")
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
    "Fibeln": ["fibula", "brooch", "pin", "fibel"],
    "Muenzen": ["coin", "numismatic", "münze", "muenze"],
    "Keramik": ["ceramic", "pottery", "vessel", "amphora", "keramik"],
    "Waffen": ["weapon", "sword", "axe", "tool", "waffe", "schwert"],
    "Schmuck": ["jewelry", "jewellery", "ring", "bracelet", "necklace", "schmuck"],
    "Kultgegenstaende": ["cult", "ritual", "religious", "votive", "kult"],
    "Alltagsgegenstaende": ["domestic", "household", "alltag"]
}

REGION_MAPPING = {
    "Mitteleuropa": ["germany", "austria", "switzerland", "deutschland", "österreich", "schweiz"],
    "Nordeuropa": ["scandinavia", "denmark", "sweden", "norway"],
    "Suedeuropa": ["italy", "greece", "spain"],
    "Westeuropa": ["france", "britain", "england"],
    "Osteuropa": ["poland", "czech", "hungary", "romania"],
    "Mittelmeerraum": ["mediterranean", "aegean"],
    "Naher Osten": ["mesopotamia", "egypt", "levant"],
    "Grossbritannien": ["britain", "england", "wales", "uk"],
    "Niederlande": ["netherlands", "dutch", "holland"]
}


app = FastAPI(title="ArchaeoFinder API", version="2.2.0", docs_url="/docs", redoc_url="/redoc")

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
                    title_data = item.get("title", ["Unbekanntes Objekt"])
                    title = title_data[0] if isinstance(title_data, list) else title_data
                    
                    previews = item.get("edmPreview", [])
                    image_url = previews[0] if isinstance(previews, list) else previews
                    
                    if image_url:
                        results.append(MuseumObject(
                            id=item.get("id", "unknown"), 
                            title=title, 
                            museum=item.get("dataProvider", [None])[0] if isinstance(item.get("dataProvider"), list) else item.get("dataProvider"),
                            image_url=image_url, 
                            source_url=item.get("guid", f"https://www.europeana.eu/item{item.get('id', '')}"),
                            source="europeana"
                        ))
                except Exception:
                    continue
            return total, results
    except Exception:
        return 0, []


# =============================================================================
# PORTABLE ANTIQUITIES SCHEME (UK)
# =============================================================================

async def search_pas_uk(keywords=None, object_type=None, rows=20):
    base_url = "https://finds.org.uk/database/search/results/format/json"
    params = {"show": rows}
    if keywords: params["q"] = keywords
    
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
                    image_url = f"https://finds.org.uk/images/thumbnails/{item.get('filename')}" if item.get("filename") else None
                    if image_url:
                        results.append(MuseumObject(
                            id=f"pas_{item.get('id', '')}",
                            title=item.get("objectType", "Unknown Object"),
                            museum="Portable Antiquities Scheme (UK)",
                            epoch=item.get("broadperiod", None),
                            image_url=image_url,
                            source_url=f"https://finds.org.uk/database/artefacts/record/id/{item.get('id', '')}",
                            source="pas_uk"
                        ))
                except Exception:
                    continue
            return total, results
    except Exception:
        return 0, []


# =============================================================================
# PORTABLE ANTIQUITIES NETHERLANDS (PAN)
# =============================================================================

async def search_pan_nl(keywords=None, rows=20):
    base_url = "https://portable-antiquities.nl/pan/api/search"
    params = {"limit": rows, "offset": 0}
    if keywords: params["q"] = keywords
    
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
                    image_url = item.get("imageUrl", item.get("thumbnail", None))
                    item_id = item.get("id", item.get("identifier", ""))
                    if image_url:
                        results.append(MuseumObject(
                            id=f"pan_{item_id}",
                            title=item.get("objectName", item.get("name", "Unknown Object")),
                            museum="Portable Antiquities Netherlands",
                            image_url=image_url,
                            source_url=f"https://portable-antiquities.nl/pan/#/object/{item_id}",
                            source="pan_nl"
                        ))
                except Exception:
                    continue
            return total, results
    except Exception:
        return 0, []


# =============================================================================
# DEUTSCHE DIGITALE BIBLIOTHEK (DDB)
# =============================================================================

async def search_ddb(keywords=None, rows=20):
    if not DDB_API_KEY:
        return 0, []
        
    base_url = "https://api.deutsche-digitale-bibliothek.de/search"
    query_parts = ["(Archäologie OR archaeology)"]
    if keywords:
        query_parts.append(f"({keywords})")
        
    params = {
        "oauth_consumer_key": DDB_API_KEY,
        "query": " AND ".join(query_parts),
        "rows": rows
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"Accept": "application/json"}
            response = await client.get(base_url, params=params, headers=headers)
            if response.status_code != 200:
                return 0, []
            
            data = response.json()
            results = []
            
            # Die DDB API verpackt Ergebnisse oft in "results" -> "docs"
            items_container = data.get("results", [])
            docs = items_container[0].get("docs", []) if items_container else []
            total = data.get("numberOfResults", len(docs))
            
            for item in docs:
                try:
                    item_id = item.get("id", "")
                    title = item.get("title", "Unbekanntes Objekt")
                    
                    # DDB Thumbnails haben meist dieses feste Muster
                    image_url = f"https://api.deutsche-digitale-bibliothek.de/binary/{item_id}/list/1.jpg" if item.get("thumbnail") else None
                    
                    if image_url:
                        results.append(MuseumObject(
                            id=f"ddb_{item_id}",
                            title=title,
                            museum=item.get("provider", "Deutsche Digitale Bibliothek"),
                            image_url=image_url,
                            source_url=f"https://www.deutsche-digitale-bibliothek.de/item/{item_id}",
                            source="ddb"
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
    
    # Ergebnisse dynamisch aufteilen
    active_sources = 3
    if DDB_API_KEY and (not region or region in ["Alle Regionen", "Mitteleuropa"]):
        active_sources += 1
        
    per_source = max(5, limit // active_sources)
    tasks = []
    
    # 1. Europeana
    europeana_query = build_europeana_query(keywords=keywords, epoch=epoch, object_type=object_type, region=region)
    tasks.append(("europeana", search_europeana(europeana_query, rows=per_source)))
    
    # 2. PAS UK
    if not region or region in ["Alle Regionen", "Grossbritannien", "Westeuropa"]:
        tasks.append(("pas_uk", search_pas_uk(keywords=keywords, object_type=object_type, rows=per_source)))
    
    # 3. PAN Netherlands
    if not region or region in ["Alle Regionen", "Niederlande", "Westeuropa"]:
        tasks.append(("pan_nl", search_pan_nl(keywords=keywords, rows=per_source)))
        
    # 4. DDB (Deutschland)
    if DDB_API_KEY and (not region or region in ["Alle Regionen", "Mitteleuropa"]):
        tasks.append(("ddb", search_ddb(keywords=keywords, rows=per_source)))
    
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
    if not image_collection: return []
    if image_collection.count() == 0: return []
    embedding = get_image_embedding(image)
    if not embedding: return []
    
    results = image_collection.query(query_embeddings=[embedding], n_results=min(limit, image_collection.count()))
    if not results or not results.get("ids") or not results["ids"][0]: return []
    
    museum_objects = []
    for i, item_id in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][i] if "metadatas" in results and results["metadatas"][0] else {}
        distance = results["distances"][0][i] if "distances" in results and results["distances"][0] else 1.0
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


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "name": "ArchaeoFinder API",
        "version": "2.2.0",
        "status": "online",
        "clip_available": CLIP_AVAILABLE,
        "images_indexed": image_collection.count() if image_collection else 0,
        "sources": ["europeana", "pas_uk", "pan_nl"] + (["ddb"] if DDB_API_KEY else [])
    }

@app.post("/api/upload", response_model=UploadResponse)
async def upload_image_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > MAX_FILE_SIZE: raise HTTPException(status_code=400, detail="Datei zu gross")
    
    image_id = f"img_{hashlib.sha256(content).hexdigest()[:16]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    uploaded_images[image_id] = {"content": content, "filename": file.filename, "content_type": file.content_type}
    return UploadResponse(success=True, image_id=image_id, message="Bild hochgeladen")

@app.get("/api/search", response_model=SearchResponse)
async def search(q: Optional[str] = Query(None), image_id: Optional[str] = Query(None), epoch: Optional[str] = Query(None), object_type: Optional[str] = Query(None), region: Optional[str] = Query(None), limit: int = Query(20, ge=1, le=50)):
    search_mode, results, total, sources_searched = "text", [], 0, []
    
    if image_id and image_id in uploaded_images and CLIP_AVAILABLE:
        try:
            image = Image.open(io.BytesIO(uploaded_images[image_id]["content"])).convert("RGB")
            results = await search_by_image(image, limit)
            total, search_mode, sources_searched = len(results), "image", ["clip_index"]
        except Exception: pass
    
    if not results or q:
        total, text_results, sources_searched = await search_all_sources(keywords=q, epoch=epoch, object_type=object_type, region=region, limit=limit)
        
        if search_mode == "text":
            import random
            for i, r in enumerate(text_results):
                r.similarity = max(50, min(99, (95 - (i * 2)) + random.randint(-5, 5)))
            results = text_results
        else:
            search_mode = "hybrid"
            existing_ids = {r.id for r in results}
            for r in text_results:
                if r.id not in existing_ids:
                    r.similarity = 50
                    results.append(r)
    
    results.sort(key=lambda x: x.similarity or 0, reverse=True)
    return SearchResponse(
        success=True, total_results=total, results=results[:limit],
        search_id=f"search_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        filters_applied={"keywords": q, "epoch": epoch, "object_type": object_type, "region": region},
        search_mode=search_mode, sources_searched=sources_searched
    )

@app.get("/api/sources")
async def get_sources():
    return {
        "sources": [
            {"id": "europeana", "name": "Europeana", "country": "EU", "status": "active", "url": "https://www.europeana.eu"},
            {"id": "pas_uk", "name": "Portable Antiquities Scheme", "country": "UK", "status": "active", "url": "https://finds.org.uk"},
            {"id": "pan_nl", "name": "Portable Antiquities Netherlands", "country": "NL", "status": "active", "url": "https://portable-antiquities.nl"},
            {"id": "ddb", "name": "Deutsche Digitale Bibliothek", "country": "DE", "status": "active" if DDB_API_KEY else "needs_api_key", "url": "https://www.deutsche-digitale-bibliothek.de"},
            {"id": "danish", "name": "Metaldetektorfund Danmark", "country": "DK", "status": "coming_soon", "url": "https://www.metaldetektorfund.dk"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
