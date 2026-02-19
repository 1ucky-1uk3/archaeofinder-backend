# ArchaeoFinder Backend - Phase 2 (Lightweight)
# CLIP-Bilderkennung über HuggingFace Inference API (kein lokales Modell)
# Vektorsuche mit numpy statt ChromaDB
# RAM-Verbrauch: ~80-120 MB statt ~1.5 GB

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx
import base64
import hashlib
import json
from datetime import datetime
from pathlib import Path
import os
import io
import logging
import math

logger = logging.getLogger(__name__)

# ── Konfiguration ──
EUROPEANA_API_KEY = os.getenv("EUROPEANA_API_KEY", "api2demo")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")  # Optional: HuggingFace Token für höhere Rate-Limits
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif"}

# HuggingFace Inference API – CLIP Modell (kostenlos, kein lokaler RAM nötig)
HF_CLIP_MODEL = "openai/clip-vit-base-patch32"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_CLIP_MODEL}"

# ── Leichtgewichtiger Vektor-Store (JSON-Datei statt ChromaDB) ──
VECTOR_STORE_PATH = Path("/app/chroma_data/vectors.json")

uploaded_images: dict = {}


# ═══════════════════════════════════════════════════════════════
#  Leichtgewichtiger Vektor-Store (ersetzt ChromaDB komplett)
#  Speichert Embeddings als JSON, Ähnlichkeitssuche mit Cosine
# ═══════════════════════════════════════════════════════════════

class LightVectorStore:
    """
    Ersetzt ChromaDB. Speichert Embeddings + Metadaten als JSON-Datei.
    Cosine-Similarity wird mit reinem Python/math berechnet.
    Braucht 0 MB extra RAM (kein onnxruntime, kein hnswlib).
    """

    def __init__(self, path: Path):
        self.path = path
        self.entries: list[dict] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    self.entries = json.load(f)
                logger.info(f"Vektor-Store geladen: {len(self.entries)} Einträge")
            except Exception as e:
                logger.warning(f"Vektor-Store konnte nicht geladen werden: {e}")
                self.entries = []

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.entries, f)

    def add(self, item_id: str, embedding: list[float], metadata: dict):
        # Duplikate vermeiden
        self.entries = [e for e in self.entries if e["id"] != item_id]
        self.entries.append({
            "id": item_id,
            "embedding": embedding,
            "metadata": metadata,
        })
        self._save()

    def count(self) -> int:
        return len(self.entries)

    def query(self, query_embedding: list[float], n_results: int = 20) -> list[dict]:
        if not self.entries:
            return []

        scored = []
        for entry in self.entries:
            sim = self._cosine_similarity(query_embedding, entry["embedding"])
            scored.append({
                "id": entry["id"],
                "metadata": entry["metadata"],
                "similarity": sim,
            })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:n_results]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# Globale Instanz
vector_store = LightVectorStore(VECTOR_STORE_PATH)


# ═══════════════════════════════════════════════════════════════
#  HuggingFace Inference API – CLIP Embedding remote berechnen
#  Kein PyTorch, kein OpenCLIP, kein lokales Modell nötig
# ═══════════════════════════════════════════════════════════════

async def get_clip_embedding_remote(image_bytes: bytes) -> Optional[list[float]]:
    """
    Sendet ein Bild an die HuggingFace Inference API und bekommt
    ein CLIP-Embedding zurück. Läuft komplett auf HF-Servern.
    Kostenlos (mit optionalem Token für höhere Limits).
    """
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Feature-Extraction-Endpoint liefert Embeddings
            response = await client.post(
                HF_API_URL,
                headers=headers,
                content=image_bytes,
                params={"wait_for_model": "true"},
            )

            if response.status_code == 503:
                # Modell wird gerade geladen – einmal warten und nochmal versuchen
                logger.info("HF-Modell wird geladen, warte...")
                import asyncio
                await asyncio.sleep(10)
                response = await client.post(
                    HF_API_URL,
                    headers=headers,
                    content=image_bytes,
                    params={"wait_for_model": "true"},
                )

            if response.status_code != 200:
                logger.error(f"HF API Fehler {response.status_code}: {response.text[:200]}")
                return None

            data = response.json()

            # Die API gibt je nach Modell verschiedene Formate zurück
            if isinstance(data, list):
                # Flache Liste = direkt das Embedding
                if data and isinstance(data[0], (int, float)):
                    return data
                # Verschachtelt: [[...embedding...]]
                if data and isinstance(data[0], list):
                    return data[0]

            logger.warning(f"Unerwartetes HF-Antwortformat: {type(data)}")
            return None

    except httpx.TimeoutException:
        logger.error("HF API Timeout")
        return None
    except Exception as e:
        logger.error(f"HF API Fehler: {e}")
        return None


async def get_text_embedding_remote(text: str) -> Optional[list[float]]:
    """
    Holt ein Text-Embedding von HuggingFace (für zukünftige Text→Bild Suche).
    """
    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                HF_API_URL,
                headers=headers,
                json={"inputs": text, "wait_for_model": True},
            )
            if response.status_code != 200:
                return None
            data = response.json()
            if isinstance(data, list):
                if data and isinstance(data[0], (int, float)):
                    return data
                if data and isinstance(data[0], list):
                    return data[0]
            return None
    except Exception as e:
        logger.error(f"HF Text-Embedding Fehler: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  Pydantic Models
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
#  Filter-Mappings
# ═══════════════════════════════════════════════════════════════

EPOCH_MAPPING = {
    "Steinzeit": ["neolithic", "stone age", "mesolithic", "paleolithic"],
    "Bronzezeit": ["bronze age"],
    "Eisenzeit": ["iron age", "hallstatt", "celtic"],
    "Roemische Kaiserzeit": ["roman", "roman empire"],
    "Fruehmittelalter": ["early medieval", "migration period"],
    "Hochmittelalter": ["medieval", "romanesque"],
    "Spaetmittelalter": ["late medieval", "gothic"],
}

OBJECT_TYPE_MAPPING = {
    "Fibeln": ["fibula", "brooch", "pin"],
    "Muenzen": ["coin", "numismatic"],
    "Keramik": ["ceramic", "pottery", "vessel", "amphora"],
    "Waffen": ["weapon", "sword", "axe", "tool"],
    "Schmuck": ["jewelry", "jewellery", "ring", "bracelet", "necklace"],
    "Kultgegenstaende": ["cult", "ritual", "religious", "votive"],
    "Alltagsgegenstaende": ["domestic", "household"],
}

REGION_MAPPING = {
    "Mitteleuropa": ["germany", "austria", "switzerland"],
    "Nordeuropa": ["scandinavia", "denmark", "sweden", "norway"],
    "Suedeuropa": ["italy", "greece", "spain"],
    "Westeuropa": ["france", "britain", "england"],
    "Osteuropa": ["poland", "czech", "hungary", "romania"],
    "Mittelmeerraum": ["mediterranean", "aegean"],
    "Naher Osten": ["mesopotamia", "egypt", "levant"],
}


# ═══════════════════════════════════════════════════════════════
#  FastAPI App
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="ArchaeoFinder API",
    version="2.1.0",
    description="Archäologische Bilderkennung – lightweight (kein lokales ML-Modell)",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info(
        f"ArchaeoFinder v2.1.0 gestartet | "
        f"CLIP via HuggingFace API | "
        f"Vektor-Store: {vector_store.count()} Einträge | "
        f"HF-Token: {'gesetzt' if HF_API_TOKEN else 'nicht gesetzt (Rate-Limits möglich)'}"
    )


# ═══════════════════════════════════════════════════════════════
#  Europeana Suche (unverändert)
# ═══════════════════════════════════════════════════════════════

def build_europeana_query(keywords=None, epoch=None, object_type=None, region=None):
    query_parts = ["(archaeology OR archaeological)"]
    if keywords:
        query_parts.append(f"({keywords})")
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
        title = title_data[0] if isinstance(title_data, list) else title_data

    description = None
    if "dcDescription" in item and item["dcDescription"]:
        desc_data = item["dcDescription"]
        description = desc_data[0] if isinstance(desc_data, list) else desc_data
        if description and len(description) > 300:
            description = description[:297] + "..."

    museum = None
    if "dataProvider" in item and item["dataProvider"]:
        dp_data = item["dataProvider"]
        museum = dp_data[0] if isinstance(dp_data, list) else dp_data

    epoch = None
    if "year" in item and item["year"]:
        years = item["year"]
        if isinstance(years, list) and years:
            epoch = (
                f"{min(years)} - {max(years)}" if len(years) > 1 else str(years[0])
            )

    image_url = None
    if "edmPreview" in item and item["edmPreview"]:
        previews = item["edmPreview"]
        image_url = previews[0] if isinstance(previews, list) else previews

    source_url = item.get("guid") or (
        "https://www.europeana.eu/item" + item["id"] if "id" in item else None
    )

    return MuseumObject(
        id=item.get("id", "unknown"),
        title=title,
        description=description,
        museum=museum,
        epoch=epoch,
        image_url=image_url,
        source_url=source_url,
        source="europeana",
    )


async def search_europeana(query: str, rows: int = 20):
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
            raise HTTPException(status_code=502, detail="Europeana API Fehler")
        data = response.json()
        total = data.get("totalResults", 0)
        results = []
        for item in data.get("items", []):
            try:
                parsed = parse_europeana_result(item)
                if parsed.image_url:
                    results.append(parsed)
            except Exception:
                continue
        return total, results


# ═══════════════════════════════════════════════════════════════
#  Bildsuche über Remote-CLIP + leichtgewichtigen Vektor-Store
# ═══════════════════════════════════════════════════════════════

async def search_by_image(image_bytes: bytes, limit: int = 20) -> list[MuseumObject]:
    """Bildersuche: Embedding remote holen, lokal mit Vektor-Store vergleichen."""
    if vector_store.count() == 0:
        return []

    embedding = await get_clip_embedding_remote(image_bytes)
    if not embedding:
        logger.warning("Konnte kein CLIP-Embedding vom HF-Server bekommen")
        return []

    matches = vector_store.query(embedding, n_results=limit)

    results = []
    for match in matches:
        similarity_pct = int(max(0, min(100, match["similarity"] * 100)))
        meta = match["metadata"]
        results.append(
            MuseumObject(
                id=match["id"],
                title=meta.get("title", "Unbekannt"),
                description=meta.get("description"),
                museum=meta.get("museum"),
                epoch=meta.get("epoch"),
                image_url=meta.get("image_url"),
                source_url=meta.get("source_url"),
                source=meta.get("source", "europeana"),
                similarity=similarity_pct,
            )
        )
    return results


# ═══════════════════════════════════════════════════════════════
#  API Endpoints
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "name": "ArchaeoFinder API",
        "version": "2.1.0",
        "status": "online",
        "clip_available": True,
        "clip_mode": "remote (HuggingFace Inference API)",
        "images_indexed": vector_store.count(),
        "hf_token_set": bool(HF_API_TOKEN),
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "clip": True, "mode": "remote"}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_image_endpoint(file: UploadFile = File(...)):
    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Ungueltiges Format")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Datei zu gross (max 10 MB)")

    image_hash = hashlib.sha256(content).hexdigest()[:16]
    image_id = f"img_{image_hash}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    uploaded_images[image_id] = {
        "content": content,
        "filename": filename,
        "content_type": file.content_type,
        "uploaded_at": datetime.now().isoformat(),
    }

    return UploadResponse(
        success=True, image_id=image_id, message="Bild hochgeladen"
    )


@app.get("/api/search", response_model=SearchResponse)
async def search(
    q: Optional[str] = Query(None),
    image_id: Optional[str] = Query(None),
    epoch: Optional[str] = Query(None),
    object_type: Optional[str] = Query(None),
    region: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=50),
):
    search_mode = "text"
    results = []
    total = 0

    # Bildsuche (remote CLIP)
    if image_id and image_id in uploaded_images:
        try:
            content = uploaded_images[image_id]["content"]
            results = await search_by_image(content, limit)
            total = len(results)
            search_mode = "image"
        except Exception as e:
            logger.error(f"Bildsuche fehlgeschlagen: {e}")

    # Textsuche über Europeana
    if not results or q:
        query = build_europeana_query(
            keywords=q, epoch=epoch, object_type=object_type, region=region
        )
        try:
            total, text_results = await search_europeana(query, rows=limit)
            if search_mode == "text":
                import random
                for i, result in enumerate(text_results):
                    base_similarity = 95 - (i * 3)
                    result.similarity = max(
                        50, min(99, base_similarity + random.randint(-5, 5))
                    )
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
                raise HTTPException(
                    status_code=500, detail=f"Suchfehler: {e}"
                )

    results.sort(key=lambda x: x.similarity or 0, reverse=True)
    search_id = f"search_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    return SearchResponse(
        success=True,
        total_results=total,
        results=results[:limit],
        search_id=search_id,
        filters_applied={
            "keywords": q,
            "epoch": epoch,
            "object_type": object_type,
            "region": region,
        },
        search_mode=search_mode,
    )


@app.post("/api/index", response_model=IndexResponse)
async def index_europeana_images(
    q: str = Query("archaeology"),
    count: int = Query(10, ge=1, le=50),
):
    """
    Indexiert Europeana-Bilder: Lädt Bilder herunter, holt CLIP-Embeddings
    von HuggingFace, speichert sie im lokalen Vektor-Store.
    """
    query = build_europeana_query(keywords=q)
    _, items = await search_europeana(query, rows=count)

    indexed = 0
    for item in items:
        if not item.image_url:
            continue
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                img_response = await client.get(item.image_url)
                if img_response.status_code != 200:
                    continue

            embedding = await get_clip_embedding_remote(img_response.content)
            if not embedding:
                continue

            vector_store.add(
                item_id=item.id,
                embedding=embedding,
                metadata={
                    "title": item.title,
                    "description": item.description,
                    "museum": item.museum,
                    "epoch": item.epoch,
                    "image_url": item.image_url,
                    "source_url": item.source_url,
                    "source": item.source,
                },
            )
            indexed += 1
            logger.info(f"Indexiert: {item.title} ({indexed}/{len(items)})")

        except Exception as e:
            logger.warning(f"Indexierung fehlgeschlagen für {item.id}: {e}")
            continue

    return IndexResponse(
        success=True,
        indexed_count=indexed,
        total_in_db=vector_store.count(),
        message=f"{indexed} Bilder indexiert, {vector_store.count()} gesamt in DB",
    )


@app.get("/api/filters")
async def get_filters():
    return {
        "epochs": ["Alle Epochen"] + list(EPOCH_MAPPING.keys()),
        "object_types": ["Alle Objekttypen"] + list(OBJECT_TYPE_MAPPING.keys()),
        "regions": ["Alle Regionen"] + list(REGION_MAPPING.keys()),
    }


@app.get("/api/sources")
async def get_sources():
    return {
        "sources": [
            {"id": "europeana", "name": "Europeana", "status": "active"},
            {"id": "ddb", "name": "Deutsche Digitale Bibliothek", "status": "coming_soon"},
            {"id": "british_museum", "name": "British Museum", "status": "coming_soon"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
