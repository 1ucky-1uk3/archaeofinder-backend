from fastapi import FastAPI, Query, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List, Dict
import httpx
import asyncio
import os
from datetime import datetime
import json
import hashlib
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("archaeofinder")

app = FastAPI(title="ArchaeoFinder API", version="4.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# CONFIGURATION
# =============================================================================

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://neyudzqjqbqfaxbfnglx.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

API_KEYS = {
    "europeana": os.getenv("EUROPEANA_API_KEY", ""),
    "smithsonian": os.getenv("SMITHSONIAN_API_KEY", ""),
    "harvard": os.getenv("HARVARD_API_KEY", ""),
    "rijksmuseum": os.getenv("RIJKSMUSEUM_API_KEY", ""),
    "cooper_hewitt": os.getenv("COOPER_HEWITT_API_KEY", ""),
}

# =============================================================================
# CONNECTION POOL (reuse connections across requests)
# =============================================================================

_http_client: Optional[httpx.AsyncClient] = None

async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=12.0,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            follow_redirects=True
        )
    return _http_client

@app.on_event("shutdown")
async def shutdown():
    global _http_client
    if _http_client:
        await _http_client.aclose()

# =============================================================================
# IN-MEMORY CACHE
# =============================================================================

class SimpleCache:
    def __init__(self, ttl: int = 300):
        self._cache: Dict[str, tuple] = {}
        self._ttl = ttl

    def _make_key(self, *args) -> str:
        raw = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, *args):
        key = self._make_key(*args)
        if key in self._cache:
            data, ts = self._cache[key]
            if time.time() - ts < self._ttl:
                return data
            del self._cache[key]
        return None

    def set(self, value, *args):
        key = self._make_key(*args)
        self._cache[key] = (value, time.time())
        if len(self._cache) > 500:
            now = time.time()
            expired = [k for k, (_, ts) in self._cache.items() if now - ts > self._ttl]
            for k in expired:
                del self._cache[k]

    @property
    def size(self):
        return len(self._cache)

search_cache = SimpleCache(ttl=300)

# =============================================================================
# API HEALTH TRACKING
# =============================================================================

_api_health: Dict[str, dict] = {}

def record_api_health(source: str, status: str, response_time_ms: float = 0, result_count: int = 0):
    _api_health[source] = {
        "status": status,
        "last_check": datetime.utcnow().isoformat(),
        "response_time_ms": round(response_time_ms),
        "last_result_count": result_count,
    }

# =============================================================================
# FILTER MAPPINGS (extended with new epochs)
# =============================================================================

MET_DEPARTMENT_MAP = {
    "prehistoric": None, "neolithic": None, "bronze_age": None, "iron_age": None,
    "chalcolithic": None,
    "greek": 13, "roman": 13, "egyptian": 10, "medieval": 17, "islamic": 14, "asian": 6,
    "migration": 17, "merovingian": 17, "carolingian": 17, "early_medieval": 17,
    "late_medieval": 17, "viking": 17, "byzantine": 17,
    "near_eastern": 3, "arms_armor": 4,
}

EUROPEANA_EPOCH_FILTERS = {
    "prehistoric": "archaeology prehistory",
    "neolithic": "neolithic stone age",
    "chalcolithic": "chalcolithic copper age eneolithic",
    "bronze_age": "bronze age",
    "iron_age": "iron age",
    "greek": "ancient greek",
    "roman": "roman empire",
    "egyptian": "ancient egypt",
    "medieval": "medieval",
    "viking": "viking norse",
    "migration": "migration period germanic voelkerwanderung",
    "merovingian": "merovingian frankish",
    "carolingian": "carolingian",
    "early_medieval": "early medieval",
    "late_medieval": "late medieval gothic",
    "byzantine": "byzantine",
    "near_eastern": "ancient near east mesopotamia",
}

VA_CATEGORY_MAP = {
    "ceramic": "Ceramics", "pottery": "Ceramics",
    "glass": "Glass",
    "bronze": "Metalwork", "iron": "Metalwork", "gold": "Metalwork", "silver": "Metalwork",
    "copper": "Metalwork", "lead": "Metalwork",
    "coin": "Coins & Medals",
    "jewelry": "Jewellery",
    "bone": "Sculpture", "stone": "Sculpture", "ivory": "Sculpture",
    "amber": "Jewellery", "shell": "Jewellery",
    "wood": "Woodwork", "textile": "Textiles", "leather": "Textiles",
    "flint": None, "antler": None,
}

# =============================================================================
# DEDUPLICATION
# =============================================================================

def _normalize_title(title):
    if not title: return ""
    t = title.lower().strip()
    for prefix in ["a ", "an ", "the ", "fragment of ", "part of "]:
        if t.startswith(prefix): t = t[len(prefix):]
    return t

def _title_hash(title):
    n = _normalize_title(title)
    cleaned = "".join(c for c in n if c.isalnum())
    return cleaned[:60]

def _img_domain(url):
    if not url: return ""
    return url.split("?")[0].split("#")[0].lower()

def deduplicate_results(results):
    if not results or len(results) < 2: return results
    seen_img = set(); seen_title_hash = set(); unique = []
    for item in results:
        img_fp = _img_domain(item.get("image_url", ""))
        title_h = _title_hash(item.get("title", ""))
        if img_fp and img_fp in seen_img: continue
        if title_h and len(title_h) > 5 and title_h in seen_title_hash: continue
        unique.append(item)
        if img_fp: seen_img.add(img_fp)
        if title_h and len(title_h) > 5: seen_title_hash.add(title_h)
    return unique

# =============================================================================
# MODELS
# =============================================================================

class FindCreate(BaseModel):
    title: str
    description: Optional[str] = None
    object_type: Optional[str] = None
    epoch: Optional[str] = None
    material: Optional[str] = None
    dimensions: Optional[str] = None
    find_date: Optional[str] = None
    find_location: Optional[str] = None
    find_coordinates: Optional[dict] = None
    image_url: Optional[str] = None
    image_data: Optional[str] = None
    ai_labels: Optional[List[dict]] = None
    matched_artifacts: Optional[List[dict]] = None
    notes: Optional[str] = None
    is_public: Optional[bool] = False

class FindUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    object_type: Optional[str] = None
    epoch: Optional[str] = None
    material: Optional[str] = None
    dimensions: Optional[str] = None
    find_date: Optional[str] = None
    find_location: Optional[str] = None
    find_coordinates: Optional[dict] = None
    image_url: Optional[str] = None
    ai_labels: Optional[List[dict]] = None
    matched_artifacts: Optional[List[dict]] = None
    notes: Optional[str] = None
    is_public: Optional[bool] = None

# =============================================================================
# AUTH HELPER
# =============================================================================

async def get_user_from_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "): return None
    token = authorization.replace("Bearer ", "")
    client = await get_http_client()
    try:
        response = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_ANON_KEY},
            timeout=10.0
        )
        if response.status_code == 200: return response.json()
    except: pass
    return None

async def require_auth(authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user: raise HTTPException(status_code=401, detail="Not authenticated")
    return user

# =============================================================================
# MUSEUM API FUNCTIONS
# =============================================================================

async def search_europeana(client, query, limit=20, epoch=None, material=None):
    if not API_KEYS["europeana"]:
        return {"results": [], "source": "europeana", "status": "disabled"}
    cache_key = ("europeana", query, limit, epoch, material)
    cached = search_cache.get(*cache_key)
    if cached is not None:
        return {"results": cached, "source": "europeana", "status": "cached"}
    t0 = time.time()
    try:
        search_query = query
        if epoch and epoch in EUROPEANA_EPOCH_FILTERS:
            search_query = f"{query} {EUROPEANA_EPOCH_FILTERS[epoch]}"
        if material: search_query = f"{search_query} {material}"
        params = {"wskey": API_KEYS["europeana"], "query": search_query, "rows": limit, "profile": "rich", "qf": ["TYPE:IMAGE"]}
        response = await client.get("https://api.europeana.eu/record/v2/search.json", params=params, timeout=12.0)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("items", []):
            img = None
            if item.get("edmIsShownBy"):
                img = item["edmIsShownBy"][0] if isinstance(item["edmIsShownBy"], list) else item["edmIsShownBy"]
            elif item.get("edmPreview"):
                img = item["edmPreview"][0] if isinstance(item["edmPreview"], list) else item["edmPreview"]
            item_epoch = ""
            if item.get("year"):
                years = item["year"] if isinstance(item["year"], list) else [item["year"]]
                item_epoch = ", ".join(str(y) for y in years[:3])
            item_material = ""
            for field in ["dcType", "dcFormat"]:
                val = item.get(field)
                if val:
                    item_material = val[0] if isinstance(val, list) and val else str(val) if val else ""
                    if item_material: break
            results.append({
                "id": item.get("id", ""),
                "title": item.get("title", ["Unbekannt"])[0] if isinstance(item.get("title"), list) else item.get("title", "Unbekannt"),
                "image_url": img, "source": "Europeana", "source_url": item.get("guid", ""),
                "museum": item.get("dataProvider", [""])[0] if isinstance(item.get("dataProvider"), list) else item.get("dataProvider", ""),
                "epoch": item_epoch, "material": item_material,
            })
        ms = (time.time() - t0) * 1000
        search_cache.set(results, *cache_key)
        record_api_health("europeana", "ok", ms, len(results))
        return {"results": results, "source": "europeana", "status": "ok", "count": len(results)}
    except Exception as e:
        ms = (time.time() - t0) * 1000
        record_api_health("europeana", f"error: {type(e).__name__}", ms)
        logger.warning(f"Europeana error: {e}")
        return {"results": [], "source": "europeana", "status": f"error: {type(e).__name__}"}

async def search_met(client, query, limit=20, epoch=None, material=None):
    cache_key = ("met", query, limit, epoch, material)
    cached = search_cache.get(*cache_key)
    if cached is not None:
        return {"results": cached, "source": "met", "status": "cached"}
    t0 = time.time()
    try:
        params = {"q": query, "hasImages": "true"}
        if epoch and epoch in MET_DEPARTMENT_MAP and MET_DEPARTMENT_MAP[epoch]:
            params["departmentId"] = MET_DEPARTMENT_MAP[epoch]
        if material: params["medium"] = material
        search_response = await client.get("https://collectionapi.metmuseum.org/public/collection/v1/search", params=params, timeout=10.0)
        search_response.raise_for_status()
        search_data = search_response.json()
        object_ids = search_data.get("objectIDs", [])
        if not object_ids:
            search_cache.set([], *cache_key)
            ms = (time.time() - t0) * 1000
            record_api_health("met", "ok", ms, 0)
            return {"results": [], "source": "met", "status": "ok", "count": 0}
        async def fetch_met_object(obj_id):
            try:
                r = await client.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}", timeout=6.0)
                obj = r.json()
                if obj.get("primaryImage"):
                    return {"id": f"met_{obj_id}", "title": obj.get("title", "Unbekannt"),
                        "image_url": obj.get("primaryImageSmall") or obj.get("primaryImage"),
                        "source": "Met Museum", "source_url": obj.get("objectURL", ""),
                        "museum": "Metropolitan Museum of Art",
                        "epoch": obj.get("objectDate", ""), "material": obj.get("medium", ""),
                        "department": obj.get("department", "")}
            except: pass
            return None
        met_results = await asyncio.gather(*[fetch_met_object(oid) for oid in object_ids[:12]])
        results = [r for r in met_results if r is not None]
        ms = (time.time() - t0) * 1000
        search_cache.set(results, *cache_key)
        record_api_health("met", "ok", ms, len(results))
        return {"results": results, "source": "met", "status": "ok", "count": len(results)}
    except Exception as e:
        ms = (time.time() - t0) * 1000
        record_api_health("met", f"error: {type(e).__name__}", ms)
        logger.warning(f"Met error: {e}")
        return {"results": [], "source": "met", "status": f"error: {type(e).__name__}"}

async def search_va(client, query, limit=20, epoch=None, material=None):
    cache_key = ("va", query, limit, epoch, material)
    cached = search_cache.get(*cache_key)
    if cached is not None:
        return {"results": cached, "source": "victoria_albert", "status": "cached"}
    t0 = time.time()
    try:
        params = {"q": query, "page_size": limit, "images_exist": 1}
        if material and material.lower() in VA_CATEGORY_MAP:
            cat = VA_CATEGORY_MAP[material.lower()]
            if cat: params["q_object_type"] = cat
        response = await client.get("https://api.vam.ac.uk/v2/objects/search", params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("records", []):
            img = item.get("_images", {}).get("_primary_thumbnail")
            if img:
                results.append({"id": f"va_{item.get('systemNumber', '')}", "title": item.get("_primaryTitle", "Unbekannt"),
                    "image_url": img, "source": "V&A Museum",
                    "source_url": f"https://collections.vam.ac.uk/item/{item.get('systemNumber', '')}",
                    "museum": "Victoria & Albert Museum", "epoch": item.get("_primaryDate", ""),
                    "material": ", ".join(item.get("_primaryMaker", {}).get("name", "").split(",")[:2]) if item.get("_primaryMaker") else ""})
        ms = (time.time() - t0) * 1000
        search_cache.set(results, *cache_key)
        record_api_health("victoria_albert", "ok", ms, len(results))
        return {"results": results, "source": "victoria_albert", "status": "ok", "count": len(results)}
    except Exception as e:
        ms = (time.time() - t0) * 1000
        record_api_health("victoria_albert", f"error: {type(e).__name__}", ms)
        logger.warning(f"V&A error: {e}")
        return {"results": [], "source": "victoria_albert", "status": f"error: {type(e).__name__}"}

async def search_rijksmuseum(client, query, limit=20, epoch=None, material=None):
    if not API_KEYS["rijksmuseum"]:
        return {"results": [], "source": "rijksmuseum", "status": "disabled"}
    cache_key = ("rijks", query, limit, epoch, material)
    cached = search_cache.get(*cache_key)
    if cached is not None:
        return {"results": cached, "source": "rijksmuseum", "status": "cached"}
    t0 = time.time()
    try:
        params = {"key": API_KEYS["rijksmuseum"], "q": query, "ps": limit, "imgonly": "true", "format": "json"}
        if material: params["material"] = material
        response = await client.get("https://www.rijksmuseum.nl/api/en/collection", params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("artObjects", []):
            if item.get("webImage", {}).get("url"):
                results.append({"id": item.get("objectNumber", ""), "title": item.get("title", "Unbekannt"),
                    "image_url": item.get("webImage", {}).get("url", ""), "source": "Rijksmuseum",
                    "source_url": item.get("links", {}).get("web", ""), "museum": "Rijksmuseum Amsterdam"})
        ms = (time.time() - t0) * 1000
        search_cache.set(results, *cache_key)
        record_api_health("rijksmuseum", "ok", ms, len(results))
        return {"results": results, "source": "rijksmuseum", "status": "ok", "count": len(results)}
    except Exception as e:
        ms = (time.time() - t0) * 1000
        record_api_health("rijksmuseum", f"error: {type(e).__name__}", ms)
        return {"results": [], "source": "rijksmuseum", "status": f"error: {type(e).__name__}"}

async def search_smithsonian(client, query, limit=20, epoch=None, material=None):
    if not API_KEYS["smithsonian"]:
        return {"results": [], "source": "smithsonian", "status": "disabled"}
    cache_key = ("smith", query, limit, epoch, material)
    cached = search_cache.get(*cache_key)
    if cached is not None:
        return {"results": cached, "source": "smithsonian", "status": "cached"}
    t0 = time.time()
    try:
        search_q = query + " AND online_media_type:Images"
        if material: search_q += f" AND {material}"
        params = {"api_key": API_KEYS["smithsonian"], "q": search_q, "rows": limit}
        response = await client.get("https://api.si.edu/openaccess/api/v1.0/search", params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        results = []
        for row in data.get("response", {}).get("rows", []):
            content = row.get("content", {}); desc = content.get("descriptiveNonRepeating", {})
            img = None
            if content.get("online_media", {}).get("media"):
                media = content["online_media"]["media"]
                if media and len(media) > 0: img = media[0].get("content", "")
            if img:
                results.append({"id": row.get("id", ""),
                    "title": desc.get("title", {}).get("content", "Unbekannt") if isinstance(desc.get("title"), dict) else desc.get("title", "Unbekannt"),
                    "image_url": img, "source": "Smithsonian", "source_url": desc.get("record_link", ""),
                    "museum": desc.get("unit_name", "Smithsonian")})
        ms = (time.time() - t0) * 1000
        search_cache.set(results, *cache_key)
        record_api_health("smithsonian", "ok", ms, len(results))
        return {"results": results, "source": "smithsonian", "status": "ok", "count": len(results)}
    except Exception as e:
        ms = (time.time() - t0) * 1000
        record_api_health("smithsonian", f"error: {type(e).__name__}", ms)
        return {"results": [], "source": "smithsonian", "status": f"error: {type(e).__name__}"}

async def search_harvard(client, query, limit=20, epoch=None, material=None):
    if not API_KEYS["harvard"]:
        return {"results": [], "source": "harvard", "status": "disabled"}
    cache_key = ("harvard", query, limit, epoch, material)
    cached = search_cache.get(*cache_key)
    if cached is not None:
        return {"results": cached, "source": "harvard", "status": "cached"}
    t0 = time.time()
    try:
        params = {"apikey": API_KEYS["harvard"], "q": query, "size": limit, "hasimage": 1}
        if material: params["medium"] = material
        response = await client.get("https://api.harvardartmuseums.org/object", params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("records", []):
            if item.get("primaryimageurl"):
                results.append({"id": f"harvard_{item.get('id', '')}", "title": item.get("title", "Unbekannt"),
                    "image_url": item.get("primaryimageurl", ""), "source": "Harvard Museums",
                    "source_url": item.get("url", ""), "museum": "Harvard Art Museums", "epoch": item.get("dated", "")})
        ms = (time.time() - t0) * 1000
        search_cache.set(results, *cache_key)
        record_api_health("harvard", "ok", ms, len(results))
        return {"results": results, "source": "harvard", "status": "ok", "count": len(results)}
    except Exception as e:
        ms = (time.time() - t0) * 1000
        record_api_health("harvard", f"error: {type(e).__name__}", ms)
        return {"results": [], "source": "harvard", "status": f"error: {type(e).__name__}"}

# ── Cleveland Museum of Art (Public, no key) ──

async def search_cleveland(client, query, limit=20, epoch=None, material=None):
    cache_key = ("cleveland", query, limit, epoch, material)
    cached = search_cache.get(*cache_key)
    if cached is not None:
        return {"results": cached, "source": "cleveland", "status": "cached"}
    t0 = time.time()
    try:
        params = {"q": query, "has_image": 1, "limit": min(limit, 20), "indent": 0}
        response = await client.get("https://openaccess-api.clevelandart.org/api/artworks/", params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("data", []):
            img = None
            if item.get("images") and item["images"].get("web"):
                img = item["images"]["web"].get("url")
            if not img:
                continue
            results.append({"id": f"cma_{item.get('id', '')}", "title": item.get("title", "Unbekannt"),
                "image_url": img, "source": "Cleveland Museum",
                "source_url": item.get("url", f"https://clevelandart.org/art/{item.get('accession_number', '')}"),
                "museum": "Cleveland Museum of Art",
                "epoch": item.get("creation_date", ""),
                "material": item.get("technique", "") or item.get("type", "")})
        ms = (time.time() - t0) * 1000
        search_cache.set(results, *cache_key)
        record_api_health("cleveland", "ok", ms, len(results))
        return {"results": results, "source": "cleveland", "status": "ok", "count": len(results)}
    except Exception as e:
        ms = (time.time() - t0) * 1000
        record_api_health("cleveland", f"error: {type(e).__name__}", ms)
        logger.warning(f"Cleveland error: {e}")
        return {"results": [], "source": "cleveland", "status": f"error: {type(e).__name__}"}

# ── Art Institute of Chicago (Public, no key) ──

async def search_chicago(client, query, limit=20, epoch=None, material=None):
    cache_key = ("chicago", query, limit, epoch, material)
    cached = search_cache.get(*cache_key)
    if cached is not None:
        return {"results": cached, "source": "chicago", "status": "cached"}
    t0 = time.time()
    try:
        params = {"q": query, "limit": min(limit, 20),
            "fields": "id,title,image_id,date_display,medium_display,artist_display,artwork_type_title",
            "query[term][is_public_domain]": "true"}
        response = await client.get("https://api.artic.edu/api/v1/artworks/search", params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        iiif_base = data.get("config", {}).get("iiif_url", "https://www.artic.edu/iiif/2")
        results = []
        for item in data.get("data", []):
            img_id = item.get("image_id")
            if not img_id:
                continue
            img_url = f"{iiif_base}/{img_id}/full/843,/0/default.jpg"
            results.append({"id": f"aic_{item.get('id', '')}", "title": item.get("title", "Unbekannt"),
                "image_url": img_url, "source": "Art Institute Chicago",
                "source_url": f"https://www.artic.edu/artworks/{item.get('id', '')}",
                "museum": "Art Institute of Chicago",
                "epoch": item.get("date_display", ""),
                "material": item.get("medium_display", "")})
        ms = (time.time() - t0) * 1000
        search_cache.set(results, *cache_key)
        record_api_health("chicago", "ok", ms, len(results))
        return {"results": results, "source": "chicago", "status": "ok", "count": len(results)}
    except Exception as e:
        ms = (time.time() - t0) * 1000
        record_api_health("chicago", f"error: {type(e).__name__}", ms)
        logger.warning(f"Chicago AIC error: {e}")
        return {"results": [], "source": "chicago", "status": f"error: {type(e).__name__}"}

# ── Cooper Hewitt, Smithsonian Design Museum (API key optional) ──

async def search_cooper_hewitt(client, query, limit=20, epoch=None, material=None):
    if not API_KEYS["cooper_hewitt"]:
        return {"results": [], "source": "cooper_hewitt", "status": "disabled"}
    cache_key = ("cooper_hewitt", query, limit, epoch, material)
    cached = search_cache.get(*cache_key)
    if cached is not None:
        return {"results": cached, "source": "cooper_hewitt", "status": "cached"}
    t0 = time.time()
    try:
        params = {"method": "cooperhewitt.search.objects", "access_token": API_KEYS["cooper_hewitt"],
            "query": query, "has_images": 1, "per_page": min(limit, 20), "page": 1}
        response = await client.get("https://api.collection.cooperhewitt.org/rest/", params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("objects", []):
            imgs = item.get("images", [])
            img = None
            if imgs:
                for img_entry in imgs:
                    for sz in ["n", "z", "b", "sq"]:
                        if img_entry.get(sz) and img_entry[sz].get("url"):
                            img = img_entry[sz]["url"]; break
                    if img: break
            if not img:
                continue
            results.append({"id": f"ch_{item.get('id', '')}", "title": item.get("title", "Unbekannt"),
                "image_url": img, "source": "Cooper Hewitt",
                "source_url": item.get("url", ""),
                "museum": "Cooper Hewitt, Smithsonian Design Museum",
                "epoch": item.get("date", ""),
                "material": item.get("medium", "")})
        ms = (time.time() - t0) * 1000
        search_cache.set(results, *cache_key)
        record_api_health("cooper_hewitt", "ok", ms, len(results))
        return {"results": results, "source": "cooper_hewitt", "status": "ok", "count": len(results)}
    except Exception as e:
        ms = (time.time() - t0) * 1000
        record_api_health("cooper_hewitt", f"error: {type(e).__name__}", ms)
        logger.warning(f"Cooper Hewitt error: {e}")
        return {"results": [], "source": "cooper_hewitt", "status": f"error: {type(e).__name__}"}

# ── Walters Art Museum (API v1 retired 2023, awaiting v2) ──

async def search_walters(client, query, limit=20, epoch=None, material=None):
    # Walters API v1 was retired in late 2023. No v2 available yet.
    # See: https://github.com/WaltersArtMuseum/walters-api
    # Keeping function for forward-compatibility when v2 launches.
    record_api_health("walters", "offline", 0, 0)
    return {"results": [], "source": "walters", "status": "offline", "message": "API v1 retired 2023, awaiting v2"}

# ── Yale University Art Gallery via LUX API (Public, no key) ──

async def search_yale_lux(client, query, limit=20, epoch=None, material=None):
    cache_key = ("yale_lux", query, limit, epoch, material)
    cached = search_cache.get(*cache_key)
    if cached is not None:
        return {"results": cached, "source": "yale", "status": "cached"}
    t0 = time.time()
    try:
        import urllib.parse
        lux_query = {"AND": [{"text": query}, {"hasDigitalImage": 1}]}
        q_json = json.dumps(lux_query)
        search_url = f"https://lux.collections.yale.edu/api/search/item?q={urllib.parse.quote(q_json)}&page=1&pageLength={min(limit, 20)}"
        response = await client.get(search_url, timeout=12.0, headers={"Accept": "application/json"})
        response.raise_for_status()
        data = response.json()
        item_uris = [item.get("id", "") for item in data.get("orderedItems", []) if item.get("id")][:10]
        if not item_uris:
            ms = (time.time() - t0) * 1000
            search_cache.set([], *cache_key)
            record_api_health("yale", "ok", ms, 0)
            return {"results": [], "source": "yale", "status": "ok", "count": 0}
        async def fetch_lux_item(uri):
            try:
                r = await client.get(uri, timeout=6.0, headers={"Accept": "application/json"})
                if r.status_code != 200:
                    return None
                obj = r.json()
                title = "Unbekannt"
                if obj.get("_label"):
                    title = obj["_label"]
                elif obj.get("identified_by"):
                    for ident in obj["identified_by"]:
                        if ident.get("type") == "Name" and ident.get("content"):
                            title = ident["content"]; break
                img = None
                if obj.get("representation"):
                    for rep in obj["representation"]:
                        if rep.get("type") == "VisualItem" and rep.get("digitally_shown_by"):
                            for dig in rep["digitally_shown_by"]:
                                if dig.get("access_point"):
                                    for ap in dig["access_point"]:
                                        if ap.get("id") and ("iiif" in ap["id"] or ap["id"].endswith((".jpg", ".jpeg", ".png"))):
                                            img = ap["id"]; break
                                if img: break
                        if img: break
                if not img and obj.get("subject_of"):
                    for sub in obj.get("subject_of", []):
                        if sub.get("digitally_carried_by"):
                            for dig in sub["digitally_carried_by"]:
                                if dig.get("access_point"):
                                    for ap in dig["access_point"]:
                                        if ap.get("id"):
                                            img = ap["id"]; break
                if not img:
                    return None
                epoch_str = ""
                if obj.get("produced_by") and obj["produced_by"].get("timespan"):
                    ts = obj["produced_by"]["timespan"]
                    epoch_str = ts.get("_label", ts.get("identified_by", [{}])[0].get("content", "") if ts.get("identified_by") else "")
                obj_id = uri.split("/")[-1] if "/" in uri else uri
                return {"id": f"yale_{obj_id}", "title": title,
                    "image_url": img, "source": "Yale Gallery",
                    "source_url": uri.replace("/data/", "/view/") if "/data/" in uri else uri,
                    "museum": "Yale University Art Gallery", "epoch": epoch_str}
            except:
                return None
        lux_results = await asyncio.gather(*[fetch_lux_item(uri) for uri in item_uris])
        results = [r for r in lux_results if r is not None]
        ms = (time.time() - t0) * 1000
        search_cache.set(results, *cache_key)
        record_api_health("yale", "ok", ms, len(results))
        return {"results": results, "source": "yale", "status": "ok", "count": len(results)}
    except Exception as e:
        ms = (time.time() - t0) * 1000
        record_api_health("yale", f"error: {type(e).__name__}", ms)
        logger.warning(f"Yale LUX error: {e}")
        return {"results": [], "source": "yale", "status": f"error: {type(e).__name__}"}

# ── Musée du Louvre (No public REST search API — JSON only per-object via ark ID) ──
# The Louvre provides JSON per-object at collections.louvre.fr/ark:/53355/{id}.json
# but has NO search endpoint. We use Europeana's Louvre-filtered results as proxy.

async def search_louvre(client, query, limit=20, epoch=None, material=None):
    if not API_KEYS.get("europeana"):
        return {"results": [], "source": "louvre", "status": "disabled", "message": "Needs Europeana key for Louvre proxy search"}
    cache_key = ("louvre_proxy", query, limit, epoch, material)
    cached = search_cache.get(*cache_key)
    if cached is not None:
        return {"results": cached, "source": "louvre", "status": "cached"}
    t0 = time.time()
    try:
        search_q = f"{query} Louvre"
        params = {
            "wskey": API_KEYS["europeana"],
            "query": search_q,
            "rows": min(limit, 15),
            "profile": "rich",
            "qf": ["TYPE:IMAGE", "DATA_PROVIDER:\"Musée du Louvre\""]
        }
        response = await client.get("https://api.europeana.eu/record/v2/search.json", params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("items", []):
            img = None
            if item.get("edmIsShownBy"):
                img = item["edmIsShownBy"][0] if isinstance(item["edmIsShownBy"], list) else item["edmIsShownBy"]
            elif item.get("edmPreview"):
                img = item["edmPreview"][0] if isinstance(item["edmPreview"], list) else item["edmPreview"]
            if not img:
                continue
            title = item.get("title", ["Unbekannt"])[0] if isinstance(item.get("title"), list) else item.get("title", "Unbekannt")
            results.append({"id": f"louvre_{item.get('id', '')}", "title": title,
                "image_url": img, "source": "Louvre (via Europeana)",
                "source_url": item.get("guid", ""),
                "museum": "Musée du Louvre",
                "epoch": "", "material": ""})
        ms = (time.time() - t0) * 1000
        search_cache.set(results, *cache_key)
        record_api_health("louvre", "ok", ms, len(results))
        return {"results": results, "source": "louvre", "status": "ok", "count": len(results)}
    except Exception as e:
        ms = (time.time() - t0) * 1000
        record_api_health("louvre", f"error: {type(e).__name__}", ms)
        logger.warning(f"Louvre proxy error: {e}")
        return {"results": [], "source": "louvre", "status": f"error: {type(e).__name__}"}

# ── National Gallery of Art, Washington D.C. (No public search API — CSV data only on GitHub) ──
# NGA provides open data via GitHub CSV dumps but has no REST search endpoint.
# We use Europeana's NGA-filtered results as proxy.

async def search_nga(client, query, limit=20, epoch=None, material=None):
    if not API_KEYS.get("europeana"):
        return {"results": [], "source": "nga", "status": "disabled", "message": "Needs Europeana key for NGA proxy search"}
    cache_key = ("nga_proxy", query, limit, epoch, material)
    cached = search_cache.get(*cache_key)
    if cached is not None:
        return {"results": cached, "source": "nga", "status": "cached"}
    t0 = time.time()
    try:
        search_q = f"{query} National Gallery"
        params = {
            "wskey": API_KEYS["europeana"],
            "query": search_q,
            "rows": min(limit, 15),
            "profile": "rich",
            "qf": ["TYPE:IMAGE", "DATA_PROVIDER:\"National Gallery of Art\""]
        }
        response = await client.get("https://api.europeana.eu/record/v2/search.json", params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("items", []):
            img = None
            if item.get("edmIsShownBy"):
                img = item["edmIsShownBy"][0] if isinstance(item["edmIsShownBy"], list) else item["edmIsShownBy"]
            elif item.get("edmPreview"):
                img = item["edmPreview"][0] if isinstance(item["edmPreview"], list) else item["edmPreview"]
            if not img:
                continue
            title = item.get("title", ["Unbekannt"])[0] if isinstance(item.get("title"), list) else item.get("title", "Unbekannt")
            results.append({"id": f"nga_{item.get('id', '')}", "title": title,
                "image_url": img, "source": "NGA Washington (via Europeana)",
                "source_url": item.get("guid", ""),
                "museum": "National Gallery of Art",
                "epoch": "", "material": ""})
        ms = (time.time() - t0) * 1000
        search_cache.set(results, *cache_key)
        record_api_health("nga", "ok", ms, len(results))
        return {"results": results, "source": "nga", "status": "ok", "count": len(results)}
    except Exception as e:
        ms = (time.time() - t0) * 1000
        record_api_health("nga", f"error: {type(e).__name__}", ms)
        logger.warning(f"NGA proxy error: {e}")
        return {"results": [], "source": "nga", "status": f"error: {type(e).__name__}"}

def _run_all_museum_searches(client, query, limit, epoch, material):
    return [
        search_europeana(client, query, limit, epoch=epoch, material=material),
        search_met(client, query, limit, epoch=epoch, material=material),
        search_va(client, query, limit, epoch=epoch, material=material),
        search_rijksmuseum(client, query, limit, epoch=epoch, material=material),
        search_smithsonian(client, query, limit, epoch=epoch, material=material),
        search_harvard(client, query, limit, epoch=epoch, material=material),
        search_cleveland(client, query, limit, epoch=epoch, material=material),
        search_chicago(client, query, limit, epoch=epoch, material=material),
        search_cooper_hewitt(client, query, limit, epoch=epoch, material=material),
        search_walters(client, query, limit, epoch=epoch, material=material),
        search_yale_lux(client, query, limit, epoch=epoch, material=material),
        search_louvre(client, query, limit, epoch=epoch, material=material),
        search_nga(client, query, limit, epoch=epoch, material=material),
    ]

@app.get("/")
async def root():
    enabled = []
    if API_KEYS["europeana"]: enabled.append("europeana")
    enabled.extend(["met", "victoria_albert", "cleveland", "chicago"])  # Always enabled (public)
    if API_KEYS["rijksmuseum"]: enabled.append("rijksmuseum")
    if API_KEYS["smithsonian"]: enabled.append("smithsonian")
    if API_KEYS["harvard"]: enabled.append("harvard")
    if API_KEYS["cooper_hewitt"]: enabled.append("cooper_hewitt")
    enabled.append("yale")  # Public LUX API
    proxy_apis = []  # Use Europeana as proxy (no native search API)
    if API_KEYS["europeana"]:
        proxy_apis.extend(["louvre", "nga"])
    offline = ["walters"]  # API v1 closed 2023
    return {"name": "ArchaeoFinder API", "version": "4.3.0", "status": "online",
        "features": ["multi_museum_search", "user_auth", "save_finds", "smart_filters", "deduplication", "caching", "api_health_tracking", "live_status", "image_proxy"],
        "enabled_apis": enabled, "proxy_apis": proxy_apis, "offline_apis": offline,
        "total_sources": len(enabled) + len(proxy_apis)}

# =============================================================================
# API STATUS ENDPOINT — Live health checks
# =============================================================================

API_INFO = {
    "europeana": {"name": "Europeana", "description": "Europaeische digitale Bibliothek mit 50+ Mio. Objekten aus 4.000+ Institutionen",
        "url": "https://www.europeana.eu", "api_docs": "https://pro.europeana.eu/page/apis",
        "key_required": True, "free": True, "collections": "Kunst, Archaeologie, Fotografie, Buecher, Musik", "region": "Europa"},
    "met": {"name": "Metropolitan Museum of Art", "description": "Groesstes Kunstmuseum der USA mit 500.000+ Objekten",
        "url": "https://www.metmuseum.org", "api_docs": "https://metmuseum.github.io/",
        "key_required": False, "free": True, "collections": "Antike, Waffen, Ruestungen, Mittelalter, Aegypten, Asien", "region": "New York, USA"},
    "victoria_albert": {"name": "Victoria & Albert Museum", "description": "Weltweit groesstes Museum fuer Kunst, Design und Performance",
        "url": "https://www.vam.ac.uk", "api_docs": "https://developers.vam.ac.uk/",
        "key_required": False, "free": True, "collections": "Keramik, Schmuck, Textilien, Skulptur, Metall", "region": "London, UK"},
    "rijksmuseum": {"name": "Rijksmuseum Amsterdam", "description": "Niederlaendisches Nationalmuseum mit 1 Mio.+ Objekten",
        "url": "https://www.rijksmuseum.nl", "api_docs": "https://data.rijksmuseum.nl/object-metadata/api/",
        "key_required": True, "free": True, "collections": "Niederlaendische Kunst, Geschichte, Archaeologie", "region": "Amsterdam, NL"},
    "smithsonian": {"name": "Smithsonian Institution", "description": "Weltweit groesster Museumsverbund mit 155 Mio.+ Objekten",
        "url": "https://www.si.edu", "api_docs": "https://api.data.gov/signup/",
        "key_required": True, "free": True, "collections": "Naturgeschichte, Amerikanische Geschichte, Luft- und Raumfahrt", "region": "Washington D.C., USA"},
    "harvard": {"name": "Harvard Art Museums", "description": "Drei Museen (Fogg, Busch-Reisinger, Sackler) mit 250.000+ Objekten",
        "url": "https://harvardartmuseums.org", "api_docs": "https://harvardartmuseums.org/collections/api",
        "key_required": True, "free": True, "collections": "Antike Kunst, Archaeologie, Asiatische Kunst, Europaeische Kunst", "region": "Cambridge, USA"},
    "cleveland": {"name": "Cleveland Museum of Art", "description": "Diverse archaeologische Artefakte und Kunstwerke, 61.000+ Open Access Objekte",
        "url": "https://www.clevelandart.org", "api_docs": "https://openaccess-api.clevelandart.org/",
        "key_required": False, "free": True, "collections": "Antike, Archaeologie, Asiatische Kunst, Europaeische Kunst", "region": "Cleveland, USA"},
    "chicago": {"name": "Art Institute of Chicago", "description": "Umfassende Sammlung mit sehr guten Bilddaten, 300.000+ Objekte",
        "url": "https://www.artic.edu", "api_docs": "https://api.artic.edu/docs/",
        "key_required": False, "free": True, "collections": "Kunst aller Epochen, Textilien, Fotografie, Design", "region": "Chicago, USA"},
    "cooper_hewitt": {"name": "Cooper Hewitt, Smithsonian", "description": "Smithsonian Designmuseum mit Fokus auf Designgeschichte",
        "url": "https://www.cooperhewitt.org", "api_docs": "https://collection.cooperhewitt.org/api/",
        "key_required": True, "free": True, "collections": "Designgeschichte, Textilien, Keramik, Grafik", "region": "New York, USA"},
    "walters": {"name": "Walters Art Museum", "description": "Spezialisiert auf antike Kunst und Manuskripte. API v1 geschlossen 2023, wartet auf v2.",
        "url": "https://art.thewalters.org", "api_docs": "https://github.com/WaltersArtMuseum/walters-api",
        "key_required": False, "free": True, "collections": "Antike Kunst, Manuskripte, Aegyptische Kunst, Mittelalter", "region": "Baltimore, USA",
        "api_status": "offline"},
    "yale": {"name": "Yale University Art Gallery", "description": "Bedeutende numismatische und antike Bestaende via LUX Discovery, 300.000+ Objekte",
        "url": "https://artgallery.yale.edu", "api_docs": "https://lux.collections.yale.edu/",
        "key_required": False, "free": True, "collections": "Numismatik, Antike, Archaeologie, Asiatische Kunst", "region": "New Haven, USA"},
    "louvre": {"name": "Musée du Louvre", "description": "500.000+ Objekte. Kein REST-Suchendpoint, Suche via Europeana-Proxy.",
        "url": "https://collections.louvre.fr", "api_docs": "https://collections.louvre.fr/en/page/documentationJSON",
        "key_required": False, "free": True, "collections": "Antiken, Aegypten, Orient, Skulpturen, Gemaelde", "region": "Paris, Frankreich",
        "api_status": "proxy"},
    "nga": {"name": "National Gallery of Art", "description": "130.000+ Objekte. Kein REST-Suchendpoint, Suche via Europeana-Proxy.",
        "url": "https://www.nga.gov", "api_docs": "https://github.com/NationalGalleryOfArt/opendata",
        "key_required": False, "free": True, "collections": "Bildende Kunst, Medaillen, Skulpturen, Grafik", "region": "Washington D.C., USA",
        "api_status": "proxy"},
}

@app.get("/api/status")
async def api_status_endpoint():
    """Live API health check — pings all museum APIs with a test query."""
    client = await get_http_client()

    async def check_api(name, url, params, parse_total):
        t0 = time.time()
        try:
            resp = await client.get(url, params=params, timeout=8.0)
            ms = round((time.time() - t0) * 1000)
            if resp.status_code == 200:
                data = resp.json()
                total = parse_total(data)
                return name, {"configured": True, "status": "online", "response_time_ms": ms, "total_records": total}
            return name, {"configured": True, "status": "error", "http_code": resp.status_code, "response_time_ms": ms}
        except Exception as e:
            ms = round((time.time() - t0) * 1000)
            return name, {"configured": True, "status": "error", "error": type(e).__name__, "response_time_ms": ms}

    tasks = []
    async def not_configured(name):
        return name, {"configured": False, "status": "not_configured", "message": "API Key nicht gesetzt"}

    if API_KEYS["europeana"]:
        tasks.append(check_api("europeana", "https://api.europeana.eu/record/v2/search.json",
            {"wskey": API_KEYS["europeana"], "query": "archaeology", "rows": 1, "profile": "minimal"},
            lambda d: d.get("totalResults", 0)))
    else:
        tasks.append(not_configured("europeana"))

    tasks.append(check_api("met", "https://collectionapi.metmuseum.org/public/collection/v1/search",
        {"q": "stone axe", "hasImages": "true"}, lambda d: d.get("total", 0)))

    tasks.append(check_api("victoria_albert", "https://api.vam.ac.uk/v2/objects/search",
        {"q": "bronze", "page_size": 1}, lambda d: d.get("info", {}).get("record_count", 0)))

    if API_KEYS["rijksmuseum"]:
        tasks.append(check_api("rijksmuseum", "https://www.rijksmuseum.nl/api/en/collection",
            {"key": API_KEYS["rijksmuseum"], "q": "archaeology", "ps": 1, "format": "json"},
            lambda d: d.get("count", 0)))

    if API_KEYS["smithsonian"]:
        tasks.append(check_api("smithsonian", "https://api.si.edu/openaccess/api/v1.0/search",
            {"api_key": API_KEYS["smithsonian"], "q": "archaeology", "rows": 1},
            lambda d: d.get("response", {}).get("rowCount", 0)))

    if API_KEYS["harvard"]:
        tasks.append(check_api("harvard", "https://api.harvardartmuseums.org/object",
            {"apikey": API_KEYS["harvard"], "q": "ancient", "size": 1},
            lambda d: d.get("info", {}).get("totalrecords", 0)))

    # Public APIs — always checked
    tasks.append(check_api("cleveland", "https://openaccess-api.clevelandart.org/api/artworks/",
        {"q": "archaeology", "has_image": 1, "limit": 1, "indent": 0},
        lambda d: d.get("info", {}).get("total", 0)))

    tasks.append(check_api("chicago", "https://api.artic.edu/api/v1/artworks/search",
        {"q": "ancient artifact", "limit": 1, "fields": "id,title"},
        lambda d: d.get("pagination", {}).get("total", 0)))

    if API_KEYS["cooper_hewitt"]:
        tasks.append(check_api("cooper_hewitt", "https://api.collection.cooperhewitt.org/rest/",
            {"method": "cooperhewitt.search.objects", "access_token": API_KEYS["cooper_hewitt"], "query": "ceramic", "per_page": 1},
            lambda d: d.get("total", 0)))

    # Walters (API v1 retired 2023)
    async def walters_offline():
        return "walters", {"configured": False, "status": "offline", "message": "API v1 seit 2023 eingestellt, v2 ausstehend"}
    tasks.append(walters_offline())

    # Yale LUX (public)
    import urllib.parse as _up
    tasks.append(check_api("yale", "https://lux.collections.yale.edu/api/search/item",
        {"q": _up.quote(json.dumps({"AND": [{"text": "pottery"}]})), "pageLength": 1},
        lambda d: d.get("partOf", [{}])[0].get("totalItems", 0) if d.get("partOf") else len(d.get("orderedItems", []))))

    # Louvre (proxy via Europeana — no native REST search API)
    if API_KEYS.get("europeana"):
        tasks.append(check_api("louvre", "https://api.europeana.eu/record/v2/search.json",
            {"wskey": API_KEYS["europeana"], "query": "Louvre antiquity", "rows": 1, "profile": "minimal",
             "qf": "DATA_PROVIDER:\"Musée du Louvre\""},
            lambda d: d.get("totalResults", 0)))
    else:
        async def louvre_no_key():
            return "louvre", {"configured": False, "status": "not_configured", "message": "Europeana Key benötigt für Louvre-Proxy"}
        tasks.append(louvre_no_key())

    # NGA (proxy via Europeana — no native REST search API)
    if API_KEYS.get("europeana"):
        tasks.append(check_api("nga", "https://api.europeana.eu/record/v2/search.json",
            {"wskey": API_KEYS["europeana"], "query": "National Gallery sculpture", "rows": 1, "profile": "minimal",
             "qf": "DATA_PROVIDER:\"National Gallery of Art\""},
            lambda d: d.get("totalResults", 0)))
    else:
        async def nga_no_key():
            return "nga", {"configured": False, "status": "not_configured", "message": "Europeana Key benötigt für NGA-Proxy"}
        tasks.append(nga_no_key())

    async def check_supabase():
        if not SUPABASE_URL or not SUPABASE_ANON_KEY:
            return "supabase", {"configured": False, "status": "not_configured"}
        t0 = time.time()
        try:
            resp = await client.get(f"{SUPABASE_URL}/rest/v1/", headers={"apikey": SUPABASE_ANON_KEY}, timeout=8.0)
            ms = round((time.time() - t0) * 1000)
            return "supabase", {"configured": True, "status": "online" if resp.status_code < 400 else "error", "response_time_ms": ms}
        except Exception as e:
            ms = round((time.time() - t0) * 1000)
            return "supabase", {"configured": True, "status": "error", "error": type(e).__name__, "response_time_ms": ms}

    tasks.append(check_supabase())

    check_results = await asyncio.gather(*tasks, return_exceptions=True)

    apis = {}
    for result in check_results:
        if isinstance(result, Exception): continue
        if not isinstance(result, tuple): continue
        key, data = result
        info = API_INFO.get(key, {}).copy()
        info.update(data)
        if key == "supabase":
            info.update({"name": "Supabase", "description": "PostgreSQL Datenbank + Authentifizierung", "url": SUPABASE_URL, "region": "Frankfurt (eu-central-1)"})
        apis[key] = info

    for key in ["rijksmuseum", "smithsonian", "harvard", "cooper_hewitt"]:
        if key not in apis and not API_KEYS.get(key):
            info = API_INFO.get(key, {}).copy()
            info.update({"configured": False, "status": "not_configured", "message": "API Key nicht gesetzt"})
            apis[key] = info

    online_count = sum(1 for v in apis.values() if v.get("status") == "online")

    return {
        "version": "4.3.0",
        "server_time": datetime.utcnow().isoformat(),
        "summary": {"online": online_count, "total": len(apis), "cache_entries": search_cache.size},
        "apis": apis,
        "last_search_health": _api_health,
    }

# =============================================================================
# IMAGE PROXY — Bypass CORS for visual reranking (Phase 2)
# =============================================================================

_image_cache: Dict[str, tuple] = {}  # url -> (content_bytes, content_type, timestamp)
IMAGE_CACHE_TTL = 3600  # 1 hour
IMAGE_CACHE_MAX = 200   # max cached images

@app.get("/api/image-proxy")
async def image_proxy(url: str = Query(..., description="Image URL to proxy")):
    """Proxy museum images to bypass CORS for CLIP embedding computation in browser."""
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL")
    
    # Check cache
    now = time.time()
    if url in _image_cache:
        content, ctype, ts = _image_cache[url]
        if now - ts < IMAGE_CACHE_TTL:
            return Response(content=content, media_type=ctype,
                headers={"Cache-Control": "public, max-age=86400", "X-Cache": "HIT"})
    
    # Fetch image
    try:
        client = await get_http_client()
        resp = await client.get(url, timeout=15.0)
        resp.raise_for_status()
        
        ctype = resp.headers.get("content-type", "image/jpeg")
        if not ctype.startswith("image/"):
            raise HTTPException(status_code=400, detail="URL is not an image")
        
        content = resp.content
        if len(content) > 10_000_000:  # 10MB limit
            raise HTTPException(status_code=413, detail="Image too large")
        
        # Cache it (evict oldest if full)
        if len(_image_cache) >= IMAGE_CACHE_MAX:
            oldest_key = min(_image_cache, key=lambda k: _image_cache[k][2])
            del _image_cache[oldest_key]
        _image_cache[url] = (content, ctype, now)
        
        return Response(content=content, media_type=ctype,
            headers={"Cache-Control": "public, max-age=86400", "X-Cache": "MISS"})
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Image fetch timeout")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Image fetch failed: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Image proxy error: {str(e)}")

@app.get("/api/search")
async def search(
    q: str = Query(..., description="Primary search query"),
    q2: Optional[str] = Query(None, description="Secondary search query"),
    epoch: Optional[str] = Query(None, description="Epoch filter"),
    material: Optional[str] = Query(None, description="Material filter"),
    limit: int = Query(20, ge=1, le=50),
):
    client = await get_http_client()
    primary_tasks = _run_all_museum_searches(client, q, limit, epoch, material)
    if q2:
        secondary_limit = max(limit // 2, 5)
        secondary_tasks = _run_all_museum_searches(client, q2, secondary_limit, epoch, material)
        all_tasks = primary_tasks + secondary_tasks
    else:
        all_tasks = primary_tasks
    all_responses = await asyncio.gather(*all_tasks, return_exceptions=True)
    combined = []; api_status = []; primary_count = len(primary_tasks)
    for i, resp in enumerate(all_responses):
        is_primary = i < primary_count
        if isinstance(resp, Exception):
            api_status.append({"source": f"task_{i}", "status": f"exception: {type(resp).__name__}"}); continue
        if isinstance(resp, dict):
            api_status.append({"source": resp.get("source", "unknown"), "status": resp.get("status", "unknown"),
                "count": resp.get("count", 0) if resp.get("status") not in ("disabled", "cached") else len(resp.get("results", []))})
            for result in resp.get("results", []):
                result["_relevance"] = "primary" if is_primary else "secondary"
                combined.append(result)
    combined = [r for r in combined if r.get("image_url")]
    combined = deduplicate_results(combined)
    combined.sort(key=lambda r: 0 if r.get("_relevance") == "primary" else 1)
    for r in combined: r.pop("_relevance", None)
    return {"query": q, "query_secondary": q2, "filters": {"epoch": epoch, "material": material},
        "total_results": len(combined), "api_status": api_status, "results": combined}

# =============================================================================
# USER FINDS ENDPOINTS
# =============================================================================

@app.get("/api/finds")
async def get_user_finds(authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user: raise HTTPException(status_code=401, detail="Not authenticated")
    client = await get_http_client()
    response = await client.get(f"{SUPABASE_URL}/rest/v1/finds",
        params={"user_id": f"eq.{user['id']}", "order": "created_at.desc"},
        headers={"Authorization": f"Bearer {authorization.replace('Bearer ', '')}", "apikey": SUPABASE_ANON_KEY}, timeout=10.0)
    if response.status_code == 200: return {"finds": response.json()}
    else: raise HTTPException(status_code=response.status_code, detail="Failed to fetch finds")

@app.post("/api/finds")
async def create_find(find: FindCreate, authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user: raise HTTPException(status_code=401, detail="Not authenticated")
    find_data = find.dict()
    find_data["user_id"] = user["id"]
    find_data["created_at"] = datetime.utcnow().isoformat()
    find_data["updated_at"] = datetime.utcnow().isoformat()
    if find_data.get("ai_labels"): find_data["ai_labels"] = json.dumps(find_data["ai_labels"])
    if find_data.get("matched_artifacts"): find_data["matched_artifacts"] = json.dumps(find_data["matched_artifacts"])
    if find_data.get("find_coordinates"): find_data["find_coordinates"] = json.dumps(find_data["find_coordinates"])
    client = await get_http_client()
    response = await client.post(f"{SUPABASE_URL}/rest/v1/finds", json=find_data,
        headers={"Authorization": f"Bearer {authorization.replace('Bearer ', '')}", "apikey": SUPABASE_ANON_KEY,
            "Content-Type": "application/json", "Prefer": "return=representation"}, timeout=10.0)
    if response.status_code == 201: return {"success": True, "find": response.json()[0] if response.json() else None}
    else: raise HTTPException(status_code=response.status_code, detail=f"Failed to create find: {response.text}")

@app.put("/api/finds/{find_id}")
async def update_find(find_id: str, find: FindUpdate, authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user: raise HTTPException(status_code=401, detail="Not authenticated")
    update_data = {k: v for k, v in find.dict().items() if v is not None}
    update_data["updated_at"] = datetime.utcnow().isoformat()
    if update_data.get("ai_labels"): update_data["ai_labels"] = json.dumps(update_data["ai_labels"])
    if update_data.get("matched_artifacts"): update_data["matched_artifacts"] = json.dumps(update_data["matched_artifacts"])
    if update_data.get("find_coordinates"): update_data["find_coordinates"] = json.dumps(update_data["find_coordinates"])
    client = await get_http_client()
    response = await client.patch(f"{SUPABASE_URL}/rest/v1/finds",
        params={"id": f"eq.{find_id}", "user_id": f"eq.{user['id']}"}, json=update_data,
        headers={"Authorization": f"Bearer {authorization.replace('Bearer ', '')}", "apikey": SUPABASE_ANON_KEY,
            "Content-Type": "application/json", "Prefer": "return=representation"}, timeout=10.0)
    if response.status_code == 200: return {"success": True, "find": response.json()[0] if response.json() else None}
    else: raise HTTPException(status_code=response.status_code, detail="Failed to update find")

@app.delete("/api/finds/{find_id}")
async def delete_find(find_id: str, authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user: raise HTTPException(status_code=401, detail="Not authenticated")
    client = await get_http_client()
    response = await client.delete(f"{SUPABASE_URL}/rest/v1/finds",
        params={"id": f"eq.{find_id}", "user_id": f"eq.{user['id']}"},
        headers={"Authorization": f"Bearer {authorization.replace('Bearer ', '')}", "apikey": SUPABASE_ANON_KEY}, timeout=10.0)
    if response.status_code == 204: return {"success": True}
    else: raise HTTPException(status_code=response.status_code, detail="Failed to delete find")

@app.get("/api/profile")
async def get_profile(authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user: raise HTTPException(status_code=401, detail="Not authenticated")
    return {"user": user}

# =============================================================================
# FIBEL FINDER ENDPOINTS (Phase 3 — 768d Vector Search)
# =============================================================================

class FibelSearchRequest(BaseModel):
    image_data: Optional[str] = None       # Base64 encoded image
    embedding: Optional[List[float]] = None # Pre-computed 768d embedding
    threshold: float = 0.35
    count: int = 20
    fibula_type: Optional[str] = None
    epoch: Optional[str] = None

# Lazy-loaded CLIP ViT-L/14 model for server-side embedding
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None

async def _get_clip_model():
    """Lazy-load CLIP ViT-L/14 model. Returns None if not available (low-memory environments)."""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is not None:
        return _clip_model, _clip_preprocess
    try:
        import open_clip
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device)
        model.eval()
        _clip_model = model
        _clip_preprocess = preprocess
        logger.info(f"CLIP ViT-L/14 loaded on {device}")
        return model, preprocess
    except Exception as e:
        logger.warning(f"CLIP model not available: {e}")
        return None, None

async def _create_embedding_from_image(image_data: str) -> Optional[List[float]]:
    """Create 768d CLIP embedding from base64 image data."""
    try:
        model, preprocess = await _get_clip_model()
        if model is None:
            return None
        import torch
        from PIL import Image
        import io
        import base64
        
        # Decode base64
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Preprocess and embed
        img_tensor = preprocess(img).unsqueeze(0)
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            embedding = model.encode_image(img_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding[0].cpu().tolist()
    except Exception as e:
        logger.error(f"Embedding creation failed: {e}")
        return None

@app.get("/api/fibel/stats")
async def fibel_stats():
    """Get statistics about indexed fibulae."""
    try:
        client = await get_http_client()
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/fibula_stats",
            params={"select": "*"},
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"},
            timeout=10.0
        )
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                return data[0]
        # Table doesn't exist yet or empty
        return {
            "total_fibulae": 0,
            "source_count": 0,
            "museum_count": 0,
            "type_count": 0,
            "epoch_count": 0,
            "first_indexed": None,
            "last_indexed": None,
            "status": "no_data",
            "message": "Fibel-Datenbank noch nicht befüllt. Bitte GPU-Pipeline ausführen."
        }
    except Exception as e:
        logger.warning(f"Fibel stats error: {e}")
        return {
            "total_fibulae": 0,
            "source_count": 0,
            "museum_count": 0,
            "status": "error",
            "message": str(e)
        }

@app.post("/api/fibel/search")
async def fibel_search(req: FibelSearchRequest):
    """Search for similar fibulae using 768d CLIP vector similarity."""
    embedding = None
    
    # Option 1: Pre-computed embedding from client
    if req.embedding and len(req.embedding) == 768:
        embedding = req.embedding
    
    # Option 2: Create embedding from image on server
    elif req.image_data:
        embedding = await _create_embedding_from_image(req.image_data)
        if embedding is None:
            raise HTTPException(
                status_code=503,
                detail="CLIP ViT-L/14 Modell nicht verfügbar auf diesem Server. "
                       "Bitte Embedding client-seitig erstellen oder Backend auf GPU-fähigem Server betreiben."
            )
    else:
        raise HTTPException(status_code=400, detail="Entweder image_data oder embedding (768d) erforderlich.")
    
    # Query Supabase pgvector via RPC
    try:
        client = await get_http_client()
        
        rpc_body = {
            "query_embedding": embedding,
            "match_threshold": req.threshold,
            "match_count": req.count
        }
        
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/search_fibulae",
            json=rpc_body,
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                "Content-Type": "application/json"
            },
            timeout=15.0
        )
        
        if response.status_code == 200:
            results = response.json()
            
            # Apply optional filters
            if req.fibula_type:
                results = [r for r in results if r.get("fibula_type") and req.fibula_type.lower() in r["fibula_type"].lower()]
            if req.epoch:
                results = [r for r in results if r.get("epoch") and req.epoch.lower() in r["epoch"].lower()]
            
            return {
                "query": "vector_similarity",
                "embedding_dimensions": 768,
                "threshold": req.threshold,
                "total_results": len(results),
                "results": results
            }
        else:
            error_detail = response.text
            logger.error(f"Supabase RPC error: {response.status_code} - {error_detail}")
            raise HTTPException(
                status_code=502,
                detail=f"Vektordatenbank-Fehler: {error_detail}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fibel search error: {e}")
        raise HTTPException(status_code=500, detail=f"Suchfehler: {str(e)}")

@app.get("/api/fibel/sources")
async def fibel_sources():
    """Get per-source statistics for indexed fibulae."""
    try:
        client = await get_http_client()
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/fibula_source_stats",
            params={"select": "*"},
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"},
            timeout=10.0
        )
        if response.status_code == 200:
            return {"sources": response.json()}
        return {"sources": []}
    except Exception as e:
        return {"sources": [], "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
