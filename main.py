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

app = FastAPI(title="ArchaeoFinder API", version="4.0.0")

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

# =============================================================================
# ENDPOINTS
# =============================================================================

def _run_all_museum_searches(client, query, limit, epoch, material):
    return [
        search_europeana(client, query, limit, epoch=epoch, material=material),
        search_met(client, query, limit, epoch=epoch, material=material),
        search_va(client, query, limit, epoch=epoch, material=material),
        search_rijksmuseum(client, query, limit, epoch=epoch, material=material),
        search_smithsonian(client, query, limit, epoch=epoch, material=material),
        search_harvard(client, query, limit, epoch=epoch, material=material),
    ]

@app.get("/")
async def root():
    enabled = []
    if API_KEYS["europeana"]: enabled.append("europeana")
    enabled.extend(["met", "victoria_albert"])
    if API_KEYS["rijksmuseum"]: enabled.append("rijksmuseum")
    if API_KEYS["smithsonian"]: enabled.append("smithsonian")
    if API_KEYS["harvard"]: enabled.append("harvard")
    return {"name": "ArchaeoFinder API", "version": "4.0.0", "status": "online",
        "features": ["multi_museum_search", "user_auth", "save_finds", "smart_filters", "deduplication", "caching", "api_health_tracking", "live_status", "image_proxy"],
        "enabled_apis": enabled, "total_sources": len(enabled)}

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

    for key in ["rijksmuseum", "smithsonian", "harvard"]:
        if key not in apis and not API_KEYS.get(key):
            info = API_INFO.get(key, {}).copy()
            info.update({"configured": False, "status": "not_configured", "message": "API Key nicht gesetzt"})
            apis[key] = info

    online_count = sum(1 for v in apis.values() if v.get("status") == "online")

    return {
        "version": "3.3.0",
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
