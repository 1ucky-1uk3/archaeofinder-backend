from fastapi import FastAPI, Query, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI(title="ArchaeoFinder API", version="3.2.0")

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
# IN-MEMORY CACHE (TTL-based)
# =============================================================================

class SimpleCache:
    def __init__(self, ttl: int = 300):  # 5 min default
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
        # Evict old entries if cache grows too large
        if len(self._cache) > 500:
            now = time.time()
            expired = [k for k, (_, ts) in self._cache.items() if now - ts > self._ttl]
            for k in expired:
                del self._cache[k]

search_cache = SimpleCache(ttl=300)

# =============================================================================
# FILTER MAPPINGS
# =============================================================================

MET_DEPARTMENT_MAP = {
    "prehistoric": None, "neolithic": None, "bronze_age": None, "iron_age": None,
    "greek": 13, "roman": 13, "egyptian": 10, "medieval": 17, "islamic": 14, "asian": 6,
}

EUROPEANA_EPOCH_FILTERS = {
    "prehistoric": "archaeology prehistory",
    "neolithic": "neolithic stone age",
    "bronze_age": "bronze age",
    "iron_age": "iron age",
    "greek": "ancient greek",
    "roman": "roman empire",
    "egyptian": "ancient egypt",
    "medieval": "medieval",
    "viking": "viking norse",
}

VA_CATEGORY_MAP = {
    "ceramic": "Ceramics", "pottery": "Ceramics",
    "glass": "Glass",
    "bronze": "Metalwork", "iron": "Metalwork", "gold": "Metalwork", "silver": "Metalwork",
    "coin": "Coins & Medals",
    "jewelry": "Jewellery",
    "bone": "Sculpture",
    "stone": "Sculpture",
    "flint": None,  # V&A has very little flint — skip filter
}

# =============================================================================
# DEDUPLICATION (optimized: hash-first, then similarity for edge cases)
# =============================================================================

def _normalize_title(title: str) -> str:
    """Normalize title for fast comparison."""
    if not title:
        return ""
    t = title.lower().strip()
    # Remove common prefixes/suffixes
    for prefix in ["a ", "an ", "the ", "fragment of ", "part of "]:
        if t.startswith(prefix):
            t = t[len(prefix):]
    return t

def _title_hash(title: str) -> str:
    """Create normalized hash for quick exact-match dedup."""
    n = _normalize_title(title)
    # Remove all non-alphanumeric for fuzzy matching
    cleaned = "".join(c for c in n if c.isalnum())
    return cleaned[:60]  # Cap length

def _img_domain(url: str) -> str:
    """Extract image URL fingerprint (domain + path, ignore params)."""
    if not url:
        return ""
    return url.split("?")[0].split("#")[0].lower()

def deduplicate_results(results: list) -> list:
    if not results or len(results) < 2:
        return results
    
    seen_img = set()
    seen_title_hash = set()
    unique = []
    
    for item in results:
        img_fp = _img_domain(item.get("image_url", ""))
        title_h = _title_hash(item.get("title", ""))
        
        # Skip exact image URL duplicates
        if img_fp and img_fp in seen_img:
            continue
        
        # Skip exact normalized title duplicates
        if title_h and len(title_h) > 5 and title_h in seen_title_hash:
            continue
        
        unique.append(item)
        if img_fp:
            seen_img.add(img_fp)
        if title_h and len(title_h) > 5:
            seen_title_hash.add(title_h)
    
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
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.replace("Bearer ", "")
    client = await get_http_client()
    try:
        response = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_ANON_KEY},
            timeout=10.0
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

async def require_auth(authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

# =============================================================================
# MUSEUM API FUNCTIONS (optimized)
# =============================================================================

async def search_europeana(client, query, limit=20, epoch=None, material=None):
    if not API_KEYS["europeana"]:
        return {"results": [], "source": "europeana", "status": "disabled"}
    
    cache_key = ("europeana", query, limit, epoch, material)
    cached = search_cache.get(*cache_key)
    if cached is not None:
        return {"results": cached, "source": "europeana", "status": "cached"}
    
    try:
        search_query = query
        if epoch and epoch in EUROPEANA_EPOCH_FILTERS:
            search_query = f"{query} {EUROPEANA_EPOCH_FILTERS[epoch]}"
        if material:
            search_query = f"{search_query} {material}"

        # Use multiple qf filters for better results
        qf_filters = ["TYPE:IMAGE"]
        
        params = {
            "wskey": API_KEYS["europeana"],
            "query": search_query,
            "rows": limit,
            "profile": "rich",
            "qf": qf_filters,
        }
        response = await client.get(
            "https://api.europeana.eu/record/v2/search.json",
            params=params, timeout=12.0
        )
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
            
            # Extract material info from dcType or dcFormat
            item_material = ""
            for field in ["dcType", "dcFormat"]:
                val = item.get(field)
                if val:
                    if isinstance(val, list):
                        item_material = val[0] if val else ""
                    else:
                        item_material = str(val)
                    if item_material:
                        break
            
            results.append({
                "id": item.get("id", ""),
                "title": item.get("title", ["Unbekannt"])[0] if isinstance(item.get("title"), list) else item.get("title", "Unbekannt"),
                "image_url": img,
                "source": "Europeana",
                "source_url": item.get("guid", ""),
                "museum": item.get("dataProvider", [""])[0] if isinstance(item.get("dataProvider"), list) else item.get("dataProvider", ""),
                "epoch": item_epoch,
                "material": item_material,
            })
        
        search_cache.set(results, *cache_key)
        return {"results": results, "source": "europeana", "status": "ok", "count": len(results)}
    except Exception as e:
        logger.warning(f"Europeana error: {e}")
        return {"results": [], "source": "europeana", "status": f"error: {type(e).__name__}"}

async def search_met(client, query, limit=20, epoch=None, material=None):
    cache_key = ("met", query, limit, epoch, material)
    cached = search_cache.get(*cache_key)
    if cached is not None:
        return {"results": cached, "source": "met", "status": "cached"}
    
    try:
        params = {"q": query, "hasImages": "true"}
        if epoch and epoch in MET_DEPARTMENT_MAP and MET_DEPARTMENT_MAP[epoch]:
            params["departmentId"] = MET_DEPARTMENT_MAP[epoch]
        if material:
            params["medium"] = material
        
        search_response = await client.get(
            "https://collectionapi.metmuseum.org/public/collection/v1/search",
            params=params, timeout=10.0
        )
        search_response.raise_for_status()
        search_data = search_response.json()
        
        object_ids = search_data.get("objectIDs", [])
        if not object_ids:
            search_cache.set([], *cache_key)
            return {"results": [], "source": "met", "status": "ok", "count": 0}
        
        # Fetch up to 12 objects in parallel
        fetch_ids = object_ids[:12]
        
        async def fetch_met_object(obj_id):
            try:
                r = await client.get(
                    f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}",
                    timeout=6.0
                )
                obj = r.json()
                if obj.get("primaryImage"):
                    return {
                        "id": f"met_{obj_id}",
                        "title": obj.get("title", "Unbekannt"),
                        "image_url": obj.get("primaryImageSmall") or obj.get("primaryImage"),
                        "source": "Met Museum",
                        "source_url": obj.get("objectURL", ""),
                        "museum": "Metropolitan Museum of Art",
                        "epoch": obj.get("objectDate", ""),
                        "material": obj.get("medium", ""),
                        "department": obj.get("department", ""),
                    }
            except:
                pass
            return None
        
        met_results = await asyncio.gather(*[fetch_met_object(oid) for oid in fetch_ids])
        results = [r for r in met_results if r is not None]
        search_cache.set(results, *cache_key)
        return {"results": results, "source": "met", "status": "ok", "count": len(results)}
    except Exception as e:
        logger.warning(f"Met error: {e}")
        return {"results": [], "source": "met", "status": f"error: {type(e).__name__}"}

async def search_va(client, query, limit=20, epoch=None, material=None):
    cache_key = ("va", query, limit, epoch, material)
    cached = search_cache.get(*cache_key)
    if cached is not None:
        return {"results": cached, "source": "victoria_albert", "status": "cached"}
    
    try:
        params = {"q": query, "page_size": limit, "images_exist": 1}
        if material and material.lower() in VA_CATEGORY_MAP:
            cat = VA_CATEGORY_MAP[material.lower()]
            if cat:  # None means skip filter
                params["q_object_type"] = cat
        
        response = await client.get(
            "https://api.vam.ac.uk/v2/objects/search",
            params=params, timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("records", []):
            img = None
            if item.get("_images", {}).get("_primary_thumbnail"):
                img = item["_images"]["_primary_thumbnail"]
            if img:
                results.append({
                    "id": f"va_{item.get('systemNumber', '')}",
                    "title": item.get("_primaryTitle", "Unbekannt"),
                    "image_url": img,
                    "source": "V&A Museum",
                    "source_url": f"https://collections.vam.ac.uk/item/{item.get('systemNumber', '')}",
                    "museum": "Victoria & Albert Museum",
                    "epoch": item.get("_primaryDate", ""),
                    "material": ", ".join(item.get("_primaryMaker", {}).get("name", "").split(",")[:2]) if item.get("_primaryMaker") else "",
                })
        search_cache.set(results, *cache_key)
        return {"results": results, "source": "victoria_albert", "status": "ok", "count": len(results)}
    except Exception as e:
        logger.warning(f"V&A error: {e}")
        return {"results": [], "source": "victoria_albert", "status": f"error: {type(e).__name__}"}

async def search_rijksmuseum(client, query, limit=20, epoch=None, material=None):
    if not API_KEYS["rijksmuseum"]:
        return {"results": [], "source": "rijksmuseum", "status": "disabled"}
    
    cache_key = ("rijks", query, limit, epoch, material)
    cached = search_cache.get(*cache_key)
    if cached is not None:
        return {"results": cached, "source": "rijksmuseum", "status": "cached"}
    
    try:
        params = {
            "key": API_KEYS["rijksmuseum"], "q": query,
            "ps": limit, "imgonly": "true", "format": "json"
        }
        if material:
            params["material"] = material
        
        response = await client.get(
            "https://www.rijksmuseum.nl/api/en/collection",
            params=params, timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("artObjects", []):
            if item.get("webImage", {}).get("url"):
                results.append({
                    "id": item.get("objectNumber", ""),
                    "title": item.get("title", "Unbekannt"),
                    "image_url": item.get("webImage", {}).get("url", ""),
                    "source": "Rijksmuseum",
                    "source_url": item.get("links", {}).get("web", ""),
                    "museum": "Rijksmuseum Amsterdam",
                })
        search_cache.set(results, *cache_key)
        return {"results": results, "source": "rijksmuseum", "status": "ok", "count": len(results)}
    except Exception as e:
        logger.warning(f"Rijksmuseum error: {e}")
        return {"results": [], "source": "rijksmuseum", "status": f"error: {type(e).__name__}"}

async def search_smithsonian(client, query, limit=20, epoch=None, material=None):
    if not API_KEYS["smithsonian"]:
        return {"results": [], "source": "smithsonian", "status": "disabled"}
    
    cache_key = ("smith", query, limit, epoch, material)
    cached = search_cache.get(*cache_key)
    if cached is not None:
        return {"results": cached, "source": "smithsonian", "status": "cached"}
    
    try:
        search_q = query + " AND online_media_type:Images"
        if material:
            search_q += f" AND {material}"
        params = {"api_key": API_KEYS["smithsonian"], "q": search_q, "rows": limit}
        response = await client.get(
            "https://api.si.edu/openaccess/api/v1.0/search",
            params=params, timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for row in data.get("response", {}).get("rows", []):
            content = row.get("content", {})
            desc = content.get("descriptiveNonRepeating", {})
            img = None
            if content.get("online_media", {}).get("media"):
                media = content["online_media"]["media"]
                if media and len(media) > 0:
                    img = media[0].get("content", "")
            if img:
                results.append({
                    "id": row.get("id", ""),
                    "title": desc.get("title", {}).get("content", "Unbekannt") if isinstance(desc.get("title"), dict) else desc.get("title", "Unbekannt"),
                    "image_url": img,
                    "source": "Smithsonian",
                    "source_url": desc.get("record_link", ""),
                    "museum": desc.get("unit_name", "Smithsonian"),
                })
        search_cache.set(results, *cache_key)
        return {"results": results, "source": "smithsonian", "status": "ok", "count": len(results)}
    except Exception as e:
        logger.warning(f"Smithsonian error: {e}")
        return {"results": [], "source": "smithsonian", "status": f"error: {type(e).__name__}"}

async def search_harvard(client, query, limit=20, epoch=None, material=None):
    if not API_KEYS["harvard"]:
        return {"results": [], "source": "harvard", "status": "disabled"}
    
    cache_key = ("harvard", query, limit, epoch, material)
    cached = search_cache.get(*cache_key)
    if cached is not None:
        return {"results": cached, "source": "harvard", "status": "cached"}
    
    try:
        params = {"apikey": API_KEYS["harvard"], "q": query, "size": limit, "hasimage": 1}
        if material:
            params["medium"] = material
        response = await client.get(
            "https://api.harvardartmuseums.org/object",
            params=params, timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("records", []):
            if item.get("primaryimageurl"):
                results.append({
                    "id": f"harvard_{item.get('id', '')}",
                    "title": item.get("title", "Unbekannt"),
                    "image_url": item.get("primaryimageurl", ""),
                    "source": "Harvard Museums",
                    "source_url": item.get("url", ""),
                    "museum": "Harvard Art Museums",
                    "epoch": item.get("dated", ""),
                })
        search_cache.set(results, *cache_key)
        return {"results": results, "source": "harvard", "status": "ok", "count": len(results)}
    except Exception as e:
        logger.warning(f"Harvard error: {e}")
        return {"results": [], "source": "harvard", "status": f"error: {type(e).__name__}"}

# =============================================================================
# SEARCH ENDPOINTS
# =============================================================================

def _run_all_museum_searches(client, query, limit, epoch, material):
    """Create tasks for all museum APIs."""
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
    
    return {
        "name": "ArchaeoFinder API",
        "version": "3.2.0",
        "status": "online",
        "features": ["multi_museum_search", "user_auth", "save_finds", "smart_filters", "deduplication", "caching"],
        "enabled_apis": enabled,
        "total_sources": len(enabled)
    }

@app.get("/api/search")
async def search(
    q: str = Query(..., description="Primary search query"),
    q2: Optional[str] = Query(None, description="Secondary search query"),
    epoch: Optional[str] = Query(None, description="Epoch filter"),
    material: Optional[str] = Query(None, description="Material filter"),
    limit: int = Query(20, ge=1, le=50),
):
    client = await get_http_client()
    
    # Primary search across ALL APIs
    primary_tasks = _run_all_museum_searches(client, q, limit, epoch, material)
    
    # Secondary search across ALL APIs too (with reduced limit)
    if q2:
        secondary_limit = max(limit // 2, 5)
        secondary_tasks = _run_all_museum_searches(client, q2, secondary_limit, epoch, material)
        all_tasks = primary_tasks + secondary_tasks
    else:
        all_tasks = primary_tasks
        secondary_tasks = []
    
    all_responses = await asyncio.gather(*all_tasks, return_exceptions=True)
    
    # Separate primary and secondary, tag results with relevance
    combined = []
    api_status = []
    primary_count = len(primary_tasks)
    
    for i, resp in enumerate(all_responses):
        is_primary = i < primary_count
        
        if isinstance(resp, Exception):
            api_status.append({"source": f"task_{i}", "status": f"exception: {type(resp).__name__}"})
            continue
        
        if isinstance(resp, dict):
            api_status.append({
                "source": resp.get("source", "unknown"),
                "status": resp.get("status", "unknown"),
                "count": resp.get("count", 0) if resp.get("status") not in ("disabled", "cached") else len(resp.get("results", [])),
            })
            for result in resp.get("results", []):
                result["_relevance"] = "primary" if is_primary else "secondary"
                combined.append(result)
    
    # Remove entries without images
    combined = [r for r in combined if r.get("image_url")]
    
    # Deduplicate
    combined = deduplicate_results(combined)
    
    # Sort: primary results first, then secondary
    combined.sort(key=lambda r: 0 if r.get("_relevance") == "primary" else 1)
    
    # Clean internal fields before returning
    for r in combined:
        r.pop("_relevance", None)
    
    return {
        "query": q,
        "query_secondary": q2,
        "filters": {"epoch": epoch, "material": material},
        "total_results": len(combined),
        "api_status": api_status,
        "results": combined
    }

# =============================================================================
# USER FINDS ENDPOINTS
# =============================================================================

@app.get("/api/finds")
async def get_user_finds(authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    client = await get_http_client()
    response = await client.get(
        f"{SUPABASE_URL}/rest/v1/finds",
        params={"user_id": f"eq.{user['id']}", "order": "created_at.desc"},
        headers={
            "Authorization": f"Bearer {authorization.replace('Bearer ', '')}",
            "apikey": SUPABASE_ANON_KEY
        },
        timeout=10.0
    )
    if response.status_code == 200:
        return {"finds": response.json()}
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch finds")

@app.post("/api/finds")
async def create_find(find: FindCreate, authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    find_data = find.dict()
    find_data["user_id"] = user["id"]
    find_data["created_at"] = datetime.utcnow().isoformat()
    find_data["updated_at"] = datetime.utcnow().isoformat()
    
    if find_data.get("ai_labels"):
        find_data["ai_labels"] = json.dumps(find_data["ai_labels"])
    if find_data.get("matched_artifacts"):
        find_data["matched_artifacts"] = json.dumps(find_data["matched_artifacts"])
    if find_data.get("find_coordinates"):
        find_data["find_coordinates"] = json.dumps(find_data["find_coordinates"])
    
    client = await get_http_client()
    response = await client.post(
        f"{SUPABASE_URL}/rest/v1/finds",
        json=find_data,
        headers={
            "Authorization": f"Bearer {authorization.replace('Bearer ', '')}",
            "apikey": SUPABASE_ANON_KEY,
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        },
        timeout=10.0
    )
    if response.status_code == 201:
        return {"success": True, "find": response.json()[0] if response.json() else None}
    else:
        raise HTTPException(status_code=response.status_code, detail=f"Failed to create find: {response.text}")

@app.put("/api/finds/{find_id}")
async def update_find(find_id: str, find: FindUpdate, authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    update_data = {k: v for k, v in find.dict().items() if v is not None}
    update_data["updated_at"] = datetime.utcnow().isoformat()
    
    if update_data.get("ai_labels"):
        update_data["ai_labels"] = json.dumps(update_data["ai_labels"])
    if update_data.get("matched_artifacts"):
        update_data["matched_artifacts"] = json.dumps(update_data["matched_artifacts"])
    if update_data.get("find_coordinates"):
        update_data["find_coordinates"] = json.dumps(update_data["find_coordinates"])
    
    client = await get_http_client()
    response = await client.patch(
        f"{SUPABASE_URL}/rest/v1/finds",
        params={"id": f"eq.{find_id}", "user_id": f"eq.{user['id']}"},
        json=update_data,
        headers={
            "Authorization": f"Bearer {authorization.replace('Bearer ', '')}",
            "apikey": SUPABASE_ANON_KEY,
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        },
        timeout=10.0
    )
    if response.status_code == 200:
        return {"success": True, "find": response.json()[0] if response.json() else None}
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to update find")

@app.delete("/api/finds/{find_id}")
async def delete_find(find_id: str, authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    client = await get_http_client()
    response = await client.delete(
        f"{SUPABASE_URL}/rest/v1/finds",
        params={"id": f"eq.{find_id}", "user_id": f"eq.{user['id']}"},
        headers={
            "Authorization": f"Bearer {authorization.replace('Bearer ', '')}",
            "apikey": SUPABASE_ANON_KEY
        },
        timeout=10.0
    )
    if response.status_code == 204:
        return {"success": True}
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to delete find")

@app.get("/api/profile")
async def get_profile(authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"user": user}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
