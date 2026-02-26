from fastapi import FastAPI, Query, Header, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx
import asyncio
import os
from datetime import datetime
import json
import math

app = FastAPI(title="ArchaeoFinder API", version="4.2.0")

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

# Tabellen-Mapping: finder_type -> Supabase-Tabellenname
EMBEDDING_TABLES = {
    "fibel": "Fibula_embeddings",
    "muenze": "coin_embeddings",
}

# Suchbegriffe fuer Kategorie-Fallback
FINDER_SEARCH_TERMS = {
    "fibel": {
        "all":                "fibula brooch ancient",
        "Steinzeit":          "prehistoric brooch pin",
        "Bronzezeit":         "bronze age fibula pin",
        "Eisenzeit":          "iron age fibula brooch celtic",
        "Roemisch":           "roman fibula brooch",
        "Voelkerwanderung":   "migration period fibula brooch",
        "Fruehmittelalter":   "early medieval brooch fibula",
        "Hochmittelalter":    "medieval brooch clasp",
        "Spaetmittelalter":   "late medieval brooch fibula",
        "Neuzeit":            "post medieval brooch clasp",
    },
    "muenze": {
        "all":                "ancient coin numismatic",
        "Steinzeit":          "prehistoric token",
        "Bronzezeit":         "bronze age coin",
        "Eisenzeit":          "iron age celtic coin",
        "Roemisch":           "roman coin denarius",
        "Voelkerwanderung":   "migration period coin",
        "Fruehmittelalter":   "early medieval coin denar",
        "Hochmittelalter":    "medieval coin penny",
        "Spaetmittelalter":   "late medieval coin groat",
        "Neuzeit":            "post medieval coin",
    }
}

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

class VisualSearchRequest(BaseModel):
    """Request fuer visuelle Aehnlichkeitssuche"""
    embedding: Optional[List[float]] = None
    finder_type: str = "fibel"               # "fibel" oder "muenze"
    epoch: Optional[str] = None
    material: Optional[str] = None
    museum: Optional[str] = None
    limit: int = 20
    min_similarity: float = 0.5

# =============================================================================
# AUTH HELPER
# =============================================================================

async def get_user_from_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.replace("Bearer ", "")
    async with httpx.AsyncClient() as client:
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
# COSINE SIMILARITY
# =============================================================================

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# =============================================================================
# MUSEUM API FUNCTIONS
# =============================================================================

async def search_europeana(client: httpx.AsyncClient, query: str, limit: int = 20):
    if not API_KEYS["europeana"]:
        return []
    try:
        params = {"wskey": API_KEYS["europeana"], "query": query, "rows": limit, "profile": "rich", "qf": "TYPE:IMAGE"}
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
            results.append({
                "id": item.get("id", ""),
                "title": item.get("title", ["Unbekannt"])[0] if isinstance(item.get("title"), list) else item.get("title", "Unbekannt"),
                "image_url": img, "source": "Europeana", "source_url": item.get("guid", ""),
                "museum": item.get("dataProvider", [""])[0] if isinstance(item.get("dataProvider"), list) else item.get("dataProvider", ""),
            })
        return results
    except Exception as e:
        print(f"Europeana error: {e}")
        return []

async def search_met(client: httpx.AsyncClient, query: str, limit: int = 20):
    try:
        sr = await client.get("https://collectionapi.metmuseum.org/public/collection/v1/search", params={"q": query, "hasImages": "true"}, timeout=10.0)
        sr.raise_for_status()
        ids = sr.json().get("objectIDs", [])[:limit]
        if not ids: return []
        results = []
        for oid in ids[:8]:
            try:
                obj = (await client.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}", timeout=5.0)).json()
                if obj.get("primaryImage"):
                    results.append({"id": f"met_{oid}", "title": obj.get("title", "Unbekannt"),
                        "image_url": obj.get("primaryImageSmall") or obj.get("primaryImage"),
                        "source": "Met Museum", "source_url": obj.get("objectURL", ""),
                        "museum": "Metropolitan Museum of Art", "epoch": obj.get("objectDate", "")})
            except: continue
        return results
    except Exception as e:
        print(f"Met error: {e}"); return []

async def search_va(client: httpx.AsyncClient, query: str, limit: int = 20):
    try:
        response = await client.get("https://api.vam.ac.uk/v2/objects/search", params={"q": query, "page_size": limit, "images_exist": 1}, timeout=10.0)
        response.raise_for_status()
        results = []
        for item in response.json().get("records", []):
            img = item.get("_images", {}).get("_primary_thumbnail")
            if img:
                results.append({"id": f"va_{item.get('systemNumber', '')}", "title": item.get("_primaryTitle", "Unbekannt"),
                    "image_url": img, "source": "V&A Museum",
                    "source_url": f"https://collections.vam.ac.uk/item/{item.get('systemNumber', '')}",
                    "museum": "Victoria & Albert Museum", "epoch": item.get("_primaryDate", "")})
        return results
    except Exception as e:
        print(f"V&A error: {e}"); return []

async def search_rijksmuseum(client: httpx.AsyncClient, query: str, limit: int = 20):
    if not API_KEYS["rijksmuseum"]: return []
    try:
        response = await client.get("https://www.rijksmuseum.nl/api/en/collection",
            params={"key": API_KEYS["rijksmuseum"], "q": query, "ps": limit, "imgonly": "true", "format": "json"}, timeout=10.0)
        response.raise_for_status()
        results = []
        for item in response.json().get("artObjects", []):
            if item.get("webImage", {}).get("url"):
                results.append({"id": item.get("objectNumber", ""), "title": item.get("title", "Unbekannt"),
                    "image_url": item["webImage"]["url"], "source": "Rijksmuseum",
                    "source_url": item.get("links", {}).get("web", ""), "museum": "Rijksmuseum Amsterdam"})
        return results
    except Exception as e:
        print(f"Rijksmuseum error: {e}"); return []

async def search_smithsonian(client: httpx.AsyncClient, query: str, limit: int = 20):
    if not API_KEYS["smithsonian"]: return []
    try:
        response = await client.get("https://api.si.edu/openaccess/api/v1.0/search",
            params={"api_key": API_KEYS["smithsonian"], "q": query + " AND online_media_type:Images", "rows": limit}, timeout=10.0)
        response.raise_for_status()
        results = []
        for row in response.json().get("response", {}).get("rows", []):
            content = row.get("content", {}); desc = content.get("descriptiveNonRepeating", {})
            media_list = content.get("online_media", {}).get("media", [])
            img = media_list[0].get("content", "") if media_list else None
            if img:
                results.append({"id": row.get("id", ""), "title": desc.get("title", {}).get("content", "Unbekannt"),
                    "image_url": img, "source": "Smithsonian", "source_url": desc.get("record_link", ""),
                    "museum": desc.get("unit_name", "Smithsonian")})
        return results
    except Exception as e:
        print(f"Smithsonian error: {e}"); return []

async def search_harvard(client: httpx.AsyncClient, query: str, limit: int = 20):
    if not API_KEYS["harvard"]: return []
    try:
        response = await client.get("https://api.harvardartmuseums.org/object",
            params={"apikey": API_KEYS["harvard"], "q": query, "size": limit, "hasimage": 1}, timeout=10.0)
        response.raise_for_status()
        results = []
        for item in response.json().get("records", []):
            if item.get("primaryimageurl"):
                results.append({"id": f"harvard_{item.get('id', '')}", "title": item.get("title", "Unbekannt"),
                    "image_url": item["primaryimageurl"], "source": "Harvard Museums",
                    "source_url": item.get("url", ""), "museum": "Harvard Art Museums", "epoch": item.get("dated", "")})
        return results
    except Exception as e:
        print(f"Harvard error: {e}"); return []

async def search_all_museums(query: str, limit: int = 20) -> list:
    async with httpx.AsyncClient() as client:
        tasks = [search_europeana(client, query, limit), search_met(client, query, limit),
                 search_va(client, query, limit), search_rijksmuseum(client, query, limit),
                 search_smithsonian(client, query, limit), search_harvard(client, query, limit)]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
    combined = []
    for r in all_results:
        if isinstance(r, list): combined.extend(r)
    return [r for r in combined if r.get("image_url")]

# =============================================================================
# HELPER: Supabase-Tabelle fuer Finder-Typ
# =============================================================================

def get_table_for_finder(finder_type: str) -> str:
    """Gibt den Supabase-Tabellennamen zurueck."""
    table = EMBEDDING_TABLES.get(finder_type)
    if not table:
        raise HTTPException(status_code=400, detail=f"Unbekannter Finder-Typ: {finder_type}. Erlaubt: fibel, muenze")
    return table

# =============================================================================
# ENDPOINTS - Status
# =============================================================================

@app.get("/")
async def root():
    enabled = []
    if API_KEYS["europeana"]: enabled.append("europeana")
    enabled.extend(["met", "victoria_albert"])
    if API_KEYS["rijksmuseum"]: enabled.append("rijksmuseum")
    if API_KEYS["smithsonian"]: enabled.append("smithsonian")
    if API_KEYS["harvard"]: enabled.append("harvard")
    return {
        "name": "ArchaeoFinder API", "version": "4.2.0", "status": "online",
        "features": ["multi_museum_search", "visual_similarity_search", "fibel_finder", "muenz_finder", "user_auth", "save_finds"],
        "enabled_apis": enabled, "total_sources": len(enabled),
        "embedding_tables": EMBEDDING_TABLES
    }

# =============================================================================
# ENDPOINTS - Kategorie-Suche
# =============================================================================

@app.get("/api/search")
async def search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    epoch: Optional[str] = Query(None, description="Epoch filter"),
):
    search_q = f"{q} {epoch}" if epoch and epoch != "all" else q
    combined = await search_all_museums(search_q, limit)
    return {"query": q, "epoch_filter": epoch, "total_results": len(combined), "results": combined[:limit]}

# =============================================================================
# VISUAL SEARCH — Fibula_embeddings / coin_embeddings
# Dreistufig: 1) pgvector RPC  2) Python Cosine  3) Museum-API Fallback
# =============================================================================

@app.post("/api/visual-search")
async def visual_search(req: VisualSearchRequest):
    req.limit = max(1, min(100, req.limit))
    req.min_similarity = max(0.0, min(1.0, req.min_similarity))

    table_name = get_table_for_finder(req.finder_type)
    results = []
    search_mode = "category_fallback"

    # --- Stufe 1+2: Embedding-basiert (wenn Embedding vorhanden) ---
    if req.embedding and len(req.embedding) >= 10:
        # Stufe 1: pgvector RPC
        try:
            rpc_name = f"match_{req.finder_type}"  # match_fibel oder match_muenze
            results = await _vs_pgvector(req, rpc_name)
            if results:
                search_mode = "pgvector"
        except Exception as e:
            print(f"pgvector ({rpc_name}) failed: {e}")

        # Stufe 2: Python Fallback direkt aus der Tabelle
        if not results:
            try:
                results = await _vs_python_fallback(req, table_name)
                if results:
                    search_mode = "python_cosine"
            except Exception as e:
                print(f"Python fallback ({table_name}) failed: {e}")

    # --- Stufe 3: Museum-API Kategorie-Suche ---
    if not results:
        search_mode = "category_fallback"
        epoch_key = req.epoch if req.epoch and req.epoch != "all" else "all"
        terms = FINDER_SEARCH_TERMS.get(req.finder_type, FINDER_SEARCH_TERMS["fibel"])
        query = terms.get(epoch_key, terms["all"])
        if req.material and req.material != "all":
            query = f"{query} {req.material}"
        print(f"Fallback search: '{query}' (type={req.finder_type}, table={table_name}, epoch={epoch_key})")
        results = await search_all_museums(query, req.limit)
        for r in results:
            r["search_mode"] = "category"

    return {
        "finder_type": req.finder_type,
        "table": table_name,
        "epoch_filter": req.epoch,
        "min_similarity": req.min_similarity,
        "search_mode": search_mode,
        "total_results": len(results),
        "results": results[:req.limit]
    }


async def _vs_pgvector(req: VisualSearchRequest, rpc_name: str) -> list:
    """pgvector RPC: match_fibel oder match_muenze."""
    rpc_params = {
        "query_embedding": req.embedding,
        "match_threshold": req.min_similarity,
        "match_count": req.limit,
    }
    # Optionale Filter nur senden wenn gesetzt
    if req.epoch and req.epoch != "all":
        rpc_params["filter_epoch"] = req.epoch
    if req.material and req.material != "all":
        rpc_params["filter_material"] = req.material
    if req.museum and req.museum != "all":
        rpc_params["filter_museum"] = req.museum

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/{rpc_name}",
            json=rpc_params,
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}", "Content-Type": "application/json"},
            timeout=15.0
        )
        if response.status_code != 200:
            raise Exception(f"RPC {rpc_name} failed: {response.status_code} - {response.text}")
        data = response.json()

    return [{
        "id": item.get("id", ""),
        "title": item.get("title", "Unbekannt"),
        "image_url": item.get("image_url", ""),
        "source": item.get("museum", item.get("source", "")),
        "source_url": item.get("source_url", ""),
        "museum": item.get("museum", ""),
        "epoch": item.get("epoch", ""),
        "material": item.get("material", ""),
        "object_type": item.get("object_type", req.finder_type),
        "similarity": round(item.get("similarity", 0), 4),
        "search_mode": "visual",
    } for item in data]


async def _vs_python_fallback(req: VisualSearchRequest, table_name: str) -> list:
    """Python Cosine-Similarity direkt aus Fibula_embeddings / coin_embeddings."""
    params = {
        "select": "id,title,image_url,source_url,museum,epoch,material,object_type,embedding",
        "order": "created_at.desc",
        "limit": 500
    }
    if req.epoch and req.epoch != "all":
        params["epoch"] = f"eq.{req.epoch}"
    if req.material and req.material != "all":
        params["material"] = f"eq.{req.material}"
    if req.museum and req.museum != "all":
        params["museum"] = f"eq.{req.museum}"

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/{table_name}",
            params=params,
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"},
            timeout=15.0
        )
        if response.status_code != 200:
            raise Exception(f"Query {table_name} failed: {response.status_code} - {response.text}")
        items = response.json()

    if not items:
        return []

    scored = []
    for item in items:
        emb = item.get("embedding")
        if not emb: continue
        if isinstance(emb, str):
            try: emb = json.loads(emb)
            except: continue
        sim = cosine_similarity(req.embedding, emb)
        if sim >= req.min_similarity:
            scored.append({
                "id": item.get("id", ""),
                "title": item.get("title", "Unbekannt"),
                "image_url": item.get("image_url", ""),
                "source": item.get("museum", item.get("source", "")),
                "source_url": item.get("source_url", ""),
                "museum": item.get("museum", ""),
                "epoch": item.get("epoch", ""),
                "material": item.get("material", ""),
                "object_type": item.get("object_type", req.finder_type),
                "similarity": round(sim, 4),
                "search_mode": "visual",
            })
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:req.limit]

# =============================================================================
# ENDPOINTS - Stats & Filters (pro Tabelle)
# =============================================================================

@app.get("/api/visual-search/stats")
async def visual_search_stats():
    """Anzahl Eintraege pro Embedding-Tabelle."""
    stats = {"fibel": 0, "muenze": 0, "total": 0}
    async with httpx.AsyncClient() as client:
        for finder_type, table_name in EMBEDDING_TABLES.items():
            try:
                response = await client.get(
                    f"{SUPABASE_URL}/rest/v1/{table_name}",
                    params={"select": "id", "limit": 1},
                    headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}", "Prefer": "count=exact"},
                    timeout=10.0
                )
                cr = response.headers.get("content-range", "")
                if "/" in cr:
                    count = int(cr.split("/")[1])
                    stats[finder_type] = count
                    stats["total"] += count
            except Exception as e:
                print(f"Stats error for {table_name}: {e}")
    return stats

@app.get("/api/visual-search/filters")
async def visual_search_filters(
    finder_type: str = Query("fibel", description="fibel oder muenze")
):
    """Verfuegbare Filter-Optionen aus der jeweiligen Tabelle."""
    table_name = EMBEDDING_TABLES.get(finder_type)
    if not table_name:
        return {"epochs": [], "materials": [], "museums": []}

    filters = {"epochs": [], "materials": [], "museums": []}
    field_map = {"epoch": "epochs", "material": "materials", "museum": "museums"}

    async with httpx.AsyncClient() as client:
        for field, key in field_map.items():
            try:
                response = await client.get(
                    f"{SUPABASE_URL}/rest/v1/{table_name}",
                    params={"select": field, "order": f"{field}.asc", "limit": 1000},
                    headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"},
                    timeout=10.0
                )
                if response.status_code == 200:
                    data = response.json()
                    filters[key] = sorted(set(
                        str(item.get(field, "")).strip()
                        for item in data
                        if item.get(field) and str(item.get(field, "")).strip()
                    ))
            except Exception as e:
                print(f"Filter error for {table_name}.{field}: {e}")
    return filters

# =============================================================================
# USER FINDS ENDPOINTS
# =============================================================================

@app.get("/api/finds")
async def get_user_finds(authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user: raise HTTPException(status_code=401, detail="Not authenticated")
    async with httpx.AsyncClient() as client:
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
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{SUPABASE_URL}/rest/v1/finds", json=find_data,
            headers={"Authorization": f"Bearer {authorization.replace('Bearer ', '')}", "apikey": SUPABASE_ANON_KEY,
                     "Content-Type": "application/json", "Prefer": "return=representation"}, timeout=10.0)
        if response.status_code == 201:
            return {"success": True, "find": response.json()[0] if response.json() else None}
        else: raise HTTPException(status_code=response.status_code, detail=f"Failed: {response.text}")

@app.put("/api/finds/{find_id}")
async def update_find(find_id: str, find: FindUpdate, authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user: raise HTTPException(status_code=401, detail="Not authenticated")
    update_data = {k: v for k, v in find.dict().items() if v is not None}
    update_data["updated_at"] = datetime.utcnow().isoformat()
    if update_data.get("ai_labels"): update_data["ai_labels"] = json.dumps(update_data["ai_labels"])
    if update_data.get("matched_artifacts"): update_data["matched_artifacts"] = json.dumps(update_data["matched_artifacts"])
    if update_data.get("find_coordinates"): update_data["find_coordinates"] = json.dumps(update_data["find_coordinates"])
    async with httpx.AsyncClient() as client:
        response = await client.patch(f"{SUPABASE_URL}/rest/v1/finds",
            params={"id": f"eq.{find_id}", "user_id": f"eq.{user['id']}"}, json=update_data,
            headers={"Authorization": f"Bearer {authorization.replace('Bearer ', '')}", "apikey": SUPABASE_ANON_KEY,
                     "Content-Type": "application/json", "Prefer": "return=representation"}, timeout=10.0)
        if response.status_code == 200:
            return {"success": True, "find": response.json()[0] if response.json() else None}
        else: raise HTTPException(status_code=response.status_code, detail="Failed to update")

@app.delete("/api/finds/{find_id}")
async def delete_find(find_id: str, authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user: raise HTTPException(status_code=401, detail="Not authenticated")
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{SUPABASE_URL}/rest/v1/finds",
            params={"id": f"eq.{find_id}", "user_id": f"eq.{user['id']}"},
            headers={"Authorization": f"Bearer {authorization.replace('Bearer ', '')}", "apikey": SUPABASE_ANON_KEY}, timeout=10.0)
        if response.status_code == 204: return {"success": True}
        else: raise HTTPException(status_code=response.status_code, detail="Failed to delete")

@app.get("/api/profile")
async def get_profile(authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user: raise HTTPException(status_code=401, detail="Not authenticated")
    return {"user": user}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
