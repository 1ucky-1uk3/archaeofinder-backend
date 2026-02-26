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
    """Request fuer visuelle Aehnlichkeitssuche (Fibelfinder/Muenzfinder)"""
    embedding: List[float]                   # CLIP Embedding vom Browser (512 oder 768 dim)
    finder_type: str = "fibel"               # "fibel", "muenze", "all"
    epoch: Optional[str] = None              # z.B. "Roemisch", "Mittelalter"
    material: Optional[str] = None           # z.B. "Bronze", "Silber"
    museum: Optional[str] = None             # Museum-Filter
    limit: int = 20                          # Max Ergebnisse (1-100)
    min_similarity: float = 0.5              # Min Cosine Similarity (0.0-1.0)

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
                headers={
                    "Authorization": f"Bearer {token}",
                    "apikey": SUPABASE_ANON_KEY
                },
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
# COSINE SIMILARITY (Python Fallback)
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
# MUSEUM API FUNCTIONS (Kategorie-Suche)
# =============================================================================

async def search_europeana(client: httpx.AsyncClient, query: str, limit: int = 20):
    if not API_KEYS["europeana"]:
        return []
    try:
        params = {
            "wskey": API_KEYS["europeana"],
            "query": query,
            "rows": limit,
            "profile": "rich",
            "qf": "TYPE:IMAGE"
        }
        response = await client.get(
            "https://api.europeana.eu/record/v2/search.json",
            params=params, timeout=10.0
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
            results.append({
                "id": item.get("id", ""),
                "title": item.get("title", ["Unbekannt"])[0] if isinstance(item.get("title"), list) else item.get("title", "Unbekannt"),
                "image_url": img,
                "source": "Europeana",
                "source_url": item.get("guid", ""),
                "museum": item.get("dataProvider", [""])[0] if isinstance(item.get("dataProvider"), list) else item.get("dataProvider", ""),
            })
        return results
    except Exception as e:
        print(f"Europeana error: {e}")
        return []

async def search_met(client: httpx.AsyncClient, query: str, limit: int = 20):
    try:
        search_response = await client.get(
            "https://collectionapi.metmuseum.org/public/collection/v1/search",
            params={"q": query, "hasImages": "true"}, timeout=10.0
        )
        search_response.raise_for_status()
        search_data = search_response.json()
        object_ids = search_data.get("objectIDs", [])[:limit]
        if not object_ids:
            return []
        results = []
        for obj_id in object_ids[:8]:
            try:
                obj_response = await client.get(
                    f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}",
                    timeout=5.0
                )
                obj = obj_response.json()
                if obj.get("primaryImage"):
                    results.append({
                        "id": f"met_{obj_id}",
                        "title": obj.get("title", "Unbekannt"),
                        "image_url": obj.get("primaryImageSmall") or obj.get("primaryImage"),
                        "source": "Met Museum",
                        "source_url": obj.get("objectURL", ""),
                        "museum": "Metropolitan Museum of Art",
                        "epoch": obj.get("objectDate", ""),
                    })
            except:
                continue
        return results
    except Exception as e:
        print(f"Met error: {e}")
        return []

async def search_va(client: httpx.AsyncClient, query: str, limit: int = 20):
    try:
        response = await client.get(
            "https://api.vam.ac.uk/v2/objects/search",
            params={"q": query, "page_size": limit, "images_exist": 1},
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("records", []):
            img = item.get("_images", {}).get("_primary_thumbnail")
            if img:
                results.append({
                    "id": f"va_{item.get('systemNumber', '')}",
                    "title": item.get("_primaryTitle", "Unbekannt"),
                    "image_url": img,
                    "source": "V&A Museum",
                    "source_url": f"https://collections.vam.ac.uk/item/{item.get('systemNumber', '')}",
                    "museum": "Victoria & Albert Museum",
                    "epoch": item.get("_primaryDate", ""),
                })
        return results
    except Exception as e:
        print(f"V&A error: {e}")
        return []

async def search_rijksmuseum(client: httpx.AsyncClient, query: str, limit: int = 20):
    if not API_KEYS["rijksmuseum"]:
        return []
    try:
        response = await client.get(
            "https://www.rijksmuseum.nl/api/en/collection",
            params={"key": API_KEYS["rijksmuseum"], "q": query, "ps": limit, "imgonly": "true", "format": "json"},
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("artObjects", []):
            if item.get("webImage", {}).get("url"):
                results.append({
                    "id": item.get("objectNumber", ""),
                    "title": item.get("title", "Unbekannt"),
                    "image_url": item["webImage"]["url"],
                    "source": "Rijksmuseum",
                    "source_url": item.get("links", {}).get("web", ""),
                    "museum": "Rijksmuseum Amsterdam",
                })
        return results
    except Exception as e:
        print(f"Rijksmuseum error: {e}")
        return []

async def search_smithsonian(client: httpx.AsyncClient, query: str, limit: int = 20):
    if not API_KEYS["smithsonian"]:
        return []
    try:
        response = await client.get(
            "https://api.si.edu/openaccess/api/v1.0/search",
            params={"api_key": API_KEYS["smithsonian"], "q": query + " AND online_media_type:Images", "rows": limit},
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
        results = []
        for row in data.get("response", {}).get("rows", []):
            content = row.get("content", {})
            desc = content.get("descriptiveNonRepeating", {})
            img = None
            media_list = content.get("online_media", {}).get("media", [])
            if media_list:
                img = media_list[0].get("content", "")
            if img:
                results.append({
                    "id": row.get("id", ""),
                    "title": desc.get("title", {}).get("content", "Unbekannt"),
                    "image_url": img,
                    "source": "Smithsonian",
                    "source_url": desc.get("record_link", ""),
                    "museum": desc.get("unit_name", "Smithsonian"),
                })
        return results
    except Exception as e:
        print(f"Smithsonian error: {e}")
        return []

async def search_harvard(client: httpx.AsyncClient, query: str, limit: int = 20):
    if not API_KEYS["harvard"]:
        return []
    try:
        response = await client.get(
            "https://api.harvardartmuseums.org/object",
            params={"apikey": API_KEYS["harvard"], "q": query, "size": limit, "hasimage": 1},
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("records", []):
            if item.get("primaryimageurl"):
                results.append({
                    "id": f"harvard_{item.get('id', '')}",
                    "title": item.get("title", "Unbekannt"),
                    "image_url": item["primaryimageurl"],
                    "source": "Harvard Museums",
                    "source_url": item.get("url", ""),
                    "museum": "Harvard Art Museums",
                    "epoch": item.get("dated", ""),
                })
        return results
    except Exception as e:
        print(f"Harvard error: {e}")
        return []

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
        "name": "ArchaeoFinder API",
        "version": "4.0.0",
        "status": "online",
        "features": ["multi_museum_search", "visual_similarity_search", "fibel_finder", "muenz_finder", "user_auth", "save_finds"],
        "enabled_apis": enabled,
        "total_sources": len(enabled)
    }

# =============================================================================
# ENDPOINTS - Kategorie-Suche (bestehend, erweitert um Epoch-Filter)
# =============================================================================

@app.get("/api/search")
async def search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    epoch: Optional[str] = Query(None, description="Epoch filter"),
):
    async with httpx.AsyncClient() as client:
        search_q = f"{q} {epoch}" if epoch and epoch != "all" else q
        tasks = [
            search_europeana(client, search_q, limit),
            search_met(client, search_q, limit),
            search_va(client, search_q, limit),
            search_rijksmuseum(client, search_q, limit),
            search_smithsonian(client, search_q, limit),
            search_harvard(client, search_q, limit),
        ]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
    combined = []
    for results in all_results:
        if isinstance(results, list):
            combined.extend(results)
    combined = [r for r in combined if r.get("image_url")]
    return {
        "query": q,
        "epoch_filter": epoch,
        "total_results": len(combined),
        "results": combined[:limit]
    }

# =============================================================================
# ENDPOINTS - Visual Similarity Search (Fibelfinder / Muenzfinder)
# =============================================================================

@app.post("/api/visual-search")
async def visual_search(req: VisualSearchRequest):
    """
    Visuelle Aehnlichkeitssuche.
    Empfaengt CLIP-Embedding vom Frontend, sucht aehnliche Artefakte in DB.
    Strategie: pgvector RPC (schnell) -> Python Fallback (langsam)
    """
    if not req.embedding or len(req.embedding) < 10:
        raise HTTPException(status_code=400, detail="Invalid embedding vector")
    req.limit = max(1, min(100, req.limit))
    req.min_similarity = max(0.0, min(1.0, req.min_similarity))
    
    try:
        results = await _visual_search_pgvector(req)
    except Exception as e:
        print(f"pgvector search failed, using fallback: {e}")
        results = await _visual_search_fallback(req)
    
    return {
        "finder_type": req.finder_type,
        "epoch_filter": req.epoch,
        "min_similarity": req.min_similarity,
        "total_results": len(results),
        "results": results
    }


async def _visual_search_pgvector(req: VisualSearchRequest) -> list:
    """Suche via Supabase pgvector RPC (match_artifacts Funktion)."""
    rpc_params = {
        "query_embedding": req.embedding,
        "match_threshold": req.min_similarity,
        "match_count": req.limit,
        "filter_type": req.finder_type if req.finder_type != "all" else None,
        "filter_epoch": req.epoch if req.epoch and req.epoch != "all" else None,
        "filter_material": req.material if req.material and req.material != "all" else None,
        "filter_museum": req.museum if req.museum and req.museum != "all" else None,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/match_artifacts",
            json=rpc_params,
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                "Content-Type": "application/json"
            },
            timeout=15.0
        )
        if response.status_code != 200:
            raise Exception(f"RPC failed: {response.status_code} - {response.text}")
        data = response.json()
    
    return [{
        "id": item.get("id", ""),
        "title": item.get("title", "Unbekannt"),
        "image_url": item.get("image_url", ""),
        "source": item.get("museum", ""),
        "source_url": item.get("source_url", ""),
        "museum": item.get("museum", ""),
        "epoch": item.get("epoch", ""),
        "material": item.get("material", ""),
        "object_type": item.get("object_type", ""),
        "similarity": round(item.get("similarity", 0), 4),
    } for item in data]


async def _visual_search_fallback(req: VisualSearchRequest) -> list:
    """Fallback: Embeddings laden, Cosine-Similarity in Python berechnen."""
    params = {
        "select": "id,title,image_url,source_url,museum,epoch,material,object_type,embedding",
        "order": "created_at.desc",
        "limit": 500
    }
    if req.finder_type and req.finder_type != "all":
        params["object_type"] = f"eq.{req.finder_type}"
    if req.epoch and req.epoch != "all":
        params["epoch"] = f"eq.{req.epoch}"
    if req.material and req.material != "all":
        params["material"] = f"eq.{req.material}"
    if req.museum and req.museum != "all":
        params["museum"] = f"eq.{req.museum}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/artifact_embeddings",
            params=params,
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"},
            timeout=15.0
        )
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to query embeddings")
        items = response.json()
    
    scored = []
    for item in items:
        emb = item.get("embedding")
        if not emb:
            continue
        if isinstance(emb, str):
            try:
                emb = json.loads(emb)
            except:
                continue
        sim = cosine_similarity(req.embedding, emb)
        if sim >= req.min_similarity:
            scored.append({
                "id": item.get("id", ""),
                "title": item.get("title", "Unbekannt"),
                "image_url": item.get("image_url", ""),
                "source": item.get("museum", ""),
                "source_url": item.get("source_url", ""),
                "museum": item.get("museum", ""),
                "epoch": item.get("epoch", ""),
                "material": item.get("material", ""),
                "object_type": item.get("object_type", ""),
                "similarity": round(sim, 4),
            })
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:req.limit]

# =============================================================================
# ENDPOINTS - Visual Search Stats & Filter-Optionen
# =============================================================================

@app.get("/api/visual-search/stats")
async def visual_search_stats():
    """Statistiken: Anzahl Embeddings pro Typ."""
    stats = {"total": 0, "fibel": 0, "muenze": 0}
    async with httpx.AsyncClient() as client:
        for key in ["total", "fibel", "muenze"]:
            try:
                p = {"select": "id", "limit": 1}
                if key != "total":
                    p["object_type"] = f"eq.{key}"
                response = await client.get(
                    f"{SUPABASE_URL}/rest/v1/artifact_embeddings",
                    params=p,
                    headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}", "Prefer": "count=exact"},
                    timeout=10.0
                )
                total = response.headers.get("content-range", "")
                if "/" in total:
                    stats[key] = int(total.split("/")[1])
            except:
                pass
    return stats

@app.get("/api/visual-search/filters")
async def visual_search_filters():
    """Verfuegbare Filter-Optionen aus der Datenbank."""
    filters = {"epochs": [], "materials": [], "museums": [], "object_types": []}
    async with httpx.AsyncClient() as client:
        for field in ["epoch", "material", "museum", "object_type"]:
            try:
                response = await client.get(
                    f"{SUPABASE_URL}/rest/v1/artifact_embeddings",
                    params={"select": field, "order": f"{field}.asc", "limit": 500},
                    headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"},
                    timeout=10.0
                )
                if response.status_code == 200:
                    data = response.json()
                    unique_vals = sorted(set(item.get(field, "") for item in data if item.get(field)))
                    key = field + "s" if not field.endswith("s") else field
                    if field == "object_type":
                        key = "object_types"
                    filters[key] = unique_vals
            except:
                pass
    return filters

# =============================================================================
# USER FINDS ENDPOINTS (bestehend)
# =============================================================================

@app.get("/api/finds")
async def get_user_finds(authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/finds",
            params={"user_id": f"eq.{user['id']}", "order": "created_at.desc"},
            headers={"Authorization": f"Bearer {authorization.replace('Bearer ', '')}", "apikey": SUPABASE_ANON_KEY},
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
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/finds",
            json=find_data,
            headers={"Authorization": f"Bearer {authorization.replace('Bearer ', '')}", "apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json", "Prefer": "return=representation"},
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
    async with httpx.AsyncClient() as client:
        response = await client.patch(
            f"{SUPABASE_URL}/rest/v1/finds",
            params={"id": f"eq.{find_id}", "user_id": f"eq.{user['id']}"},
            json=update_data,
            headers={"Authorization": f"Bearer {authorization.replace('Bearer ', '')}", "apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json", "Prefer": "return=representation"},
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
    async with httpx.AsyncClient() as client:
        response = await client.delete(
            f"{SUPABASE_URL}/rest/v1/finds",
            params={"id": f"eq.{find_id}", "user_id": f"eq.{user['id']}"},
            headers={"Authorization": f"Bearer {authorization.replace('Bearer ', '')}", "apikey": SUPABASE_ANON_KEY},
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
