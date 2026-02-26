from fastapi import FastAPI, Query, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx
import asyncio
import os
from datetime import datetime
import json

app = FastAPI(title="ArchaeoFinder API", version="5.0.0")

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

class FibelSearchRequest(BaseModel):
    embedding: List[float]
    match_threshold: float = 0.5
    match_count: int = 20
    source_filter: Optional[str] = None

class CoinSearchRequest(BaseModel):
    embedding: List[float]
    match_threshold: float = 0.5
    match_count: int = 20
    source_filter: Optional[str] = None

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

# =============================================================================
# MUSEUM API FUNCTIONS (unveraendert)
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
        search_response = await client.get("https://collectionapi.metmuseum.org/public/collection/v1/search", params={"q": query, "hasImages": "true"}, timeout=10.0)
        search_response.raise_for_status()
        object_ids = search_response.json().get("objectIDs", [])[:limit]
        if not object_ids:
            return []
        results = []
        for obj_id in object_ids[:8]:
            try:
                obj_response = await client.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}", timeout=5.0)
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
        response = await client.get("https://api.vam.ac.uk/v2/objects/search", params={"q": query, "page_size": limit, "images_exist": 1}, timeout=10.0)
        response.raise_for_status()
        results = []
        for item in response.json().get("records", []):
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
        params = {"key": API_KEYS["rijksmuseum"], "q": query, "ps": limit, "imgonly": "true", "format": "json"}
        response = await client.get("https://www.rijksmuseum.nl/api/en/collection", params=params, timeout=10.0)
        response.raise_for_status()
        results = []
        for item in response.json().get("artObjects", []):
            if item.get("webImage", {}).get("url"):
                results.append({
                    "id": item.get("objectNumber", ""),
                    "title": item.get("title", "Unbekannt"),
                    "image_url": item.get("webImage", {}).get("url", ""),
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
        params = {"api_key": API_KEYS["smithsonian"], "q": query + " AND online_media_type:Images", "rows": limit}
        response = await client.get("https://api.si.edu/openaccess/api/v1.0/search", params=params, timeout=10.0)
        response.raise_for_status()
        results = []
        for row in response.json().get("response", {}).get("rows", []):
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
        params = {"apikey": API_KEYS["harvard"], "q": query, "size": limit, "hasimage": 1}
        response = await client.get("https://api.harvardartmuseums.org/object", params=params, timeout=10.0)
        response.raise_for_status()
        results = []
        for item in response.json().get("records", []):
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
        return results
    except Exception as e:
        print(f"Harvard error: {e}")
        return []

# =============================================================================
# FIBEL-FINDER — Visuelle Suche (768d vom Browser)
# =============================================================================

@app.post("/api/fibel/search")
async def fibel_search(request: FibelSearchRequest):
    """
    Empfaengt CLIP ViT-L/14 Embedding (768d) vom Browser,
    sucht in Supabase via pgvector Cosine Similarity.
    """
    if len(request.embedding) != 768:
        raise HTTPException(status_code=400, detail=f"Embedding muss 768d haben, nicht {len(request.embedding)}")

    embedding_str = "[" + ",".join(str(x) for x in request.embedding) + "]"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/search_fibulae",
            json={
                "query_embedding": embedding_str,
                "match_threshold": request.match_threshold,
                "match_count": request.match_count,
            },
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Content-Type": "application/json",
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            },
            timeout=15.0,
        )
        if response.status_code != 200:
            print(f"Supabase RPC error: {response.status_code} {response.text}")
            raise HTTPException(status_code=502, detail=f"Supabase Fehler: {response.status_code}")
        results = response.json()

    if request.source_filter:
        results = [r for r in results if r.get("source") == request.source_filter]

    for r in results:
        if "similarity" in r:
            r["similarity_pct"] = round(r["similarity"] * 100, 1)

    return {"total_results": len(results), "results": results}


@app.get("/api/fibel/stats")
async def fibel_stats():
    async with httpx.AsyncClient() as client:
        r1 = await client.get(f"{SUPABASE_URL}/rest/v1/fibula_stats", params={"select": "*"},
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"}, timeout=10.0)
        r2 = await client.get(f"{SUPABASE_URL}/rest/v1/fibula_source_stats", params={"select": "*"},
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"}, timeout=10.0)
        total = r1.json()[0] if r1.status_code == 200 and r1.json() else {}
        sources = r2.json() if r2.status_code == 200 else []
    return {"total": total, "sources": sources}


@app.get("/api/fibel/random")
async def fibel_random(count: int = Query(12, ge=1, le=50)):
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{SUPABASE_URL}/rest/v1/fibula_embeddings",
            params={"select": "id,source,source_id,title,image_url,thumbnail_url,source_url,museum,epoch,material,fibula_type", "limit": count, "order": "created_at.desc"},
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"}, timeout=10.0)
        return {"results": r.json() if r.status_code == 200 else []}


# =============================================================================
# MUENZ-FINDER — Visuelle Suche (768d vom Browser)
# =============================================================================

@app.post("/api/coin/search")
async def coin_search(request: CoinSearchRequest):
    """
    Empfaengt CLIP ViT-L/14 Embedding (768d) vom Browser,
    sucht in Supabase via pgvector Cosine Similarity.
    """
    if len(request.embedding) != 768:
        raise HTTPException(status_code=400, detail=f"Embedding muss 768d haben, nicht {len(request.embedding)}")

    embedding_str = "[" + ",".join(str(x) for x in request.embedding) + "]"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/search_coins",
            json={
                "query_embedding": embedding_str,
                "match_threshold": request.match_threshold,
                "match_count": request.match_count,
            },
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Content-Type": "application/json",
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            },
            timeout=15.0,
        )
        if response.status_code != 200:
            print(f"Supabase RPC error (coins): {response.status_code} {response.text}")
            raise HTTPException(status_code=502, detail=f"Supabase Fehler: {response.status_code}")
        results = response.json()

    if request.source_filter:
        results = [r for r in results if r.get("source") == request.source_filter]

    for r in results:
        if "similarity" in r:
            r["similarity_pct"] = round(r["similarity"] * 100, 1)

    return {"total_results": len(results), "results": results}


@app.get("/api/coin/stats")
async def coin_stats():
    async with httpx.AsyncClient() as client:
        r1 = await client.get(f"{SUPABASE_URL}/rest/v1/coin_stats", params={"select": "*"},
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"}, timeout=10.0)
        r2 = await client.get(f"{SUPABASE_URL}/rest/v1/coin_source_stats", params={"select": "*"},
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"}, timeout=10.0)
        total = r1.json()[0] if r1.status_code == 200 and r1.json() else {}
        sources = r2.json() if r2.status_code == 200 else []
    return {"total": total, "sources": sources}


@app.get("/api/coin/random")
async def coin_random(count: int = Query(12, ge=1, le=50)):
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{SUPABASE_URL}/rest/v1/coin_embeddings",
            params={"select": "id,source,source_id,title,image_url,thumbnail_url,source_url,museum,epoch,material,coin_type", "limit": count, "order": "created_at.desc"},
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"}, timeout=10.0)
        return {"results": r.json() if r.status_code == 200 else []}


# =============================================================================
# GENERAL ENDPOINTS
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
        "name": "ArchaeoFinder API", "version": "5.0.0", "status": "online",
        "features": ["multi_museum_search", "user_auth", "save_finds", "fibel_visual_search", "coin_visual_search"],
        "enabled_apis": enabled, "total_sources": len(enabled),
    }

@app.get("/api/search")
async def search(q: str = Query(...), limit: int = Query(20, ge=1, le=50)):
    async with httpx.AsyncClient() as client:
        tasks = [
            search_europeana(client, q, limit), search_met(client, q, limit),
            search_va(client, q, limit), search_rijksmuseum(client, q, limit),
            search_smithsonian(client, q, limit), search_harvard(client, q, limit),
        ]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
    combined = []
    for r in all_results:
        if isinstance(r, list): combined.extend(r)
    combined = [r for r in combined if r.get("image_url")]
    return {"query": q, "total_results": len(combined), "results": combined}

# =============================================================================
# USER FINDS (unveraendert)
# =============================================================================

@app.get("/api/finds")
async def get_user_finds(authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user: raise HTTPException(status_code=401, detail="Not authenticated")
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{SUPABASE_URL}/rest/v1/finds",
            params={"user_id": f"eq.{user['id']}", "order": "created_at.desc"},
            headers={"Authorization": f"Bearer {authorization.replace('Bearer ', '')}", "apikey": SUPABASE_ANON_KEY}, timeout=10.0)
        if r.status_code == 200: return {"finds": r.json()}
        raise HTTPException(status_code=r.status_code, detail="Failed to fetch finds")

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
        r = await client.post(f"{SUPABASE_URL}/rest/v1/finds", json=find_data,
            headers={"Authorization": f"Bearer {authorization.replace('Bearer ', '')}", "apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json", "Prefer": "return=representation"}, timeout=10.0)
        if r.status_code == 201: return {"success": True, "find": r.json()[0] if r.json() else None}
        raise HTTPException(status_code=r.status_code, detail=f"Create failed: {r.text}")

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
        r = await client.patch(f"{SUPABASE_URL}/rest/v1/finds",
            params={"id": f"eq.{find_id}", "user_id": f"eq.{user['id']}"}, json=update_data,
            headers={"Authorization": f"Bearer {authorization.replace('Bearer ', '')}", "apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json", "Prefer": "return=representation"}, timeout=10.0)
        if r.status_code == 200: return {"success": True, "find": r.json()[0] if r.json() else None}
        raise HTTPException(status_code=r.status_code, detail="Update failed")

@app.delete("/api/finds/{find_id}")
async def delete_find(find_id: str, authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user: raise HTTPException(status_code=401, detail="Not authenticated")
    async with httpx.AsyncClient() as client:
        r = await client.delete(f"{SUPABASE_URL}/rest/v1/finds",
            params={"id": f"eq.{find_id}", "user_id": f"eq.{user['id']}"},
            headers={"Authorization": f"Bearer {authorization.replace('Bearer ', '')}", "apikey": SUPABASE_ANON_KEY}, timeout=10.0)
        if r.status_code == 204: return {"success": True}
        raise HTTPException(status_code=r.status_code, detail="Delete failed")

@app.get("/api/profile")
async def get_profile(authorization: Optional[str] = Header(None)):
    user = await get_user_from_token(authorization)
    if not user: raise HTTPException(status_code=401, detail="Not authenticated")
    return {"user": user}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
