from fastapi import FastAPI, Query, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx
import asyncio
import os
from datetime import datetime
import json
from difflib import SequenceMatcher

app = FastAPI(title="ArchaeoFinder API", version="3.1.0")

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
# EPOCH / DEPARTMENT MAPPINGS (Phase 1: API-Filter)
# =============================================================================

MET_DEPARTMENT_MAP = {
    "prehistoric": None,
    "neolithic": None,
    "bronze_age": None,
    "iron_age": None,
    "greek": 13,
    "roman": 13,
    "egyptian": 10,
    "medieval": 17,
    "islamic": 14,
    "asian": 6,
}

EUROPEANA_EPOCH_FILTERS = {
    "prehistoric": "archaeology prehistory",
    "neolithic": "neolithic stone age",
    "bronze_age": "bronze age",
    "iron_age": "iron age celtic",
    "greek": "ancient greek",
    "roman": "roman empire",
    "egyptian": "ancient egypt",
    "medieval": "medieval middle ages",
    "viking": "viking norse",
}

VA_CATEGORY_MAP = {
    "stone": "Metalwork",
    "bronze": "Metalwork",
    "iron": "Metalwork",
    "ceramic": "Ceramics",
    "pottery": "Ceramics",
    "glass": "Glass",
    "coin": "Coins & Medals",
    "jewelry": "Jewellery",
}

# =============================================================================
# DEDUPLICATION (Phase 1.4)
# =============================================================================

def title_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a_clean = a.lower().strip()
    b_clean = b.lower().strip()
    if a_clean == b_clean:
        return 1.0
    return SequenceMatcher(None, a_clean, b_clean).ratio()

def deduplicate_results(results: list, threshold: float = 0.85) -> list:
    if not results:
        return results
    
    seen_urls = set()
    unique = []
    
    for item in results:
        img_url = (item.get("image_url") or "").split("?")[0].lower()
        
        if img_url and img_url in seen_urls:
            continue
        
        is_duplicate = False
        item_title = item.get("title", "")
        for accepted in unique:
            if title_similarity(item_title, accepted.get("title", "")) > threshold:
                if not accepted.get("image_url") and item.get("image_url"):
                    unique.remove(accepted)
                    break
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(item)
            if img_url:
                seen_urls.add(img_url)
    
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
# MUSEUM API FUNCTIONS (Phase 1.3: Enhanced filters)
# =============================================================================

async def search_europeana(client, query, limit=20, epoch=None, material=None):
    if not API_KEYS["europeana"]:
        return []
    try:
        search_query = query
        if epoch and epoch in EUROPEANA_EPOCH_FILTERS:
            search_query = f"{query} {EUROPEANA_EPOCH_FILTERS[epoch]}"
        if material:
            search_query = f"{search_query} {material}"

        params = {
            "wskey": API_KEYS["europeana"],
            "query": search_query,
            "rows": limit,
            "profile": "rich",
            "qf": "TYPE:IMAGE"
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
            
            results.append({
                "id": item.get("id", ""),
                "title": item.get("title", ["Unbekannt"])[0] if isinstance(item.get("title"), list) else item.get("title", "Unbekannt"),
                "image_url": img,
                "source": "Europeana",
                "source_url": item.get("guid", ""),
                "museum": item.get("dataProvider", [""])[0] if isinstance(item.get("dataProvider"), list) else item.get("dataProvider", ""),
                "epoch": item_epoch,
            })
        return results
    except Exception as e:
        print(f"Europeana error: {e}")
        return []

async def search_met(client, query, limit=20, epoch=None, material=None):
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
        
        object_ids = search_data.get("objectIDs", [])[:limit]
        if not object_ids:
            return []
        
        async def fetch_met_object(obj_id):
            try:
                r = await client.get(
                    f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}",
                    timeout=5.0
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
                    }
            except:
                pass
            return None
        
        met_results = await asyncio.gather(*[fetch_met_object(oid) for oid in object_ids[:10]])
        return [r for r in met_results if r is not None]
    except Exception as e:
        print(f"Met error: {e}")
        return []

async def search_va(client, query, limit=20, epoch=None, material=None):
    try:
        params = {"q": query, "page_size": limit, "images_exist": 1}
        if material and material.lower() in VA_CATEGORY_MAP:
            params["q_object_type"] = VA_CATEGORY_MAP[material.lower()]
        
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
                })
        return results
    except Exception as e:
        print(f"V&A error: {e}")
        return []

async def search_rijksmuseum(client, query, limit=20, epoch=None, material=None):
    if not API_KEYS["rijksmuseum"]:
        return []
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
        return results
    except Exception as e:
        print(f"Rijksmuseum error: {e}")
        return []

async def search_smithsonian(client, query, limit=20, epoch=None, material=None):
    if not API_KEYS["smithsonian"]:
        return []
    try:
        search_q = query + " AND online_media_type:Images"
        if material:
            search_q += f" AND {material}"
        params = {
            "api_key": API_KEYS["smithsonian"],
            "q": search_q, "rows": limit
        }
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

async def search_harvard(client, query, limit=20, epoch=None, material=None):
    if not API_KEYS["harvard"]:
        return []
    try:
        params = {
            "apikey": API_KEYS["harvard"], "q": query,
            "size": limit, "hasimage": 1
        }
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
        return results
    except Exception as e:
        print(f"Harvard error: {e}")
        return []

# =============================================================================
# API ENDPOINTS
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
        "version": "3.1.0",
        "status": "online",
        "features": ["multi_museum_search", "user_auth", "save_finds", "smart_filters", "deduplication"],
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
    async with httpx.AsyncClient() as client:
        primary_tasks = [
            search_europeana(client, q, limit, epoch=epoch, material=material),
            search_met(client, q, limit, epoch=epoch, material=material),
            search_va(client, q, limit, epoch=epoch, material=material),
            search_rijksmuseum(client, q, limit, epoch=epoch, material=material),
            search_smithsonian(client, q, limit, epoch=epoch, material=material),
            search_harvard(client, q, limit, epoch=epoch, material=material),
        ]
        
        all_tasks = primary_tasks
        
        if q2:
            secondary_limit = max(limit // 2, 5)
            secondary_tasks = [
                search_europeana(client, q2, secondary_limit, epoch=epoch, material=material),
                search_met(client, q2, secondary_limit, epoch=epoch, material=material),
                search_va(client, q2, secondary_limit, epoch=epoch, material=material),
            ]
            all_tasks.extend(secondary_tasks)
        
        all_results = await asyncio.gather(*all_tasks, return_exceptions=True)
    
    combined = []
    for results in all_results:
        if isinstance(results, list):
            combined.extend(results)
    
    combined = [r for r in combined if r.get("image_url")]
    combined = deduplicate_results(combined)
    
    return {
        "query": q,
        "query_secondary": q2,
        "filters": {"epoch": epoch, "material": material},
        "total_results": len(combined),
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
    async with httpx.AsyncClient() as client:
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
    
    async with httpx.AsyncClient() as client:
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
    
    async with httpx.AsyncClient() as client:
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
    async with httpx.AsyncClient() as client:
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
