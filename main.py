from fastapi import FastAPI, Query, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx
import asyncio
import os
from datetime import datetime
import json

app = FastAPI(title="ArchaeoFinder API", version="3.0.0")

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

# Google Cloud Vision API Key (für visuelle Suche v2)
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "")

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
# MUSEUM API FUNCTIONS
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
            params=params,
            timeout=10.0
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
            params={"q": query, "hasImages": "true"},
            timeout=10.0
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
        params = {
            "q": query,
            "page_size": limit,
            "images_exist": 1
        }
        response = await client.get(
            "https://api.vam.ac.uk/v2/objects/search",
            params=params,
            timeout=10.0
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

async def search_rijksmuseum(client: httpx.AsyncClient, query: str, limit: int = 20):
    if not API_KEYS["rijksmuseum"]:
        return []
    try:
        params = {
            "key": API_KEYS["rijksmuseum"],
            "q": query,
            "ps": limit,
            "imgonly": "true",
            "format": "json"
        }
        response = await client.get(
            "https://www.rijksmuseum.nl/api/en/collection",
            params=params,
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
        params = {
            "api_key": API_KEYS["smithsonian"],
            "q": query + " AND online_media_type:Images",
            "rows": limit
        }
        response = await client.get(
            "https://api.si.edu/openaccess/api/v1.0/search",
            params=params,
            timeout=10.0
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

async def search_harvard(client: httpx.AsyncClient, query: str, limit: int = 20):
    if not API_KEYS["harvard"]:
        return []
    try:
        params = {
            "apikey": API_KEYS["harvard"],
            "q": query,
            "size": limit,
            "hasimage": 1
        }
        response = await client.get(
            "https://api.harvardartmuseums.org/object",
            params=params,
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
    enabled.extend(["met", "victoria_albert"])  # Always enabled
    if API_KEYS["rijksmuseum"]: enabled.append("rijksmuseum")
    if API_KEYS["smithsonian"]: enabled.append("smithsonian")
    if API_KEYS["harvard"]: enabled.append("harvard")
    
    return {
        "name": "ArchaeoFinder API",
        "version": "3.0.0",
        "status": "online",
        "features": ["multi_museum_search", "user_auth", "save_finds"],
        "enabled_apis": enabled,
        "total_sources": len(enabled)
    }

@app.get("/api/search")
async def search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=50),
):
    async with httpx.AsyncClient() as client:
        tasks = [
            search_europeana(client, q, limit),
            search_met(client, q, limit),
            search_va(client, q, limit),
            search_rijksmuseum(client, q, limit),
            search_smithsonian(client, q, limit),
            search_harvard(client, q, limit),
        ]
        
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    combined = []
    for results in all_results:
        if isinstance(results, list):
            combined.extend(results)
    
    combined = [r for r in combined if r.get("image_url")]
    
    return {
        "query": q,
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
            params={
                "user_id": f"eq.{user['id']}",
                "order": "created_at.desc"
            },
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
    
    # Convert lists to JSON strings for Supabase
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
            params={
                "id": f"eq.{find_id}",
                "user_id": f"eq.{user['id']}"
            },
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
            params={
                "id": f"eq.{find_id}",
                "user_id": f"eq.{user['id']}"
            },
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


# =============================================================================
# V2: VISUELLE SUCHE (Google Cloud Vision Web Detection)
# =============================================================================
# Env Variable auf DigitalOcean setzen:
#   GOOGLE_VISION_API_KEY = "AIzaSy..."
# =============================================================================

from urllib.parse import urlparse

DOMAIN_WHITELIST = {
    # Tier 1: Große internationale Museen
    "www.metmuseum.org": {"name": "Metropolitan Museum", "tier": 1, "flag": "🇺🇸"},
    "metmuseum.org": {"name": "Metropolitan Museum", "tier": 1, "flag": "🇺🇸"},
    "images.metmuseum.org": {"name": "Metropolitan Museum", "tier": 1, "flag": "🇺🇸"},
    "collectionapi.metmuseum.org": {"name": "Metropolitan Museum", "tier": 1, "flag": "🇺🇸"},
    "collections.vam.ac.uk": {"name": "Victoria & Albert Museum", "tier": 1, "flag": "🇬🇧"},
    "framemark.vam.ac.uk": {"name": "Victoria & Albert Museum", "tier": 1, "flag": "🇬🇧"},
    "www.vam.ac.uk": {"name": "Victoria & Albert Museum", "tier": 1, "flag": "🇬🇧"},
    "www.artic.edu": {"name": "Art Institute Chicago", "tier": 1, "flag": "🇺🇸"},
    "artic.edu": {"name": "Art Institute Chicago", "tier": 1, "flag": "🇺🇸"},
    "www.britishmuseum.org": {"name": "British Museum", "tier": 1, "flag": "🇬🇧"},
    "media.britishmuseum.org": {"name": "British Museum", "tier": 1, "flag": "🇬🇧"},
    "www.clevelandart.org": {"name": "Cleveland Museum of Art", "tier": 1, "flag": "🇺🇸"},
    "openaccess-cdn.clevelandart.org": {"name": "Cleveland Museum of Art", "tier": 1, "flag": "🇺🇸"},
    "www.rijksmuseum.nl": {"name": "Rijksmuseum", "tier": 1, "flag": "🇳🇱"},
    "www.smb.museum": {"name": "Staatliche Museen Berlin", "tier": 1, "flag": "🇩🇪"},
    # Tier 2: Archäologie-Spezialisten
    "finds.org.uk": {"name": "Portable Antiquities Scheme", "tier": 2, "flag": "🇬🇧"},
    "www.finds.org.uk": {"name": "Portable Antiquities Scheme", "tier": 2, "flag": "🇬🇧"},
    "arachne.dainst.org": {"name": "Arachne / DAI", "tier": 2, "flag": "🇩🇪"},
    "www.europeana.eu": {"name": "Europeana", "tier": 2, "flag": "🇪🇺"},
    "europeana.eu": {"name": "Europeana", "tier": 2, "flag": "🇪🇺"},
    "api.europeana.eu": {"name": "Europeana", "tier": 2, "flag": "🇪🇺"},
    "www.harvardartmuseums.org": {"name": "Harvard Art Museums", "tier": 2, "flag": "🇺🇸"},
    "nrs.harvard.edu": {"name": "Harvard Art Museums", "tier": 2, "flag": "🇺🇸"},
    "ids.si.edu": {"name": "Smithsonian", "tier": 2, "flag": "🇺🇸"},
    "collections.si.edu": {"name": "Smithsonian", "tier": 2, "flag": "🇺🇸"},
    "www.si.edu": {"name": "Smithsonian", "tier": 2, "flag": "🇺🇸"},
    # Tier 3: Regionale Museen
    "nat.museum-digital.de": {"name": "museum-digital", "tier": 3, "flag": "🇩🇪"},
    "nrw.museum-digital.de": {"name": "museum-digital NRW", "tier": 3, "flag": "🇩🇪"},
    "hessen.museum-digital.de": {"name": "museum-digital Hessen", "tier": 3, "flag": "🇩🇪"},
    "bw.museum-digital.de": {"name": "museum-digital BW", "tier": 3, "flag": "🇩🇪"},
    "www.pop.culture.gouv.fr": {"name": "POP France", "tier": 3, "flag": "🇫🇷"},
    "pop.culture.gouv.fr": {"name": "POP France", "tier": 3, "flag": "🇫🇷"},
    "commons.wikimedia.org": {"name": "Wikimedia Commons", "tier": 3, "flag": "🌐"},
    "upload.wikimedia.org": {"name": "Wikimedia Commons", "tier": 3, "flag": "🌐"},
    "www.khm.at": {"name": "KHM Wien", "tier": 3, "flag": "🇦🇹"},
    "www.gnm.de": {"name": "Germanisches Nationalmuseum", "tier": 3, "flag": "🇩🇪"},
    "www.landesmuseum.de": {"name": "Badisches Landesmuseum", "tier": 3, "flag": "🇩🇪"},
    "art.thewalters.org": {"name": "Walters Art Museum", "tier": 3, "flag": "🇺🇸"},
    "thewalters.org": {"name": "Walters Art Museum", "tier": 3, "flag": "🇺🇸"},
    "www.penn.museum": {"name": "Penn Museum", "tier": 3, "flag": "🇺🇸"},
    "balat.kikirpa.be": {"name": "KIKIRPA Brüssel", "tier": 3, "flag": "🇧🇪"},
    # Tier 4: Akademische Quellen
    "www.jstor.org": {"name": "JSTOR", "tier": 4, "flag": "📚"},
    "www.academia.edu": {"name": "Academia.edu", "tier": 4, "flag": "📚"},
    "archive.org": {"name": "Internet Archive", "tier": 4, "flag": "📚"},
    "www.numismatics.org": {"name": "American Numismatic Society", "tier": 4, "flag": "🇺🇸"},
    "www.deutsche-digitale-bibliothek.de": {"name": "DDB", "tier": 4, "flag": "🇩🇪"},
}


def _match_domain(url: str):
    """Prüft ob URL auf der Museums-Whitelist ist."""
    try:
        domain = urlparse(url).hostname.lower()
    except Exception:
        return None
    if domain in DOMAIN_WHITELIST:
        return {"domain": domain, **DOMAIN_WHITELIST[domain]}
    for wl_domain, info in DOMAIN_WHITELIST.items():
        if domain.endswith("." + wl_domain) or wl_domain in domain:
            return {"domain": domain, **info}
    return None


class VisualSearchRequest(BaseModel):
    image: str  # Base64-encoded Bild


@app.post("/api/v2/visual-search")
async def visual_search(req: VisualSearchRequest):
    """Visuelle Artefakt-Suche via Google Cloud Vision Web Detection."""

    if not GOOGLE_VISION_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_VISION_API_KEY nicht konfiguriert")

    # 1. Google Vision API aufrufen
    payload = {
        "requests": [{
            "image": {"content": req.image},
            "features": [{"type": "WEB_DETECTION", "maxResults": 50}],
            "imageContext": {"webDetectionParams": {"includeGeoResults": True}},
        }]
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}",
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Google Vision API Fehler: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Google Vision API nicht erreichbar: {str(e)}")

    if data.get("responses", [{}])[0].get("error"):
        raise HTTPException(status_code=502, detail=data["responses"][0]["error"].get("message", "Unbekannt"))

    wd = data["responses"][0].get("webDetection", {})

    # 2. Alle Ergebnisse strukturiert sammeln
    # — Seiten mit Bildern (reichhaltigste Quelle)
    page_results = []
    for page in wd.get("pagesWithMatchingImages", []):
        page_url = page.get("url", "")
        page_title = page.get("pageTitle", "")
        page_score = page.get("score", 0)
        # Bilder auf dieser Seite sammeln
        page_images = []
        for img in page.get("fullMatchingImages", []):
            page_images.append(img.get("url", ""))
        for img in page.get("partialMatchingImages", []):
            page_images.append(img.get("url", ""))
        page_results.append({
            "page_url": page_url,
            "page_title": page_title,
            "score": page_score,
            "images": page_images[:3],  # Max 3 Bilder pro Seite
        })

    # — Direkte Bild-URLs
    full_images = [img.get("url", "") for img in wd.get("fullMatchingImages", [])]
    partial_images = [img.get("url", "") for img in wd.get("partialMatchingImages", [])]
    similar_images = [img.get("url", "") for img in wd.get("visuallySimilarImages", [])]

    # 3. Alle URLs sammeln für Domain-Filter
    all_urls = []
    for p in page_results:
        all_urls.append({"url": p["page_url"], "type": "Seite", "title": p["page_title"],
                         "score": p["score"], "images": p["images"]})
    for u in full_images:
        all_urls.append({"url": u, "type": "Exakt", "title": "", "score": 0, "images": [u]})
    for u in partial_images:
        all_urls.append({"url": u, "type": "Teilweise", "title": "", "score": 0, "images": [u]})
    for u in similar_images:
        all_urls.append({"url": u, "type": "Ähnlich", "title": "", "score": 0, "images": [u]})

    # 4. Domain-Filter — Museums-Treffer anreichern
    museum_hits = []
    other_hits = []
    seen_urls = set()

    for item in all_urls:
        url = item["url"]
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        match = _match_domain(url)
        if match:
            # Bild-URL bestimmen: Entweder von der Seite oder die URL selbst
            image_url = ""
            if item["images"]:
                image_url = item["images"][0]
            elif url.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                image_url = url

            museum_hits.append({
                "url": url,
                "type": item["type"],
                "title": item.get("title", ""),
                "museum": match["name"],
                "tier": match["tier"],
                "flag": match["flag"],
                "domain": match["domain"],
                "image_url": image_url,
                "images": item.get("images", [])[:3],
            })
        else:
            try:
                domain = urlparse(url).hostname
            except Exception:
                domain = ""
            image_url = ""
            if item["images"]:
                image_url = item["images"][0]
            elif url.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                image_url = url
            other_hits.append({
                "url": url, "type": item["type"], "domain": domain,
                "title": item.get("title", ""),
                "image_url": image_url,
            })

    # 5. Visuell ähnliche Bilder separat (immer interessant)
    visual_similar = []
    for u in similar_images[:12]:
        if not u:
            continue
        wl = _match_domain(u)
        visual_similar.append({
            "image_url": u,
            "domain": urlparse(u).hostname if not wl else wl["domain"],
            "museum": wl["name"] if wl else "",
            "is_museum": bool(wl),
            "tier": wl["tier"] if wl else 0,
            "flag": wl["flag"] if wl else "",
        })

    # 6. Web Entities & Best Guess
    entities = [
        {"description": e.get("description", ""), "score": round(e.get("score", 0), 3),
         "entity_id": e.get("entityId", "")}
        for e in wd.get("webEntities", [])
        if e.get("description") and e.get("score", 0) > 0.3
    ]
    entities.sort(key=lambda x: -x["score"])
    entities = entities[:10]

    best_guess = [l.get("label", "") for l in wd.get("bestGuessLabels", [])]

    total = len(seen_urls)
    hit_rate = round(len(museum_hits) / total * 100, 1) if total > 0 else 0

    return {
        "best_guess": best_guess,
        "entities": entities,
        "museum_hits": museum_hits,
        "visual_similar": visual_similar,
        "other_count": len(other_hits),
        "total_urls": total,
        "museum_count": len(museum_hits),
        "hit_rate": hit_rate,
    }


@app.get("/api/v2/status")
async def v2_status():
    """Zeigt ob die visuelle Suche konfiguriert ist."""
    return {
        "visual_search_enabled": bool(GOOGLE_VISION_API_KEY),
        "whitelist_domains": len(DOMAIN_WHITELIST),
        "version": "2.0.0",
    }


@app.get("/api/v2/debug")
async def v2_debug():
    """Testet die Google Vision API Verbindung direkt."""
    key = GOOGLE_VISION_API_KEY
    if not key:
        return {"error": "GOOGLE_VISION_API_KEY nicht gesetzt"}

    # Minimaler Test: 1x1 weisses Pixel als Base64
    test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

    payload = {
        "requests": [{
            "image": {"content": test_image},
            "features": [{"type": "WEB_DETECTION", "maxResults": 1}],
        }]
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://vision.googleapis.com/v1/images:annotate?key={key}",
                json=payload,
                timeout=15.0,
            )
            return {
                "status_code": resp.status_code,
                "key_prefix": key[:10] + "...",
                "response": resp.json(),
            }
    except Exception as e:
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "key_prefix": key[:10] + "...",
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
