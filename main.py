from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
import os
from typing import Optional, List
from datetime import datetime

app = FastAPI(title="ArchaeoFinder API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# API CONFIGURATION - Add your API keys as environment variables
# =============================================================================

API_KEYS = {
    "europeana": os.getenv("EUROPEANA_API_KEY", ""),
    "smithsonian": os.getenv("SMITHSONIAN_API_KEY", ""),
    "harvard": os.getenv("HARVARD_API_KEY", ""),
    "rijksmuseum": os.getenv("RIJKSMUSEUM_API_KEY", ""),
    "met": os.getenv("MET_API_KEY", ""),  # Met is free, no key needed
    "pas": os.getenv("PAS_API_KEY", ""),  # Portable Antiquities Scheme
    "british_museum": os.getenv("BRITISH_MUSEUM_API_KEY", ""),
    "victoria_albert": os.getenv("VA_API_KEY", ""),
}

# Track which APIs are enabled
def get_enabled_apis():
    enabled = []
    # Europeana - needs key
    if API_KEYS["europeana"]:
        enabled.append("europeana")
    # Smithsonian - needs key
    if API_KEYS["smithsonian"]:
        enabled.append("smithsonian")
    # Harvard - needs key
    if API_KEYS["harvard"]:
        enabled.append("harvard")
    # Rijksmuseum - needs key
    if API_KEYS["rijksmuseum"]:
        enabled.append("rijksmuseum")
    # Met Museum - FREE, no key needed
    enabled.append("met")
    # PAS UK - needs key
    if API_KEYS["pas"]:
        enabled.append("pas")
    # British Museum - check if available
    if API_KEYS["british_museum"]:
        enabled.append("british_museum")
    # V&A Museum
    if API_KEYS["victoria_albert"]:
        enabled.append("victoria_albert")
    return enabled


# =============================================================================
# EUROPEANA API
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


# =============================================================================
# METROPOLITAN MUSEUM OF ART API (FREE - No key needed)
# =============================================================================
async def search_met(client: httpx.AsyncClient, query: str, limit: int = 20):
    try:
        # Search for object IDs
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
        
        # Fetch details for each object (limit concurrent requests)
        results = []
        for obj_id in object_ids[:10]:  # Limit to 10 to avoid rate limiting
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
                        "image_url": obj.get("primaryImage") or obj.get("primaryImageSmall"),
                        "source": "Metropolitan Museum",
                        "source_url": obj.get("objectURL", ""),
                        "museum": "The Metropolitan Museum of Art, New York",
                        "epoch": obj.get("objectDate", ""),
                    })
            except:
                continue
        return results
    except Exception as e:
        print(f"Met Museum error: {e}")
        return []


# =============================================================================
# SMITHSONIAN OPEN ACCESS API
# =============================================================================
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
            
            results.append({
                "id": row.get("id", ""),
                "title": desc.get("title", {}).get("content", "Unbekannt"),
                "image_url": img,
                "source": "Smithsonian",
                "source_url": desc.get("record_link", ""),
                "museum": desc.get("unit_name", "Smithsonian Institution"),
            })
        return results
    except Exception as e:
        print(f"Smithsonian error: {e}")
        return []


# =============================================================================
# RIJKSMUSEUM API
# =============================================================================
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
            results.append({
                "id": item.get("objectNumber", ""),
                "title": item.get("title", "Unbekannt"),
                "image_url": item.get("webImage", {}).get("url", ""),
                "source": "Rijksmuseum",
                "source_url": item.get("links", {}).get("web", ""),
                "museum": "Rijksmuseum, Amsterdam",
            })
        return results
    except Exception as e:
        print(f"Rijksmuseum error: {e}")
        return []


# =============================================================================
# HARVARD ART MUSEUMS API
# =============================================================================
async def search_harvard(client: httpx.AsyncClient, query: str, limit: int = 20):
    if not API_KEYS["harvard"]:
        return []
    
    try:
        params = {
            "apikey": API_KEYS["harvard"],
            "q": query,
            "size": limit,
            "hasimage": 1,
            "classification": "Archaeological Objects|Vessels|Tools and Equipment"
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
            results.append({
                "id": f"harvard_{item.get('id', '')}",
                "title": item.get("title", "Unbekannt"),
                "image_url": item.get("primaryimageurl", ""),
                "source": "Harvard Art Museums",
                "source_url": item.get("url", ""),
                "museum": "Harvard Art Museums",
                "epoch": item.get("dated", ""),
            })
        return results
    except Exception as e:
        print(f"Harvard error: {e}")
        return []


# =============================================================================
# PORTABLE ANTIQUITIES SCHEME (UK) API
# =============================================================================
async def search_pas(client: httpx.AsyncClient, query: str, limit: int = 20):
    if not API_KEYS["pas"]:
        return []
    
    try:
        params = {
            "q": query,
            "rows": limit,
            "format": "json"
        }
        headers = {"Authorization": f"Token {API_KEYS['pas']}"} if API_KEYS["pas"] else {}
        
        response = await client.get(
            "https://finds.org.uk/database/search/results",
            params=params,
            headers=headers,
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("results", []):
            results.append({
                "id": f"pas_{item.get('id', '')}",
                "title": item.get("broadperiod", "") + " " + item.get("objecttype", ""),
                "image_url": item.get("thumbnail", ""),
                "source": "Portable Antiquities Scheme",
                "source_url": f"https://finds.org.uk/database/artefacts/record/id/{item.get('id', '')}",
                "museum": "PAS UK",
                "epoch": item.get("broadperiod", ""),
            })
        return results
    except Exception as e:
        print(f"PAS error: {e}")
        return []


# =============================================================================
# VICTORIA & ALBERT MUSEUM API
# =============================================================================
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
            
            results.append({
                "id": f"va_{item.get('systemNumber', '')}",
                "title": item.get("_primaryTitle", "Unbekannt"),
                "image_url": img,
                "source": "Victoria & Albert Museum",
                "source_url": f"https://collections.vam.ac.uk/item/{item.get('systemNumber', '')}",
                "museum": "Victoria & Albert Museum, London",
                "epoch": item.get("_primaryDate", ""),
            })
        return results
    except Exception as e:
        print(f"V&A error: {e}")
        return []


# =============================================================================
# MAIN SEARCH ENDPOINT
# =============================================================================
@app.get("/")
async def root():
    enabled = get_enabled_apis()
    return {
        "name": "ArchaeoFinder API",
        "version": "2.0.0",
        "status": "online",
        "enabled_apis": enabled,
        "total_sources": len(enabled),
        "available_apis": [
            {"name": "europeana", "needs_key": True, "enabled": "europeana" in enabled},
            {"name": "met", "needs_key": False, "enabled": "met" in enabled},
            {"name": "smithsonian", "needs_key": True, "enabled": "smithsonian" in enabled},
            {"name": "rijksmuseum", "needs_key": True, "enabled": "rijksmuseum" in enabled},
            {"name": "harvard", "needs_key": True, "enabled": "harvard" in enabled},
            {"name": "pas", "needs_key": True, "enabled": "pas" in enabled},
            {"name": "victoria_albert", "needs_key": False, "enabled": True},
        ]
    }


@app.get("/api/search")
async def search(
    q: str = Query(..., description="Search query"),
    sources: Optional[str] = Query(None, description="Comma-separated list of sources to search"),
    limit: int = Query(20, ge=1, le=50, description="Results per source"),
    epoch: Optional[str] = None,
    region: Optional[str] = None
):
    enabled_apis = get_enabled_apis()
    
    # Filter sources if specified
    if sources:
        requested = [s.strip().lower() for s in sources.split(",")]
        search_sources = [s for s in requested if s in enabled_apis]
    else:
        search_sources = enabled_apis
    
    # Modify query based on filters
    search_query = q
    if epoch and epoch != "Alle Epochen":
        search_query += f" {epoch}"
    if region and region != "Alle Regionen":
        search_query += f" {region}"
    
    async with httpx.AsyncClient() as client:
        tasks = []
        source_names = []
        
        for source in search_sources:
            if source == "europeana":
                tasks.append(search_europeana(client, search_query, limit))
                source_names.append("Europeana")
            elif source == "met":
                tasks.append(search_met(client, search_query, limit))
                source_names.append("Metropolitan Museum")
            elif source == "smithsonian":
                tasks.append(search_smithsonian(client, search_query, limit))
                source_names.append("Smithsonian")
            elif source == "rijksmuseum":
                tasks.append(search_rijksmuseum(client, search_query, limit))
                source_names.append("Rijksmuseum")
            elif source == "harvard":
                tasks.append(search_harvard(client, search_query, limit))
                source_names.append("Harvard Art Museums")
            elif source == "pas":
                tasks.append(search_pas(client, search_query, limit))
                source_names.append("PAS UK")
            elif source == "victoria_albert":
                tasks.append(search_va(client, search_query, limit))
                source_names.append("V&A Museum")
        
        # Execute all searches in parallel
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Combine results
    combined = []
    sources_searched = []
    
    for i, results in enumerate(all_results):
        if isinstance(results, list):
            combined.extend(results)
            if results:
                sources_searched.append(source_names[i])
    
    # Filter out items without images
    combined = [r for r in combined if r.get("image_url")]
    
    return {
        "query": q,
        "total_results": len(combined),
        "sources_searched": sources_searched,
        "results": combined
    }


@app.get("/api/sources")
async def get_sources():
    """Get list of all available and enabled sources"""
    return {
        "sources": [
            {
                "id": "europeana",
                "name": "Europeana",
                "description": "European cultural heritage",
                "needs_key": True,
                "enabled": bool(API_KEYS["europeana"]),
                "key_name": "EUROPEANA_API_KEY",
                "register_url": "https://pro.europeana.eu/page/get-api"
            },
            {
                "id": "met",
                "name": "Metropolitan Museum of Art",
                "description": "New York, USA",
                "needs_key": False,
                "enabled": True,
                "key_name": None,
                "register_url": None
            },
            {
                "id": "smithsonian",
                "name": "Smithsonian Institution",
                "description": "Washington D.C., USA",
                "needs_key": True,
                "enabled": bool(API_KEYS["smithsonian"]),
                "key_name": "SMITHSONIAN_API_KEY",
                "register_url": "https://api.data.gov/signup/"
            },
            {
                "id": "rijksmuseum",
                "name": "Rijksmuseum",
                "description": "Amsterdam, Netherlands",
                "needs_key": True,
                "enabled": bool(API_KEYS["rijksmuseum"]),
                "key_name": "RIJKSMUSEUM_API_KEY",
                "register_url": "https://data.rijksmuseum.nl/object-metadata/api/"
            },
            {
                "id": "harvard",
                "name": "Harvard Art Museums",
                "description": "Cambridge, USA",
                "needs_key": True,
                "enabled": bool(API_KEYS["harvard"]),
                "key_name": "HARVARD_API_KEY",
                "register_url": "https://harvardartmuseums.org/collections/api"
            },
            {
                "id": "pas",
                "name": "Portable Antiquities Scheme",
                "description": "UK archaeological finds",
                "needs_key": True,
                "enabled": bool(API_KEYS["pas"]),
                "key_name": "PAS_API_KEY",
                "register_url": "https://finds.org.uk/info/api"
            },
            {
                "id": "victoria_albert",
                "name": "Victoria & Albert Museum",
                "description": "London, UK",
                "needs_key": False,
                "enabled": True,
                "key_name": None,
                "register_url": None
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
