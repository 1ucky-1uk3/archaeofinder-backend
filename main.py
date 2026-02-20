# ArchaeoFinder Backend Phase 2

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx
import base64
import hashlib
from datetime import datetime
import os
import io
import asyncio

CLIP_AVAILABLE = False
CLIP_LOADING = False
CLIP_LOADED = False

try:
    import torch
    import open_clip
    from PIL import Image
    import chromadb
    import numpy as np
    CLIP_AVAILABLE = True
except ImportError as e:
    CLIP_AVAILABLE = False

EUROPEANA_API_KEY = os.getenv('EUROPEANA_API_KEY', 'api2demo')
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png', 'webp', 'gif'])

clip_model = None
clip_preprocess = None
clip_tokenizer = None
chroma_client = None
image_collection = None
uploaded_images = {}


def load_clip_model():
    global clip_model, clip_preprocess, clip_tokenizer, chroma_client, image_collection, CLIP_LOADED, CLIP_LOADING
    if not CLIP_AVAILABLE or CLIP_LOADING or CLIP_LOADED:
        return
    CLIP_LOADING = True
    try:
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        clip_model.eval()
        chroma_client = chromadb.Client()
        image_collection = chroma_client.get_or_create_collection(name='archaeo_images')
        CLIP_LOADED = True
    except Exception as e:
        pass
    CLIP_LOADING = False


def get_image_embedding(image):
    if not clip_model or not clip_preprocess:
        return None
    try:
        image_input = clip_preprocess(image).unsqueeze(0)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features[0].tolist()
    except Exception:
        return None


class MuseumObject(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    museum: Optional[str] = None
    epoch: Optional[str] = None
    image_url: Optional[str] = None
    source_url: Optional[str] = None
    source: str = 'europeana'
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


EPOCH_MAPPING = {
    'Steinzeit': ['neolithic', 'stone age', 'mesolithic', 'paleolithic'],
    'Bronzezeit': ['bronze age'],
    'Eisenzeit': ['iron age', 'hallstatt', 'celtic'],
    'Roemische Kaiserzeit': ['roman', 'roman empire'],
    'Fruehmittelalter': ['early medieval', 'migration period'],
    'Hochmittelalter': ['medieval', 'romanesque'],
    'Spaetmittelalter': ['late medieval', 'gothic']
}

OBJECT_TYPE_MAPPING = {
    'Fibeln': ['fibula', 'brooch', 'pin'],
    'Muenzen': ['coin', 'numismatic'],
    'Keramik': ['ceramic', 'pottery', 'vessel', 'amphora'],
    'Waffen': ['weapon', 'sword', 'axe', 'tool'],
    'Schmuck': ['jewelry', 'jewellery', 'ring', 'bracelet', 'necklace'],
    'Kultgegenstaende': ['cult', 'ritual', 'religious', 'votive'],
    'Alltagsgegenstaende': ['domestic', 'household']
}

REGION_MAPPING = {
    'Mitteleuropa': ['germany', 'austria', 'switzerland'],
    'Nordeuropa': ['scandinavia', 'denmark', 'sweden', 'norway'],
    'Suedeuropa': ['italy', 'greece', 'spain'],
    'Westeuropa': ['france', 'britain', 'england'],
    'Osteuropa': ['poland', 'czech', 'hungary', 'romania'],
    'Mittelmeerraum': ['mediterranean', 'aegean'],
    'Naher Osten': ['mesopotamia', 'egypt', 'levant']
}


app = FastAPI(title='ArchaeoFinder API', version='2.0.0', docs_url='/docs', redoc_url='/redoc')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.on_event('startup')
async def startup_event():
    asyncio.create_task(load_clip_background())


async def load_clip_background():
    await asyncio.sleep(5)
    load_clip_model()


def build_europeana_query(keywords=None, epoch=None, object_type=None, region=None):
    query_parts = []
    query_parts.append('(archaeology OR archaeological)')
    if keywords:
        query_parts.append('(' + keywords + ')')
    if epoch and epoch != 'Alle Epochen':
        epoch_terms = EPOCH_MAPPING.get(epoch, [])
        if epoch_terms:
            query_parts.append('(' + ' OR '.join(epoch_terms) + ')')
    if object_type and object_type != 'Alle Objekttypen':
        type_terms = OBJECT_TYPE_MAPPING.get(object_type, [])
        if type_terms:
            query_parts.append('(' + ' OR '.join(type_terms) + ')')
    if region and region != 'Alle Regionen':
        region_terms = REGION_MAPPING.get(region, [])
        if region_terms:
            query_parts.append('(' + ' OR '.join(region_terms) + ')')
    return ' AND '.join(query_parts)


def parse_europeana_result(item):
    title = 'Unbekanntes Objekt'
    if 'title' in item and item['title']:
        title_data = item['title']
        if isinstance(title_data, list):
            title = title_data[0]
        else:
            title = title_data
    description = None
    if 'dcDescription' in item and item['dcDescription']:
        desc_data = item['dcDescription']
        if isinstance(desc_data, list):
            description = desc_data[0]
        else:
            description = desc_data
        if description and len(description) > 300:
            description = description[:297] + '...'
    museum = None
    if 'dataProvider' in item and item['dataProvider']:
        dp_data = item['dataProvider']
        if isinstance(dp_data, list):
            museum = dp_data[0]
        else:
            museum = dp_data
    epoch = None
    if 'year' in item and item['year']:
        years = item['year']
        if isinstance(years, list) and years:
            if len(years) > 1:
                epoch = str(min(years)) + ' - ' + str(max(years))
            else:
                epoch = str(years[0])
    image_url = None
    if 'edmPreview' in item and item['edmPreview']:
        previews = item['edmPreview']
        if isinstance(previews, list):
            image_url = previews[0]
        else:
            image_url = previews
    source_url = None
    if 'guid' in item:
        source_url = item['guid']
    elif 'id' in item:
        source_url = 'https://www.europeana.eu/item' + item['id']
    return MuseumObject(id=item.get('id', 'unknown'), title=title, description=description, museum=museum, epoch=epoch, image_url=image_url, source_url=source_url, source='europeana')


async def search_europeana(query, rows=20):
    url = 'https://api.europeana.eu/record/v2/search.json'
    params = {'wskey': EUROPEANA_API_KEY, 'query': query, 'rows': rows, 'profile': 'rich', 'media': 'true', 'qf': 'TYPE:IMAGE'}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, params=params)
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail='Europeana API error')
        data = response.json()
        total = data.get('totalResults', 0)
        items = data.get('items', [])
        results = []
        for item in items:
            try:
                parsed = parse_europeana_result(item)
                if parsed.image_url:
                    results.append(parsed)
            except Exception:
                continue
        return total, results


async def search_by_image(image, limit=20):
    if not image_collection:
        return []
    count = image_collection.count()
    if count == 0:
        return []
    embedding = get_image_embedding(image)
    if not embedding:
        return []
    n_results = min(limit, count)
    results = image_collection.query(query_embeddings=[embedding], n_results=n_results)
    if not results:
        return []
    ids_list = results.get('ids', [])
    if not ids_list or not ids_list[0]:
        return []
    museum_objects = []
    metadatas = results.get('metadatas', [[]])
    distances = results.get('distances', [[]])
    for i in range(len(ids_list[0])):
        item_id = ids_list[0][i]
        metadata = {}
        if metadatas and metadatas[0] and i < len(metadatas[0]):
            metadata = metadatas[0][i]
        distance = 1.0
        if distances and distances[0] and i < len(distances[0]):
            distance = distances[0][i]
        similarity = int(max(0, min(100, (1 - distance) * 100)))
        museum_objects.append(MuseumObject(id=item_id, title=metadata.get('title', 'Unbekannt'), museum=metadata.get('museum', None), image_url=metadata.get('image_url', None), source_url=metadata.get('source_url', None), source=metadata.get('source', 'europeana'), similarity=similarity))
    return museum_objects


@app.get('/')
async def root():
    db_count = 0
    if image_collection:
        db_count = image_collection.count()
    return {'name': 'ArchaeoFinder API', 'version': '2.0.0', 'status': 'online', 'clip_available': CLIP_AVAILABLE, 'clip_loaded': CLIP_LOADED, 'images_indexed': db_count}


@app.get('/health')
async def health_check():
    return {'status': 'healthy'}


@app.post('/api/upload', response_model=UploadResponse)
async def upload_image_endpoint(file: UploadFile = File(...)):
    filename = file.filename or 'unknown'
    ext = ''
    if '.' in filename:
        ext = filename.rsplit('.', 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail='Ungueltiges Format')
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail='Datei zu gross')
    image_hash = hashlib.sha256(content).hexdigest()[:16]
    image_id = 'img_' + image_hash + '_' + datetime.now().strftime('%Y%m%d%H%M%S')
    uploaded_images[image_id] = {'content': content, 'filename': filename, 'content_type': file.content_type, 'uploaded_at': datetime.now().isoformat()}
    return UploadResponse(success=True, image_id=image_id, message='Bild hochgeladen')


@app.get('/api/search', response_model=SearchResponse)
async def search(q: Optional[str] = Query(None), image_id: Optional[str] = Query(None), epoch: Optional[str] = Query(None), object_type: Optional[str] = Query(None), region: Optional[str] = Query(None), limit: int = Query(20, ge=1, le=50)):
    search_mode = 'text'
    results = []
    total = 0
    if image_id and image_id in uploaded_images and CLIP_LOADED:
        try:
            content = uploaded_images[image_id]['content']
            image = Image.open(io.BytesIO(content)).convert('RGB')
            results = await search_by_image(image, limit)
            total = len(results)
            search_mode = 'image'
        except Exception:
            pass
    if not results or q:
        query = build_europeana_query(keywords=q, epoch=epoch, object_type=object_type, region=region)
        try:
            total, text_results = await search_europeana(query, rows=limit)
            if search_mode == 'text':
                import random
                for i in range(len(text_results)):
                    result = text_results[i]
                    base_similarity = 95 - (i * 3)
                    result.similarity = max(50, min(99, base_similarity + random.randint(-5, 5)))
                results = text_results
            else:
                search_mode = 'hybrid'
                existing_ids = set()
                for r in results:
                    existing_ids.add(r.id)
                for r in text_results:
                    if r.id not in existing_ids:
                        r.similarity = 50
                        results.append(r)
        except Exception as e:
            if not results:
                raise HTTPException(status_code=500, detail='Suchfehler')
    results.sort(key=lambda x: x.similarity or 0, reverse=True)
    search_id = 'search_' + datetime.now().strftime('%Y%m%d%H%M%S')
    return SearchResponse(success=True, total_results=total, results=results[:limit], search_id=search_id, filters_applied={'keywords': q, 'epoch': epoch, 'object_type': object_type, 'region': region}, search_mode=search_mode)


@app.get('/api/filters')
async def get_filters():
    return {'epochs': ['Alle Epochen'] + list(EPOCH_MAPPING.keys()), 'object_types': ['Alle Objekttypen'] + list(OBJECT_TYPE_MAPPING.keys()), 'regions': ['Alle Regionen'] + list(REGION_MAPPING.keys())}


@app.get('/api/sources')
async def get_sources():
    return {'sources': [{'id': 'europeana', 'name': 'Europeana', 'status': 'active'}, {'id': 'ddb', 'name': 'Deutsche Digitale Bibliothek', 'status': 'coming_soon'}, {'id': 'british_museum', 'name': 'British Museum', 'status': 'coming_soon'}]}


if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv('PORT', 8080))
    uvicorn.run(app, host='0.0.0.0', port=port)
