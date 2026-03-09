# ArchaeoFinder Backend

[![Version](https://img.shields.io/badge/version-3.0.1-blue.svg)](https://archaeofinder.de)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Backend für die KI-gestützte Archäologie-Fundbestimmung

## 🏺 Über ArchaeoFinder

ArchaeoFinder ist eine innovative Plattform zur automatischen Identifikation archäologischer Artefakte mittels Künstlicher Intelligenz. Dieses Repository enthält die Backend-API und Datenbank-Layer.

**Live:** [archaeofinder.de](https://archaeofinder.de)

## ✨ Features

- 🔍 **Vector Search** – 768-dimensionale Ähnlichkeitssuche
- 🖼️ **Bildverarbeitung** – Automatische Optimierung und Normalisierung
- 🗄️ **Datenbank** – Museumsexponate und Metadaten
- 🔐 **API-Sicherheit** – Rate Limiting und Authentifizierung
- 📊 **Analytics** – Nutzungsstatistiken und Monitoring

## 🚀 Schnellstart

### Voraussetzungen

- Python 3.9+
- PostgreSQL 13+
- Redis (für Caching)

### Installation

```bash
# Repository klonen
git clone https://github.com/1ucky-1uk3/archaeofinder-backend.git
cd archaeofinder-backend

# Virtual Environment erstellen
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Dependencies installieren
pip install -r requirements.txt

# Umgebungsvariablen konfigurieren
cp .env.example .env
# .env Datei bearbeiten

# Datenbank migrieren
python manage.py migrate

# Server starten
python manage.py runserver
```

Die API ist dann unter `http://localhost:8000` erreichbar.

## 🏗️ Architektur

```
archaeofinder-backend/
├── api/                    # REST API Endpoints
│   ├── uploads/           # Bild-Upload Handler
│   ├── search/            # Vector Search API
│   └── filters/           # Filter-Metadaten
├── core/                   # Business Logic
│   ├── models.py          # Datenbank-Modelle
│   ├── embeddings.py      # Vector Embeddings
│   └── similarity.py      # Ähnlichkeitsberechnung
├── pipelines/              # ML Pipeline Integration
└── tests/                  # Unit & Integration Tests
```

## 🔌 API Endpoints

### Upload
```http
POST /api/upload
Content-Type: multipart/form-data

image: <file>
metadata: {"period": "roman", "region": "germany"}
```

### Search
```http
GET /api/search?embedding=[...]&limit=10
Authorization: Bearer <token>
```

### Filters
```http
GET /api/filters
# Returns: {"periods": [...], "regions": [...], "materials": [...]}
```

## 🛠️ Tech Stack

- **Framework:** Django / FastAPI
- **Datenbank:** PostgreSQL + pgvector
- **Cache:** Redis
- **ML Integration:** REST API zu Pipelines
- **Deployment:** Docker + Docker Compose

## 🔧 Umgebungsvariablen

```env
DEBUG=False
SECRET_KEY=your-secret-key
DATABASE_URL=postgresql://user:pass@localhost/archaeofinder
REDIS_URL=redis://localhost:6379/0
ML_PIPELINE_URL=http://localhost:5000
MAX_UPLOAD_SIZE=10485760  # 10MB
```

## 🧪 Tests

```bash
# Alle Tests ausführen
pytest

# Mit Coverage
pytest --cov=api --cov-report=html

# Spezifischen Test
pytest tests/test_search.py -v
```

## 📋 Roadmap

- [ ] OAuth2 Authentifizierung
- [ ] Rate Limiting pro User
- [ ] Batch-Upload API
- [ ] Webhook Support
- [ ] Admin Dashboard

## 🤝 Mitwirken

1. Fork erstellen
2. Feature-Branch: `git checkout -b feature/api-endpoint`
3. Committen: `git commit -am 'Neuer Endpoint'`
4. Pushen: `git push origin feature/api-endpoint`
5. Pull Request erstellen

## 📄 Lizenz

MIT License – siehe [LICENSE](LICENSE)

## 🔗 Verwandte Projekte

- [Frontend](https://github.com/1ucky-1uk3/archaeofinder-frontend)
- [ML Pipelines](https://github.com/1ucky-1uk3/archaeofinder-pipelines)

---

**ArchaeoFinder** – KI für die Archäologie 🏺
