# ArchaeoFinder Backend - Dockerfile für DigitalOcean
# ================================================

# Python 3.11 als Basis (stabil und schnell)
FROM python:3.11-slim

# Arbeitsverzeichnis setzen
WORKDIR /app

# System-Abhängigkeiten installieren
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python-Abhängigkeiten kopieren und installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Anwendungscode kopieren
COPY main.py .

# Port freigeben (DigitalOcean erwartet 8080)
EXPOSE 8080

# Server starten
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
