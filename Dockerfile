# ArchaeoFinder Backend Phase 2 - Mit CLIP & ChromaDB
# ===================================================

FROM python:3.11-slim

# Arbeitsverzeichnis
WORKDIR /app

# System-Abhängigkeiten für PyTorch und Bildverarbeitung
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Pip upgraden
RUN pip install --upgrade pip

# PyTorch separat installieren (CPU-Version, kleiner)
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Restliche Abhängigkeiten
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Anwendungscode
COPY main.py .

# Verzeichnis für ChromaDB Daten
RUN mkdir -p /app/chroma_data

# Port
EXPOSE 8080

# Weniger Worker für weniger RAM-Verbrauch
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
