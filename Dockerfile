# ArchaeoFinder Backend Phase 2 - FIXED
# ======================================

FROM python:3.11-slim

WORKDIR /app

# System-Abhängigkeiten
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Pip upgraden
RUN pip install --upgrade pip setuptools wheel

# PyTorch CPU-Version installieren (kompatibel)
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Restliche Abhängigkeiten (ohne torch, da schon installiert)
COPY requirements.txt .
RUN grep -v "^torch" requirements.txt | pip install --no-cache-dir -r /dev/stdin

# Anwendungscode
COPY main.py .

# Verzeichnis für ChromaDB
RUN mkdir -p /app/chroma_data

# Port
EXPOSE 8080

# Start mit wenig RAM
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
