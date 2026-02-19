FROM python:3.11-slim

WORKDIR /app

# System-Abhängigkeiten installieren
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pip aktualisieren
RUN pip install --upgrade pip setuptools wheel

# PyTorch CPU-Version installieren
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Requirements kopieren und installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Anwendung kopieren
COPY main.py .

# Verzeichnis für ChromaDB erstellen
RUN mkdir -p /app/chroma_data && chmod 777 /app/chroma_data

# Port freigeben
EXPOSE 8080

# Längerer Timeout für Startup
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "120"]
