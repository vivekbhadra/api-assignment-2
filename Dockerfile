# === SmartLegal Rental Assistant - Optimised Dockerfile ===

# Base image: lightweight but compatible with Streamlit & AI SDKs
FROM python:3.10-slim

# Prevent Python bytecode + buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install essential system dependencies only
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirement installation layer
# (copy only requirements first to leverage Docker layer caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code (minimal files)
COPY app.py README.md ./

# Environment variables (Streamlit default port)
ENV PORT=8501

# Expose port and run Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]

