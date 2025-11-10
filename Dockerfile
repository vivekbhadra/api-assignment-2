# === SmartLegal Rental Assistant - Dockerfile ===
FROM python:3.10-slim

# Prevent Python from writing pyc files / buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system deps if needed by PyPDF2/docx/ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy code
COPY app.py README.md ./

# Install Python deps
RUN pip install --no-cache-dir streamlit google-generativeai openai python-dotenv \
    PyPDF2 python-docx torch transformers pillow

# Streamlit defaults to port 8501
EXPOSE 8501

# Entrypoint
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]

