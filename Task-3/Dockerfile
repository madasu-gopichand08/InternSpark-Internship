# ============================================================
# DRIVER DROWSINESS DETECTION API
# Dockerfile
# ============================================================

FROM python:3.11-slim

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure Python output appears immediately
ENV PYTHONUNBUFFERED=1

# Work directory inside container
WORKDIR /app


# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# PYTHON DEPENDENCIES
# ============================================================

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# ============================================================
# APPLICATION FILES
# ============================================================

COPY app.py .

COPY model ./model


# ============================================================
# API PORT
# ============================================================

EXPOSE 8000


# ============================================================
# START FASTAPI
# ============================================================

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]