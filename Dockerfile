# Multi-stage build for full-stack app on Hugging Face Spaces
FROM python:3.12-slim

# Install Node.js, nginx, git, and system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    nginx \
    git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend
COPY backend/ ./backend/

# Copy frontend
COPY frontend/ ./frontend/

# Install frontend dependencies and build
WORKDIR /app/frontend
RUN npm install && npm run build

# Create data directory
RUN mkdir -p /app/data/chroma_db

WORKDIR /app

# Create nginx configuration
RUN echo 'server {\n\
    listen 7860;\n\
    server_name _;\n\
\n\
    # Frontend\n\
    location / {\n\
        root /app/frontend/dist;\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
\n\
    # Backend API proxy\n\
    location /api/ {\n\
        proxy_pass http://localhost:8000/api/;\n\
        proxy_http_version 1.1;\n\
        proxy_set_header Upgrade $http_upgrade;\n\
        proxy_set_header Connection "upgrade";\n\
        proxy_set_header Host $host;\n\
        proxy_set_header X-Real-IP $remote_addr;\n\
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n\
        proxy_set_header X-Forwarded-Proto $scheme;\n\
        proxy_buffering off;\n\
        proxy_cache off;\n\
        client_max_body_size 100M;\n\
    }\n\
}' > /etc/nginx/sites-available/default

# Expose port 7860 (HF Spaces standard)
EXPOSE 7860

# Create startup script
RUN echo '#!/bin/bash\n\
nginx\n\
python backend/api/main.py' > /app/start.sh && chmod +x /app/start.sh

ENV PYTHONUNBUFFERED=1

CMD ["/app/start.sh"]
