FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn==0.28.0 \
    pydantic==2.6.4 \
    minio==7.2.5 \
    redis==5.0.3 \
    httpx==0.27.0 \
    pandas==2.2.1 \
    pyarrow==15.0.0 \
    matplotlib==3.8.3 \
    statsmodels==0.14.1 \
    scipy==1.12.0
RUN mkdir -p /workspace/sandboxes
RUN useradd -u 1001 -m sandbox_user && \
    chown -R sandbox_user:sandbox_user /app /workspace/sandboxes
USER sandbox_user
COPY --chown=sandbox_user:sandbox_user Minio.py /app/Minio.py
COPY --chown=sandbox_user:sandbox_user Redis.py /app/Redis.py
COPY --chown=sandbox_user:sandbox_user Main.py /app/main.py
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]