FROM postgres:16-alpine

RUN apk add --no-cache python3 py3-pip python3-dev gcc musl-dev postgresql-dev libffi-dev

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY models.py .
COPY init_db.py .

RUN printf "#!/bin/sh\n/opt/venv/bin/python /app/init_db.py" > /docker-entrypoint-initdb.d/01_init_schema.sh \
    && chmod +x /docker-entrypoint-initdb.d/01_init_schema.sh