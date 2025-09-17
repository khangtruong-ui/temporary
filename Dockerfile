
FROM python:3.11-slim

# set workdir
WORKDIR /app
# copy files
COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# install python deps
RUN apt-get update && apt-get install -y git

RUN sh setup.sh

EXPOSE 8000
CMD ["python", "app.py"]
