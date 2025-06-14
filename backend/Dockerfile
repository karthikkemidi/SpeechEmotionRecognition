FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 5050

CMD ["python", "-m", "ser.main"]
