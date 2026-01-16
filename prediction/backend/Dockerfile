FROM python:3.11-slim

WORKDIR /app

# Install system deps (libpcap optional for scapy sniffing)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Use gunicorn to run the Flask app (exported `app` in ServerML_DDoS.py)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "ServerML_DDoS:app", "--workers", "1"]
