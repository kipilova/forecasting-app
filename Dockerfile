FROM python:3.11-slim

# Установка системных библиотек, необходимых для сборки PyWavelets и pmdarima
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    gfortran \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 80
CMD ["streamlit", "run", "app.py"]