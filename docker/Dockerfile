FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps . && useradd --create-home --uid 10001 app && chown -R app /app
USER app
ENTRYPOINT ["trajetl"]
CMD ["--help"]
