# Use official Python slim image
FROM python:3.13.5-slim

# Set working directory
WORKDIR /app

# Install dependencies for building some packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY config.py ./               # Make config.py available
COPY streamlit_app.py ./        # Streamlit entrypoint
COPY src/ ./src/                # Source code folder

# Expose Streamlit port
EXPOSE 8501

# Healthcheck for HF Spaces
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Set project root in PYTHONPATH to fix imports (config, src.*)
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
