FROM python:3.13.5-slim

WORKDIR /app

# MODIFIED: Added git-lfs to the install command and ran git lfs install
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# MODIFIED: Replaced multiple COPY lines with one to ensure all files,
# including the data folder with LFS pointers, are included.
COPY . .

# ADDED: This is the crucial command to download the actual LFS files.
RUN git lfs pull

# Your original pip install command (no changes)
RUN pip3 install -r requirements.txt

# Your original EXPOSE and HEALTHCHECK commands (no changes)
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# MODIFIED: Changed to app.py to match our project's standard file name.
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]