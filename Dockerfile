# Use an official lightweight Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libexpat1 \
    gdal-bin libgdal-dev libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

      
      
      
# Set work directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Now copy the entire project, including supres-40k-swir/
COPY . .

# Set entrypoint to run your tile processing script
ENTRYPOINT ["python", "getTiles.py"]
CMD []

