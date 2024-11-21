# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set environment variables to avoid buffering and use UTF-8 encoding
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Set a working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose port for the application
EXPOSE 5000

# Default command to run the application (modify as needed)
CMD ["python", "notebooks/rosmann pridiction.py"]
