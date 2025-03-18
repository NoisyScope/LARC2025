# Use the lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/luxonis/depthai-python.git

# Set the working directory to the examples folder
WORKDIR /app/depthai-python/examples

# Install the required dependencies
RUN python3 install_requirements.py

# Default command
CMD ["python3"]