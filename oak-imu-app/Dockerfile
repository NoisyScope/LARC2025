FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for DepthAI
RUN apt-get update && apt-get install -y \
    libusb-1.0-0 \
    udev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY imu_processor.py .
COPY templates/ templates/

# Create directory for logs
RUN mkdir -p /app/logs

# Set up udev rules for OAK devices
RUN echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' > /etc/udev/rules.d/80-movidius.rules

# Expose port for web interface
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
