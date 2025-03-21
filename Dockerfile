# Use the lightweight Python image
FROM luxonis/depthai-library

# Set the working directory
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# # Clone the repository
# RUN git clone https://github.com/luxonis/depthai-python.git

# # Set the working directory to the examples folder
# WORKDIR /app/depthai-python/examples

# Install the required dependencies
RUN python3 -m pip install -U pip

# Install the required dependencies
RUN python3 -m pip install --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/ depthai

# Grant access to USB devices
RUN apt-get update && apt-get install -y udev && rm -rf /var/lib/apt/lists/*

# Set environment variable for display forwarding
ENV DISPLAY=$DISPLAY

# Mount X11 socket for GUI applications
VOLUME /tmp/.X11-unix

# Default command to run the rgb_preview.py script
CMD ["python3", "/depthai-python/examples/ColorCamera/rgb_preview.py"]