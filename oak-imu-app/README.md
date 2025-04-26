# OAK-D Pro IMU Orientation Visualization

This application reads IMU data from a Luxonis OAK-D Pro camera, applies filtering to estimate orientation, and displays the results in a web-based visualization.

## Requirements

- Luxonis OAK-D Pro camera
- Raspberry Pi 4
- Docker

## Running the Application

1. Build the Docker container:
   ```
   docker build -t oak-imu-app .
   ```

2. Run the container with privileged access (required for camera access):
   ```
   docker run --privileged -p 5000:5000 oak-imu-app
   ```

3. Access the visualization in your web browser:
   ```
   http://localhost:5000
   ```

## Project Structure

- `app.py`: Main application code
- `imu_processor.py`: IMU data processing and filtering
- `templates/`: Web interface templates
- `static/`: Static assets for the web interface
- `Dockerfile`: Container configuration
- `requirements.txt`: Python dependencies
