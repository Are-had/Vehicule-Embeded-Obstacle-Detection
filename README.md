#  Vehicle Embedded Obstacle Detection

Embedded system for Raspberry Pi 5 that syncs GPS tracking with dual-camera capture for road obstacle detection.

##  Features
- **GPS Sync**: Real-time NMEA parsing (u-blox)
- **Dual Capture**: Stereo-vision Left/Right frames
- **Geotagged Data**: Auto `.csv` with coordinates per image
- **High Performance**: Local storage, no network delays

##  Structure
```
main.py          # Core execution
gps_handler.py   # GPS driver
utils.py         # Coordinate utils
gpu_client.py    # Remote GPU API
config.py        # Settings
```

##  Quick Start

**Hardware**: Connect u-blox GPS to Pi 5 UART + 2 cameras
```bash
pip install pyserial requests picamera2
python main.py
```

**Output**: `data/session_*/left_xxx.jpg + right_xxx.jpg + metadata_xxx.csv`

##  CSV Format
```csv
timestamp,latitude,longitude,altitude,speed
2024-01-15 14:23:45,48.8566,2.3522,35.0,15.2
```
