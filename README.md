# Vehicule-Embedded Obstacle Detection (Demo Branch)

This branch implements a **Buffered Data Transmission** system. It allows the Raspberry Pi to synchronize stereo image pairs with GPS data, store them locally during network instability, and transmit them to a GPU server in the background.

## System Architecture

The demo uses a **Producer-Consumer** model:

1. **Producer (`main_demo.py`)**: 
   - Reads the GPS CSV and the Stereo (Left/Right) images.
   - Synchronizes data based on frame names.
   - Packages them into "tasks" inside the `temp_buffer/` folder.

2. **Consumer (`sender_worker.py`)**: 
   - Monitors the `temp_buffer/` continuously.
   - Attempts to send data via the **ngrok** tunnel.
   - **Important:** Files are only deleted from the Raspberry Pi once the GPU server returns a "Success" (HTTP 200) response.



##  Directory Structure

- `main_demo.py` : Main logic for data synchronization and buffering.
- `sender_worker.py` : Background job for network transmission.
- `demo_data/` : Source folder for your dataset (Images + GPS CSV).
- `temp_buffer/` : Auto-generated queue for pending transmissions.

## How to Run the Demo

### 1. Setup
Ensure your virtual environment is active:
```bash
source my_env/bin/activate
