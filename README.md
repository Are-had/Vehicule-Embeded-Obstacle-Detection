# 🚗 Vehicle-Embedded Obstacle Detection (Demo System)

This repository contains the full end-to-end system for real-time obstacle detection and geolocation. The project is organized into a **Client-Server** architecture, separating the embedded vehicle logic from the high-performance inference engine.



## 🏗️ System Architecture

The system uses an **Asynchronous Buffer-Sender** model. This design ensures that data captured by the vehicle is never lost, even if the 4G/Wi-Fi connection is unstable during transit.

1.  **Vehicle (Raspberry Pi):** Acts as the **Producer**. It synchronizes stereo images with GPS coordinates and stores "tasks" in a local temporary buffer.
2.  **Network Tunnel (ngrok):** Acts as the **Bridge**. It provides a secure, public URL for the Raspberry Pi to reach the GPU server.
3.  **Inference Server (GPU):** Acts as the **Consumer**. It receives the data, runs a multi-model ensemble for detection, and logs the geolocated results.

---

## 📂 Project Structure

### 📡 [Raspi_Embedded/](./Raspi_Embedded)
Code running on the vehicle's hardware (Raspberry Pi 4).
* **`main_demo.py`**: Reads the dataset, synchronizes Left/Right images with GPS, and creates buffered tasks.
* **`sender_worker.py`**: A background job that monitors the buffer and handles the HTTP transmission to the GPU via ngrok.
* **`demo_data/`**: Directory containing the demo datasets (images and GPS CSV).

### 🧠 [GPU_Server/](./GPU_Server)
AI inference and data logging logic running on the high-performance server.
* **`gpu_server.py`**: A FastAPI-based server that handles incoming data requests.
* **`processor.py`**: The core Inference Engine that runs 7 specialized YOLO models (Road Damage, Debris, Stop Signs, etc.) and draws cumulative bounding boxes.

---

## 🚀 How to Run the Demo

### 1. Start the GPU Server
On the GPU machine, ensure your environment is ready and run:
```bash
python GPU_Server/gpu_server.py
