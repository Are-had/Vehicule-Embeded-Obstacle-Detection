# Obstacle Detection Project for Autonomous Vehicles

This repository gathers the development work related to the detection and classification of road obstacles using the YOLOv8 architecture.

## Analysis Status

The project is currently undergoing an in-depth analysis on a high-performance GPU server. This phase allows for processing larger data volumes and refining weight precision. More performant models, optimized for small object detection and false positive reduction, are currently being deployed.

## Repository Structure

- datasets/: YAML configuration files for data management ( Road Damage, Lost and Found.....).
- detection-and-classification/:
    - training/: Training scripts for YOLOv8 models (Nano and Large versions).
    - inference/: Test notebooks for validation on real-world data streams.
    - evaluation/: Detailed metrics analysis (mAP, confusion matrices) and a samples folder containing detection result highlights.
    - utils/: Annotation format conversion scripts and preprocessing tools.

## Main Objectives

1. Analysis and detection of road debris.
2. Recognition of specific signage (Stop signs, Roadworks).
3. Identification of road anomalies (Speed bumps).
4. Server-side performance benchmarking for future embedded system integration.

## Installation

To configure the environment:
pip install -r requirements.txt

Output data and visual performance evidence are listed in the evaluation/samples/ directory.

