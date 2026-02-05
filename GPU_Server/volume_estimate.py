import cv2
import numpy as np
import json
import os

# Fixed camera parameters (same for all scenes)
FX = 2268.36
FY = 2312.0
BASELINE = 0.222126

def compute_disparity(left_img, right_img):
    """Calculate disparity map using StereoSGBM"""
    if left_img.shape != right_img.shape:
        right_img = cv2.resize(right_img, (left_img.shape[1], left_img.shape[0]))
    
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    window_size = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*12,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disparity

def compute_depth(disparity, fx, baseline):
    """Calculate depth from disparity"""
    disparity[disparity <= 0] = 0.1
    depth = (fx * baseline) / disparity
    return depth

def estimate_object_volume(bbox, depth_map, fx, fy):
    """
    Estimate object surface from bounding box
    bbox: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    
    depth_region = depth_map[y1:y2, x1:x2]
    distance_m = np.median(depth_region[depth_region > 0])
    
    width_px = x2 - x1
    height_px = y2 - y1
    
    width_m = (width_px * distance_m) / fx
    height_m = (height_px * distance_m) / fy
    surface_m2 = width_m * height_m
    
    return width_m, height_m, distance_m, surface_m2