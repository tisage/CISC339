"""
Live Object Detection Demo using PyTorch and OpenCV.

This script completely decouples the Video Rendering (Main Thread) from the 
AI Inference (Background Thread). This ensures your webcam video runs flawlessly 
at 30+ FPS (zero lag, zero stutter) while the AI updates the bounding boxes 
as fast as the hardware allows (e.g., 5-10 FPS on Mac).

Features:
- Asynchronous AI inference
- Real-time smooth video playback
- Lower confidence threshold for detecting more objects
"""

import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import time
import threading
import argparse

# List of COCO dataset class names (91 classes)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# --- Global variables for Async Thread Communication ---
latest_frame_for_inference = None
latest_detections = None
lock = threading.Lock()
is_running = True
inference_fps = 0.0

def get_device():
    """Select the best available device for computation."""
    if torch.backends.mps.is_available():
        print("Hardware Match: Apple Silicon (MPS) detected. GPU acceleration enabled!")
        return torch.device("mps")  
    elif torch.cuda.is_available():
        print("Hardware Match: NVIDIA GPU (CUDA) detected. GPU acceleration enabled!")
        return torch.device("cuda") 
    else:
        print("Hardware Match: No compatible GPU found. Defaulting to CPU.")
        return torch.device("cpu")  

def inference_worker(model, device):
    """
    Background thread that constantly checks if a new frame is available.
    If so, it runs PyTorch inference and saves the latest bounding boxes.
    """
    global latest_frame_for_inference, latest_detections, is_running, inference_fps
    
    while is_running:
        frame_to_process = None
        
        # Safely grab the latest frame
        with lock:
            if latest_frame_for_inference is not None:
                frame_to_process = latest_frame_for_inference.copy()
                # Clear it so we don't process the same frame twice
                latest_frame_for_inference = None
                
        if frame_to_process is not None:
            inf_start = time.time()
            
            # Prepare image for PyTorch
            rgb_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
            tensor_frame = F.to_tensor(rgb_frame).to(device)
            
            # Run Model
            with torch.no_grad():
                predictions = model([tensor_frame])[0]
                
                # Move to CPU explicitly and immediately to free device memory
                preds_cpu = {
                    'scores': predictions['scores'].cpu().numpy(),
                    'boxes': predictions['boxes'].cpu().numpy(),
                    'labels': predictions['labels'].cpu().numpy()
                }

            # Safely store results
            with lock:
                latest_detections = preds_cpu
                
            inf_time = time.time() - inf_start
            inference_fps = 1.0 / inf_time if inf_time > 0 else 0
        else:
            # Prevent 100% CPU utilization when waiting
            time.sleep(0.005)

def main():
    parser = argparse.ArgumentParser(description="Live Object Detection Demo")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Starting camera index (0 for built-in, 1 or 2 for USB usually)")
    args = parser.parse_args()

    global latest_frame_for_inference, latest_detections, is_running
    
    device = get_device()
    print(f"Using compute device: {device}")

    # Initialize PyTorch Model
    print("Loading pre-trained model...")
    weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
    model.to(device)
    model.eval()  

    # Test camera
    target_camera_index = args.camera
    cap = cv2.VideoCapture(target_camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {target_camera_index}.")
        print("Falling back to scanning available cameras...")
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened() and cap.read()[0]:
                target_camera_index = i
                print(f"Auto-selected working camera index: {i}")
                break
        else:
            print("Fatal Error: No working cameras found.")
            return

    # Keep resolution reasonable to balance webcam speed and accuracy
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting background inference thread...")
    worker_thread = threading.Thread(target=inference_worker, args=(model, device))
    worker_thread.start()

    print("\n" + "-"*30)
    print("🎮 Controls:")
    print(" [q] : Quit the application")
    print(" [c] : Switch to next camera (e.g., from Mac to USB)")
    print("-" * 30 + "\n")
    
    # Confidence threshold to show boxes
    # IMPORTANT: Lowered from 0.5 to 0.35 so more objects will be detected!
    CONFIDENCE_THRESHOLD = 0.35 

    while True:
        main_start = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # 1. Dispatch copy of this frame to the background AI thread
        with lock:
             # Just overwrite the old one; the background thread will pick it up when ready.
             latest_frame_for_inference = frame.copy()
             # Grab whatever detection results are currently available
             current_detections = latest_detections

        display_frame = frame.copy()
        objects_detected = 0

        # 2. Draw the bounding boxes if we have any cached
        if current_detections is not None:
            scores = current_detections['scores']
            boxes = current_detections['boxes']
            labels = current_detections['labels']
            
            for i in range(len(scores)):
                if scores[i] > CONFIDENCE_THRESHOLD:
                    objects_detected += 1
                    
                    box = boxes[i].astype(int)
                    x1, y1, x2, y2 = box

                    label_idx = labels[i]
                    label_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else "Unknown"
                    label_text = f"{label_name}: {scores[i]:.2f}"

                    color = (0, 255, 0) # Green box
                    thickness = 2
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Highlight text background
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(display_frame, (x1, y1 - text_h - baseline), (x1 + text_w, y1), color, -1)
                    
                    # Add Text
                    cv2.putText(display_frame, label_text, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 3. Render Status UI on top
        ui_fps = 1.0 / (time.time() - main_start)
        # Show both the UI Smoothness (FPS) and the AI's Brain Speed (Inference FPS)
        status_text_1 = f"Camera/Video FPS: {ui_fps:.1f} (Smooth)"
        status_text_2 = f"AI Backend FPS: {inference_fps:.1f} | Objects: {objects_detected}"
        
        cv2.putText(display_frame, status_text_1, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display_frame, status_text_2, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 4. Show Window
        cv2.imshow('Live Object Detection (True Async)', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Switch to next camera
            print("\nScanning for next available camera...")
            cap.release()
            new_index = target_camera_index
            
            # test up to 4 other indices
            for i in range(1, 5):
                test_idx = (target_camera_index + i) % 5
                temp_cap = cv2.VideoCapture(test_idx)
                if temp_cap.isOpened() and temp_cap.read()[0]:
                    new_index = test_idx
                    cap = temp_cap
                    break
                temp_cap.release()
                
            if new_index == target_camera_index:
                 print("No other working cameras found. Reopening current one.")
                 cap = cv2.VideoCapture(target_camera_index)
            else:
                 target_camera_index = new_index
                 print(f">>> Successfully switched to Camera Index: {target_camera_index} <<<")
                 
            # Re-apply resolution limits
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Clean UI and shutdown background thread
    is_running = False
    worker_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    print("Shutdown complete.")

if __name__ == "__main__":
    main()
