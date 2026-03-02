"""
Live Object Detection Demo using PyTorch and OpenCV.

This script captures video from a webcam, performs real-time object detection
using a pre-trained SSDLite model (MobileNetV3 backbone), and displays the results.
The model is optimized and lightweight, suitable for running on Mac CPUs/GPUs.
"""

import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import time

# List of COCO dataset class names (91 classes)
# These are the labels the pre-trained model was trained on.
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

def get_device():
    """Select the best available device for computation."""
    if torch.backends.mps.is_available():
        # Metal Performance Shaders for Apple Silicon (M1/M2/M3)
        print("Hardware Match: Apple Silicon (MPS) detected. GPU acceleration enabled!")
        return torch.device("mps")  
    elif torch.cuda.is_available():
        # NVIDIA GPU
        print("Hardware Match: NVIDIA GPU (CUDA) detected. GPU acceleration enabled!")
        return torch.device("cuda") 
    else:
        # Fallback to CPU
        print("Hardware Match: No compatible GPU found. Defaulting to CPU.")
        return torch.device("cpu")  

def main():
    # 1. Setup device
    device = get_device()
    print(f"Using compute device: {device}")

    # 2. Load pre-trained model
    print("Loading pre-trained SSDLite MobileNetV3 model...")
    # SSDLite is optimized for mobile/edge CPU/GPU, offering a good balance of speed and accuracy
    weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
    
    # Move model to the selected device and set it to evaluation mode
    model.to(device)
    model.eval()  

    # 3. Initialize video capture
    # Automatically find the first available camera index (checks indices 0 to 4)
    print("Searching for available cameras...")
    cap = None
    camera_index = -1
    
    for i in range(5):
        print(f"  -> Checking camera index {i}...")
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            # Test if you can actually read a valid frame
            ret, _ = temp_cap.read()
            if ret:
                cap = temp_cap
                camera_index = i
                print(f"  [!] Success: Found working camera at index {camera_index}!")
                break
            else:
                temp_cap.release()
        else:
            temp_cap.release()
            
    if cap is None or not cap.isOpened():
        print("Error: Could not find or open any webcams.")
        print("Please check if the camera is connected and you have granted Terminal/Python camera permissions in Mac Privacy settings.")
        return

    # Set camera resolution (optional, lower resolution = faster inference)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting video source. Press 'q' to quit.")

    # 4. Process video stream
    frame_count = 0
    process_every_n_frames = 3  # [优化] 跳帧机制：每 3 帧进行一次模型推理，避免视频阻塞卡顿
    last_predictions = None

    with torch.no_grad(): # Disable gradient calculation for inference (saves memory & speeds up)
        while True:
            start_time = time.time()
            
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
                
            frame_count += 1

            # Only run heavy PyTorch inference every N frames
            if frame_count % process_every_n_frames == 0 or last_predictions is None:
                # Convert OpenCV frame (BGR) to RGB format expected by PyTorch
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert numpy array to PyTorch tensor
                # The model expects a list of tensors, each of shape [C, H, W], normalized to [0, 1]
                tensor_frame = F.to_tensor(rgb_frame).to(device)
    
                # Perform object detection
                # The output is a list of dictionaries (one for each input image)
                predictions = model([tensor_frame])[0]
                
                # Move to CPU and compute numpy arrays only when a new inference is made
                last_predictions = {
                    'scores': predictions['scores'].cpu().numpy(),
                    'boxes': predictions['boxes'].cpu().numpy(),
                    'labels': predictions['labels'].cpu().numpy()
                }

            # Use the latest available predictions for drawing
            scores = last_predictions['scores']
            boxes = last_predictions['boxes']
            labels = last_predictions['labels']

            # Filter predictions by confidence score threshold
            threshold = 0.5

            for i in range(len(scores)):
                if scores[i] > threshold:
                    # Extract bounding box coordinates
                    box = boxes[i].astype(int)
                    x1, y1, x2, y2 = box

                    # Get predicted class label
                    label_idx = labels[i]
                    label_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else "Unknown"
                    
                    # Create label text with confidence score
                    label_text = f"{label_name}: {scores[i]:.2f}"

                    # Draw bounding box on the original frame
                    color = (0, 255, 0) # Green color in BGR format
                    thickness = 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw text background to make text readable
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - text_h - baseline), (x1 + text_w, y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label_text, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Calculate and display FPS (Frames Per Second)
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # Display the resulting frame
            cv2.imshow('Live Object Detection (Press Q to quit)', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 5. Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
