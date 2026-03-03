"""
Smart AI Scanner Demo
Combines Live Object Detection (SSDLite) with Document Understanding (Donut).

Controls:
  [Q] : Quit
  [C] : Switch Camera
  [SPACE] / [S] : Take a photo and process document using Transformer.

Note: Camera resolution is forced to HD (1280x720) to ensure the 
Document parsing model can read the text clearly.
"""

import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np
import time
import threading
import argparse
import os
import json
import re

# COCO Classes for Live Detection
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

# Threading Globals for Object Detection
latest_frame_for_inference = None
latest_detections = None
lock = threading.Lock()
is_running = True
inference_fps = 0.0

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  
    elif torch.cuda.is_available():
        return torch.device("cuda") 
    else:
        return torch.device("cpu")  

def detection_worker(model, device):
    """Background thread for continuous live object detection."""
    global latest_frame_for_inference, latest_detections, is_running, inference_fps
    
    while is_running:
        frame_to_process = None
        with lock:
            if latest_frame_for_inference is not None:
                frame_to_process = latest_frame_for_inference.copy()
                latest_frame_for_inference = None
                
        if frame_to_process is not None:
            inf_start = time.time()
            rgb_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
            tensor_frame = F.to_tensor(rgb_frame).to(device)
            
            with torch.no_grad():
                predictions = model([tensor_frame])[0]
                preds_cpu = {
                    'scores': predictions['scores'].cpu().numpy(),
                    'boxes': predictions['boxes'].cpu().numpy(),
                    'labels': predictions['labels'].cpu().numpy()
                }

            with lock:
                latest_detections = preds_cpu
                
            inf_time = time.time() - inf_start
            inference_fps = 1.0 / inf_time if inf_time > 0 else 0
        else:
            time.sleep(0.005)

def process_document(image_bgr, donut_model, donut_processor, device, output_dir):
    """Synchronous function to process a high-res image for document parsing."""
    print("\n" + "="*50)
    print("📸 Photo Taken! Processing document using Donut Transformer...")
    
    # 1. Save the raw image
    timestamp = int(time.time())
    image_filename = os.path.join(output_dir, f"scan_{timestamp}.jpg")
    cv2.imwrite(image_filename, image_bgr)
    print(f"[*] Saved image: {image_filename}")

    # 2. Prepare image for Donut
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    pixel_values = donut_processor(pil_image, return_tensors="pt").pixel_values.to(device)
    
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = donut_processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    # 3. Generate JSON predictions
    print("[*] Running heavy NLP/Vision inference (This may take a few seconds)...")
    with torch.no_grad():
         outputs = donut_model.generate(
             pixel_values,
             decoder_input_ids=decoder_input_ids,
             max_length=donut_model.decoder.config.max_position_embeddings,
             pad_token_id=donut_processor.tokenizer.pad_token_id,
             eos_token_id=donut_processor.tokenizer.eos_token_id,
             use_cache=True,
             bad_words_ids=[[donut_processor.tokenizer.unk_token_id]],
             return_dict_in_generate=True,
         )

    # 4. Decode
    sequence = donut_processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(donut_processor.tokenizer.eos_token, "").replace(donut_processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
    
    organized_data = donut_processor.token2json(sequence)
    
    # 5. Save JSON
    json_filename = os.path.join(output_dir, f"scan_{timestamp}.json")
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(organized_data, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Document successfully parsed!")
    print(f"[*] Saved structured data: {json_filename}")
    print("Preview:\n" + json.dumps(organized_data, indent=4, ensure_ascii=False)[:300] + "...\n")
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Smart AI Scanner Demo")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    global latest_frame_for_inference, latest_detections, is_running
    device = get_device()
    print(f"Using compute device: {device}")

    # ==========================================
    # 1. Load Object Detection Model (SSDLite)
    # ==========================================
    print("Loading Object Detection Model...")
    det_weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    det_model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=det_weights)
    det_model.to(device)
    det_model.eval()

    # ==========================================
    # 2. Load Document Understanding Model (Donut)
    # ==========================================
    print("Loading Document Understanding Model (Donut)...")
    donut_id = "naver-clova-ix/donut-base-finetuned-cord-v2"
    donut_processor = DonutProcessor.from_pretrained(donut_id)
    donut_model = VisionEncoderDecoderModel.from_pretrained(donut_id)
    donut_model.to(device)
    donut_model.eval()
    
    # Create output directory for scans
    output_dir = os.path.join(os.path.dirname(__file__), "Scanned_Documents")
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # 3. Setup Camera (HD Resolution Required)
    # ==========================================
    target_camera_index = args.camera
    cap = cv2.VideoCapture(target_camera_index)
    
    # IMPORTANT: We must request at least 720p (1280x720) or 1080p (1920x1080).
    # Mac cameras usually default to 720p. Reading small text on a receipt requires High Def.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Start detection background thread
    worker_thread = threading.Thread(target=detection_worker, args=(det_model, device))
    worker_thread.start()

    CONFIDENCE_THRESHOLD = 0.35 

    while True:
        main_start = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        with lock:
             latest_frame_for_inference = frame.copy()
             current_detections = latest_detections

        display_frame = frame.copy()

        # Draw Object Detection Bounding Boxes
        if current_detections is not None:
            scores = current_detections['scores']
            boxes = current_detections['boxes']
            labels = current_detections['labels']
            
            for i in range(len(scores)):
                if scores[i] > CONFIDENCE_THRESHOLD:
                    box = boxes[i].astype(int)
                    x1, y1, x2, y2 = box
                    label_idx = labels[i]
                    label_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else "Unknown"
                    label_text = f"{label_name}: {scores[i]:.2f}"

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    (tw, th), bl = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(display_frame, (x1, y1 - th - bl), (x1 + tw, y1), (0, 255, 0), -1)
                    cv2.putText(display_frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw UI
        cv2.putText(display_frame, "Smart Scanner: AI Object Tracking Active", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(display_frame, "[SPACE] Take Photo & Parse Document | [C] Switch Cam | [Q] Quit", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        window_title = 'AI Smart Scanner (Object Detection + Document NLP)'
        cv2.imshow(window_title, display_frame)

        # Handle Keyboard Inputs
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("\nSwitching camera...")
            cap.release()
            target_camera_index = (target_camera_index + 1) % 4
            cap = cv2.VideoCapture(target_camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        elif key == ord(' ') or key == ord('s'):
            # taking photo!
            print("\n>>> FREEZING FRAME FOR NLP DOCUMENT ANALYSIS <<<")
            
            # Show a "PROCESSING" overlay to the user
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            cv2.putText(display_frame, "PROCESSING DOCUMENT...", (display_frame.shape[1]//2 - 250, display_frame.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow(window_title, display_frame)
            cv2.waitKey(1) # Force UI refresh
            
            # Call synchronous document processing
            # We pass the raw un-drawn frame so bounding boxes don't mess up text recognition
            process_document(frame.copy(), donut_model, donut_processor, device, output_dir)
            
            print(">>> RESUMING LIVE FEED <<<\n")

    is_running = False
    worker_thread.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
