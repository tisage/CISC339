# Lab 07: Edge AI & Deep Learning Tutorial
**Instructor:** Professor Tianyu | **Course:** CISC339

Welcome to the second half of Lab 07! After understanding the theory behind Convolutional Neural Networks (CNNs) in the Jupyter Notebook, it's time to run **State-of-the-art AI Models** directly on your local computers.

---

## 🛠 Prerequisites

Before starting, make sure you have downloaded the following files to a folder on your computer:
- `requirements.txt`
- `live_object_detection.py`
- `document_parsing.py`
- `smart_scanner.py`
*(Wait, where is `smart_scanner.py`? It might be inside the `Combined_AI_Scanner` folder if you downloaded the whole Zip!)*

Open your **Terminal** (Mac) or **Command Prompt / PowerShell** (Windows), navigate to your folder, and set up your Python environment:

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# On Mac/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# 3. Install required PyTorch dependencies
pip install -r requirements.txt
```
*(Windows users: If you do not have a GPU, installation might be faster using the CPU-only PyTorch version. See PyTorch.org for the exact command).*

---

## 🚀 Demo 1: Live Object Detection

This script uses a pre-trained **SSDLite (MobileNetV3)** model. It's incredibly fast and runs entirely offline.

```bash
python live_object_detection.py
```
**Controls for Demo 1:**
- `[c]` : Switch cameras (e.g., from your built-in webcam to a USB webcam).
- `[q]` : Quit the application.

*Observation Task: Notice the UI Video FPS vs. AI Backend FPS. Watch how multithreading ensures your video feed doesn't freeze even when the AI takes a fraction of a second to "think".*

---

## 📄 Demo 2: Document Understanding (Donut Model)

This script uses an End-to-End Multimodal Transformer (**Donut**). It takes a photograph of a receipt or notes and outputs structured JSON data without any traditional OCR (Optical Character Recognition). 

```bash
python document_parsing.py
```
*(Note: It will download a ~800MB model on the first run. Please be patient).*

*Observation Task: Try placing your own `sample_document.png` in the directory and run it again. Check the newly generated `.json` file containing the extracted data!*

---

## 🪄 Demo 3: The Smart AI Scanner

This combines both worlds! It tracks objects live using the SSD model. When you press the spacebar, it captures an HD frame and feeds it to the heavy Donut Transformer.

```bash
python smart_scanner.py
```

**Controls for Demo 3:**
- `[SPACE]` : Snap a photo and extract the document data.
- `[c]` : Switch cameras.
- `[q]` : Quit the application.

*Observation Task: When the "PROCESSING DOCUMENT" screen pops up, watch your terminal. Once finished, open the `Scanned_Documents/` folder to see the saved image and parsed data.*
