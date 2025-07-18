## **Project Workflow Documentation**

### **1. Data Preparation using Label Studio (Labeling & Organization)**
- **Dataset:** Images labeled in Label Studio into `to_address` and `from_address`.
- **Format:** Exported in YOLO format.
- **Structure:**
  - `images/`: Contains raw images.
  - `labels/`: Contains YOLO annotation files.
  - `test set/`: Contains test images for validation.
  - `data.yaml`: Configuration file specifying dataset paths and classes.
  - `classes.txt`: Contains class id names.

---

### **2. Model Training (YOLO)**
- **Training with YOLOv10:**
  - `trainmain.py`: Script for training the YOLO model.
  - Model is trained on labeled images to detect `to_address` and `from_address`.
- **Training Output:**
  - `runs/train/`: Stores training logs, model weights, and evaluation metrics.
  - `confusion_matrix.png`: Shows model performance.
  - `results.csv`: Stores training results.
  - `yolov10n.pt`: Trained model weights.

---

### **3. FastAPI Backend for Image Processing and JSON Extraction**
- **FastAPI Service** handles:
  1. **Uploading images**: Images are sent to the `/upload/` endpoint.
  2. **Object detection with YOLO**: The trained model detects `to_address` and `from_address` from images.
  3. **Sending detected text to Mistral** for structured JSON extraction.
  4. **Returning JSON output** with extracted `to_address` and `from_address`.

- **Key Endpoints in FastAPI:**
  - `POST /upload/` → Accepts an image file and starts processing.
  - `GET /status/{image_id}` → Checks processing status.
  - `GET /result/{image_id}` → Returns extracted JSON.

---

### **4. How Image Uploads Convert to JSON**
1. **User uploads an image** via the Streamlit UI.
2. **FastAPI stores the image** in the `uploads/` directory.
3. **YOLO detects bounding boxes** for `to_address` and `from_address`.
4. **Google Doc AI OCR extracts text** from the detected regions.
5. **Mistral processes the extracted text** to format it into a structured JSON.
6. **FastAPI returns JSON output** containing `to_address` and `from_address`.

---

### **5. Streamlit UI for Interaction**
- **Features:**
  - Upload test images.
  - View extracted `to_address` and `from_address` in JSON format.
  - Display detected bounding boxes on images.

- **Flow:**
  1. User uploads an image.
  2. Streamlit sends it to the FastAPI `/upload/` endpoint.
  3. Streamlit checks the `/status/` endpoint.
  4. When processing is complete, Streamlit retrieves JSON from `/result/`.
  5. The JSON is displayed along with the processed image.

---
