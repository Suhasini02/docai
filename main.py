from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import cv2
import numpy as np
import json
import uuid
from typing import List
import time
from pydantic import BaseModel
import shutil
from ultralytics import YOLO
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI

# Load environment variables
load_dotenv()

app = FastAPI(title="Document Processing API", 
              description="API for document text extraction and address parsing")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ideally, you'd restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
MERGED_DIR = "merged"
JSON_OUTPUT_DIR = "json_output"

# Ensure directories exist
for directory in [UPLOAD_DIR, OUTPUT_DIR, MERGED_DIR, JSON_OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Load models and clients
try:
    # Load YOLO Model - Adjust path to your model
    model = YOLO("/Users/suhasini.chunduri/Documents/doc ai ocrmain/runs/detect/train/weights/best.pt")
    
    # Load Google Cloud credentials
    SERVICE_ACCOUNT_JSON = os.getenv("SERVICE_ACCOUNT_PATH", "/Users/suhasini.chunduri/Documents/doc ai ocrmain/service account.json")
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON)
    
    # Initialize Document AI client
    client = documentai.DocumentProcessorServiceClient(credentials=credentials)
    
    # Document AI Configuration
    PROJECT_ID = os.getenv("PROJECT_ID", "882762201937")
    LOCATION = os.getenv("LOCATION", "us")
    PROCESSOR_ID = os.getenv("PROCESSOR_ID", "15bf088b0ccf5abc")
    OCR_ENDPOINT = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"
    
    # Initialize Mistral LLM
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    llm = ChatMistralAI(model="mistral-medium", mistral_api_key=MISTRAL_API_KEY)
except Exception as e:
    print(f"Error initializing models and clients: {e}")
    # We'll handle this better in the endpoints

# Status tracking for background tasks
job_status = {}

# Helper functions from the original code
def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def remove_duplicate_boxes(yolo_boxes, iou_threshold=0.5):
    """Remove duplicate bounding boxes based on IoU threshold."""
    unique_boxes = []
    
    for i, box1 in enumerate(yolo_boxes):
        duplicate_found = False
        for box2 in unique_boxes:
            if calculate_iou(box1, box2) > iou_threshold:
                duplicate_found = True
                break
        if not duplicate_found:
            unique_boxes.append(box1)

    return unique_boxes

def merge_yolo_regions(image_path, yolo_boxes, merged_dir, spacing=10):
    """Merge YOLO-detected regions into a single image with spacing."""
    image = cv2.imread(image_path)
    
    if not yolo_boxes:
        return None, []

    # Sort boxes by y-coordinate (top to bottom)
    yolo_boxes.sort(key=lambda box: box[1])

    # Compute merged image dimensions
    max_width = max(x2 - x1 for x1, y1, x2, y2 in yolo_boxes)
    total_height = sum(y2 - y1 for x1, y1, x2, y2 in yolo_boxes) + spacing * (len(yolo_boxes) - 1)

    # Create a blank white image
    merged_image = np.ones((total_height, max_width, 3), dtype=np.uint8) * 255  

    y_offset = 0
    region_heights = []  # Store height of each cropped region

    for (x1, y1, x2, y2) in yolo_boxes:
        cropped_region = image[y1:y2, x1:x2]  
        height, width = cropped_region.shape[:2]
        merged_image[y_offset:y_offset + height, :width] = cropped_region  
        region_heights.append(height)
        y_offset += height + spacing  # Move y-offset for next region

    # Save merged image
    merged_image_path = os.path.join(merged_dir, os.path.basename(image_path).replace(".jpg", "_merged.jpg"))
    cv2.imwrite(merged_image_path, merged_image)

    return merged_image_path, region_heights

def extract_text_from_image(image_path):
    """Extract text from a single image using Document AI."""
    with open(image_path, "rb") as image_file:
        image_content = image_file.read()

    request = documentai.ProcessRequest(
        name=OCR_ENDPOINT,
        raw_document=documentai.RawDocument(content=image_content, mime_type="image/jpeg"),
    )

    result = client.process_document(request=request)
    return result.document.text.strip()

# Address prompt template
address_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""
    Extract and classify the following address information from the given text.
    - If the text explicitly mentions 'from', keep only 'from'.
    - If the text explicitly mentions 'to', keep only 'to'.
    - If neither 'from' nor 'to' is mentioned, assume it is 'to'.
    - Remove any unnecessary content that is not relevant to an address.
    - If there are additional relevant entities like phone numbers or emails, include them.
    Respond in valid JSON format.
    -If "To" or "From" is explicitly mentioned, extract those addresses accordingly.
    If a gap exists between detected addresses and only one label ("To" or "From") is found, assume the other address is the opposite one and generate.

    -if two textfiles has same content address, have the file with less garbage words and dont output json for the other text file.
    -every file has only one set of address, so if it has just to address or from address no need to have the json of other.
    -remove meaningless words and letters, also check the entity and label properly.
    -name can be company name or person name, detect properly and remove garbage words.
Text Input:
    {text}

    Output format:
    {{
        "imagename": "<filename>",
        "from": {{ "name": "", "address": "", "city": "", "state": "", "country": "", "pincode": "", "contactno": "" }}  # Include only if 'from' is in text
        "to": {{ "name": "", "address": "", "city": "", "state": "", "country": "", "pincode": "", "contactno": "" }}  # Include only if 'to' is in text or inferred
    }}
    """
)

async def process_document(image_path, job_id):
    """Process a single document image: detect text regions, OCR, and extract addresses."""
    try:
        # Update status
        job_status[job_id]["status"] = "detecting_text"
        
        # Run YOLO for text detection
        results = model(image_path)
        yolo_boxes = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for x1, y1, x2, y2 in boxes:
                yolo_boxes.append((int(x1), int(y1), int(x2), int(y2)))

        # Update status
        job_status[job_id]["status"] = "processing_regions"
        job_status[job_id]["detected_regions"] = len(yolo_boxes)

        # Remove duplicate boxes
        unique_yolo_boxes = remove_duplicate_boxes(yolo_boxes)
        job_status[job_id]["unique_regions"] = len(unique_yolo_boxes)

        if not unique_yolo_boxes:
            job_status[job_id]["status"] = "error"
            job_status[job_id]["error"] = "No valid text regions detected"
            return None

        # Merge cropped regions into one image with spacing
        merged_image_path, region_heights = merge_yolo_regions(
            image_path, unique_yolo_boxes, MERGED_DIR, spacing=10
        )
        
        # Update status
        job_status[job_id]["status"] = "performing_ocr"
        
        # Perform OCR on the merged image
        extracted_text = extract_text_from_image(merged_image_path)
        
        # Split text based on newline and approximate region heights
        extracted_lines = extracted_text.split("\n")
        
        # Calculate the number of lines per region using height ratios
        total_text_lines = len(extracted_lines)
        total_image_height = sum(region_heights)
        region_line_counts = [max(1, int((height / total_image_height) * total_text_lines)) 
                             for height in region_heights]

        # Ensure region_line_counts sum matches the total lines extracted
        while sum(region_line_counts) < total_text_lines:
            region_line_counts[-1] += 1  # Adjust the last region count if necessary

        # Insert blank lines based on actual spacing in the merged image
        formatted_text = []
        line_index = 0

        for i, line_count in enumerate(region_line_counts):
            formatted_text.extend(extracted_lines[line_index:line_index + line_count])
            
            # Add spacing lines between text regions
            if i < len(region_heights) - 1:  
                spacing_lines = max(2, int(10 / 10))  # Convert pixels to lines
                formatted_text.extend([""] * spacing_lines)

            line_index += line_count

        # Save formatted text
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_text_file = os.path.join(OUTPUT_DIR, f"{base_filename}.txt")

        with open(output_text_file, "w", encoding="utf-8") as txt_file:
            txt_file.write("\n".join(formatted_text))

        # Update status
        job_status[job_id]["status"] = "extracting_addresses"
        
        # Extract addresses using LLM
        text_content = "\n".join(formatted_text)
        chain = address_prompt_template | llm
        response = chain.invoke({"text": text_content})
        
        # Extract the response text
        response_text = response.content if hasattr(response, "content") else str(response)

        # Parse and save JSON
        try:
            address_data = json.loads(response_text)
            address_data["imagename"] = os.path.basename(image_path)
            
            # Save JSON output
            json_output_path = os.path.join(JSON_OUTPUT_DIR, f"{base_filename}.json")
            with open(json_output_path, "w", encoding="utf-8") as json_file:
                json.dump(address_data, json_file, indent=4)
                
            # Update status - completed successfully
            job_status[job_id]["status"] = "completed"
            job_status[job_id]["result"] = json_output_path
            return json_output_path
            
        except json.JSONDecodeError:
            job_status[job_id]["status"] = "error"
            job_status[job_id]["error"] = "Invalid JSON output from LLM"
            return None
            
    except Exception as e:
        job_status[job_id]["status"] = "error"
        job_status[job_id]["error"] = str(e)
        return None

# API Endpoints
@app.get("/")
def read_root():
    return {"message": "Document Processing API is running"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Upload a document image for processing."""
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Initialize job status
    job_status[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "status": "uploaded",
        "start_time": time.time(),
        "filepath": file_path
    }
    
    # Start processing in background
    background_tasks.add_task(process_document, file_path, job_id)
    
    return {"job_id": job_id, "message": "File uploaded successfully. Processing started."}

@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    """Get the status of a document processing job."""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status[job_id]

@app.get("/result/{job_id}")
def get_job_result(job_id: str):
    """Get the results of a completed job."""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_status[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not completed. Current status: {job['status']}")
    
    # Read and return the JSON result
    with open(job["result"], "r") as f:
        result = json.load(f)
    
    return result

@app.delete("/job/{job_id}")
def delete_job(job_id: str):
    """Delete a job and its associated files."""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_status[job_id]
    
    # Delete associated files
    files_to_delete = [
        job.get("filepath"),
        job.get("result"),
        # Add other files if needed
    ]
    
    for file_path in files_to_delete:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    
    # Remove from status tracking
    del job_status[job_id]
    
    return {"message": "Job and associated files deleted successfully"}

# Processing multiple files at once
class BatchProcessRequest(BaseModel):
    job_ids: List[str]

@app.post("/batch-process/")
def batch_process(request: BatchProcessRequest):
    """Get status of multiple jobs."""
    results = {}
    for job_id in request.job_ids:
        if job_id in job_status:
            results[job_id] = job_status[job_id]
        else:
            results[job_id] = {"error": "Job not found"}
    
    return results

# Run with: uvicorn main:app --reload