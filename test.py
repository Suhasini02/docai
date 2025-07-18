

import os
import cv2
import numpy as np
from ultralytics import YOLO
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account


# Load YOLO Model
model = YOLO("/Users/suhasini.chunduri/Documents/doc ai ocr/runs/detect/train/weights/best.pt")

# Folder Paths
IMAGE_FOLDER = "/Users/suhasini.chunduri/Documents/doc ai ocrmain/label studiomain/test set"
OUTPUT_FOLDER = "/Users/suhasini.chunduri/Documents/doc ai ocrmain/ocrout_texts"
MERGED_FOLDER = "/Users/suhasini.chunduri/Documents/doc ai ocrmain/merged_images"

# Ensure Output Folders Exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MERGED_FOLDER, exist_ok=True)

# Load Google Cloud credentials
SERVICE_ACCOUNT_JSON = "/Users/suhasini.chunduri/Documents/doc ai ocr/service account.json"
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON)

# Initialize Document AI client
client = documentai.DocumentProcessorServiceClient(credentials=credentials)
# Set your service account JSON key file


#  Document AI Configuration
PROJECT_ID = "882762201937"
LOCATION = "us"  
PROCESSOR_ID = "15bf088b0ccf5abc"
OCR_ENDPOINT = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"

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

def merge_yolo_regions(image_path, yolo_boxes, spacing=10):
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
    merged_image_path = os.path.join(MERGED_FOLDER, os.path.basename(image_path).replace(".jpg", "_merged.jpg"))
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

def process_image(image_path):
    """Detect, merge, extract text, and ensure proper spacing."""
    print(f"🔍 Processing {os.path.basename(image_path)}...")

    # Run YOLO for text detection
    results = model(image_path)
    yolo_boxes = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for x1, y1, x2, y2 in boxes:
            yolo_boxes.append((int(x1), int(y1), int(x2), int(y2)))

    print(f"📌 YOLO detected {len(yolo_boxes)} text regions.")

    # Remove duplicate boxes
    unique_yolo_boxes = remove_duplicate_boxes(yolo_boxes)
    print(f"✅ Removed duplicates, {len(unique_yolo_boxes)} unique text regions remain.")

    if not unique_yolo_boxes:
        print(f"⚠️ No valid text regions detected in {os.path.basename(image_path)}, skipping OCR.\n")
        return None

    # Merge cropped regions into one image with spacing
    merged_image_path, region_heights = merge_yolo_regions(image_path, unique_yolo_boxes, spacing=10)
    print(f"📌 Merged image saved: {merged_image_path}")

    # Perform OCR on the merged image
    extracted_text = extract_text_from_image(merged_image_path)
    
    # Split text based on newline and approximate region heights
    extracted_lines = extracted_text.split("\n")
    
    # Calculate the number of lines per region using height ratios
    total_text_lines = len(extracted_lines)
    total_image_height = sum(region_heights)
    region_line_counts = [max(1, int((height / total_image_height) * total_text_lines)) for height in region_heights]

    # Ensure region_line_counts sum matches the total lines extracted
    while sum(region_line_counts) < total_text_lines:
        region_line_counts[-1] += 1  # Adjust the last region count if necessary

    # Insert blank lines based on actual spacing in the merged image
    formatted_text = []
    line_index = 0

    for i, line_count in enumerate(region_line_counts):
        formatted_text.extend(extracted_lines[line_index:line_index + line_count])
        
        # Add spacing lines between text regions (based on the spacing in the image)
        if i < len(region_heights) - 1:  # Don't add spacing after the last region
            spacing_lines = max(2, int(10 / 10))  # Convert pixels to lines
            formatted_text.extend([""] * spacing_lines)

        line_index += line_count

    # Save formatted text
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_text_file = os.path.join(OUTPUT_FOLDER, f"{base_filename}.txt")

    with open(output_text_file, "w", encoding="utf-8") as txt_file:
        txt_file.write("\n".join(formatted_text))

    print(f"✅ OCR complete! Extracted text saved to: {output_text_file}\n")

# Process all images
for image_file in os.listdir(IMAGE_FOLDER):
    if image_file.lower().endswith((".jpg", ".png", ".jpeg")):
        process_image(os.path.join(IMAGE_FOLDER, image_file))

print("✅ All images processed. OCR text files are saved.")
import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI

# Load environment variables
load_dotenv()

# Ensure API key is available
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY environment variable not set.")

# Initialize Mistral LLM
llm = ChatMistralAI(model="mistral-medium", mistral_api_key=MISTRAL_API_KEY)

# Define input and output directories
INPUT_FOLDER = "/Users/suhasini.chunduri/Documents/doc ai ocrmain/ocrout_texts"  # Folder containing text files
OUTPUT_FOLDER = "/Users/suhasini.chunduri/Documents/doc ai ocrmain/finaljson"  # Folder to save JSON files
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Prompt template to extract and classify addresses
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
If a gap exists between detected addresses and only one label ("To" or "From") is found, assume the other address is the opposite one and generate. example:


    -if two textvfiles has same content address, have the file with less garbage words and dont output json for the other text file.
    -every file has only one set of address , so if it has just to address or from address no need to jave the json of other.
    -remove meaningless words and letters, also check the entity and label properly.
    -name can company name or person name, detect properly and remove garbage words.

    Text:
    {text}

    Output format:
    {{
        "imagename": "<filename>",
        "from": {{ "name": "", "address": "", "city": "", "state": "", "country": "", "pincode": "", "contactno": "" }}  # Include only if 'from' is in text
        "to": {{ "name": "", "address": "", "city": "", "state": "", "country": "", "pincode": "", "contactno": "" }}  # Include only if 'to' is in text or inferred
    }}
    """
)

# Process each text file
for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".txt"):
        input_path = os.path.join(INPUT_FOLDER, filename)
        
        with open(input_path, "r", encoding="utf-8") as file:
            text_content = file.read()
        
        # Generate JSON output
        chain = address_prompt_template | llm
        response = chain.invoke({"text": text_content})

# Extract the response text
        response_text = response.content if hasattr(response, "content") else str(response)

# Parse and save JSON
        try:
            address_data = json.loads(response_text)

            address_data["imagename"] = filename  # Add filename to JSON
            output_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.json")
            with open(output_path, "w", encoding="utf-8") as json_file:
                json.dump(address_data, json_file, indent=4)
            print(f"Processed: {filename} -> {output_path}")
        except json.JSONDecodeError:
            print(f"Error processing {filename}: Invalid JSON output from Mistral")