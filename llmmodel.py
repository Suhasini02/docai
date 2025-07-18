
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
INPUT_FOLDER = "/Users/suhasini.chunduri/Documents/doc ai ocr/ocrout_texts"  # Folder containing text files
OUTPUT_FOLDER = "/Users/suhasini.chunduri/Documents/doc ai ocr/finaljson"  # Folder to save JSON files
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
    -if two textvfiles has same content address, have the file with less garbage words and dont output json for the other text file.
    -every file has only one set of address , so if it has just to address or from address no need to jave the json of other.

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

