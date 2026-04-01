

import streamlit as st
import requests
import time

API_URL = "http://127.0.0.1:8000"  # Update if running FastAPI on a different host

st.title("Document Processing and Address Extraction")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.write("Uploading and processing...")

    # Send file to FastAPI
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    response = requests.post(f"{API_URL}/upload/", files=files)

    if response.status_code == 200:
        job_id = response.json()["job_id"]
        st.write(f"Processing started. Job ID: {job_id}")

        # Polling the status
        status = "uploaded"
        while status not in ["completed", "error"]:
            time.sleep(2)  # Wait before polling again
            status_response = requests.get(f"{API_URL}/status/{job_id}")
            if status_response.status_code == 200:
                status = status_response.json()["status"]
                st.write(f"Current Status: {status}")
            else:
                st.error("Error checking job status")
                break

        if status == "completed":
            # Fetch and display the final JSON result
            result_response = requests.get(f"{API_URL}/result/{job_id}")
            if result_response.status_code == 200:
                result_json = result_response.json()
                st.json(result_json)  # Display JSON in a readable format
            else:
                st.error("Error fetching result")
        else:
            st.error("Processing failed.")

    else:
        st.error("Failed to upload the image. Please try again.")
