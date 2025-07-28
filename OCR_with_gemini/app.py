from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
from google.generativeai import configure, GenerativeModel
import io

# Load environment variables from a .env file.
# This is where your GOOGLE_API_KEY should be stored.
load_dotenv()

# Configure the Google Generative AI library with your API key.
# The API key is retrieved from the environment variables.
# Make sure you have GOOGLE_API_KEY set in your .env file or environment.
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found in environment variables. Please set it.")
    st.stop() # Stop the Streamlit app if API key is missing
configure(api_key=api_key)

# Initialize the Gemini model.
# The 'gemini-1.5-flash' model is used for its efficiency and multimodal capabilities.
model = GenerativeModel("gemini-1.5-flash")

def get_gemini_response(prompt, image_parts):
    """
    Sends a prompt and image data to the Gemini model and returns the text response.

    Args:
        prompt (str): The text prompt/question for the model.
        image_parts (list): A list containing image data in the format expected by the model.
                            Expected format: [{"mime_type": "image/jpeg", "data": bytes_data}]

    Returns:
        str: The generated text response from the Gemini model.
    """
    try:
        response = model.generate_content([
            prompt,
            image_parts[0]  # Assuming only one image is passed at a time
        ])
        return response.text
    except Exception as e:
        st.error(f"Error generating content from Gemini: {e}")
        return "Error: Could not get response from Gemini."

def input_image_details(uploaded_file):
    """
    Converts an uploaded Streamlit file object into the format required by the Gemini model.

    Args:
        uploaded_file (streamlit.runtime.uploaded_file_manager.UploadedFile): The file uploaded via Streamlit.

    Returns:
        list: A list containing a dictionary with mime_type and data of the image.

    Raises:
        FileNotFoundError: If no file is uploaded.
    """
    if uploaded_file is not None:
        # Read the file as bytes
        bytes_data = uploaded_file.getvalue()
        return [{
            "mime_type": uploaded_file.type, # Get the MIME type from the uploaded file object
            "data": bytes_data # The raw bytes data of the image
        }]
    else:
        # Raise an error if no file is provided, which should be handled by the UI logic.
        raise FileNotFoundError("No image uploaded")

# Streamlit UI setup
# Sets the page title and icon for the Streamlit application.
st.set_page_config(page_title="Multilanguage Invoice Extractor", page_icon=":robot:")

st.header("Multilanguage Invoice Extractor with Gemini 🤖")
st.markdown("Upload an invoice image and ask questions to extract information.")

# Input field for the user's prompt/question.
user_input = st.text_input("Input Prompt:", key="input", placeholder="e.g., What is the total amount due? Who is the invoice from?")

# File uploader widget for image uploads.
uploaded_file = st.file_uploader("Upload an image of the invoice", type=["jpg", "jpeg", "png"])

image_display = None # Initialize to None
# Display the uploaded image if one is provided.
if uploaded_file is not None:
    try:
        # Use io.BytesIO to open the image from bytes data
        image_display = Image.open(io.BytesIO(uploaded_file.getvalue()))
        # Updated: Changed use_column_width to use_container_width
        st.image(image_display, caption="Uploaded Invoice Image", use_container_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        uploaded_file = None # Reset uploaded_file if there's an error

# Button to trigger the invoice analysis.
submit = st.button("Tell me about the invoice")

# Base prompt for the Gemini model, defining its role.
base_prompt = """
You are an expert in understanding invoices.
You will be given an image of an invoice and a prompt.
Your task is to extract the requested information from the invoice and answer the prompt accurately.
The prompt will be in the form of a question.
"""

# Logic to execute when the "Tell me about the invoice" button is clicked.
if submit:
    # Check if both an image is uploaded and a prompt is entered.
    if uploaded_file is not None and user_input.strip():
        # Prepare the image data for the Gemini model.
        try:
            image_data = input_image_details(uploaded_file)
            # Combine the base prompt with the user's specific question.
            full_prompt = base_prompt + "\n\nQuestion: " + user_input
            # Get the response from the Gemini model.
            response = get_gemini_response(full_prompt, image_data)
            # Display the response from Gemini.
            st.subheader("Response from Gemini:")
            st.write(response)
        except FileNotFoundError as e:
            st.warning(str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        # Warn the user if either the image or prompt is missing.
        st.warning("Please upload an image and enter a prompt to proceed.")
