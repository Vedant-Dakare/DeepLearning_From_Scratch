from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
from PIL import Image
import streamlit as st

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_gemini_response(system_prompt, image_data, user_query):

    image_bytes, mime_type = image_data

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            system_prompt,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            ),
            user_query
        ]
    )

    return response.text


def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        mime_type = uploaded_file.type

        return bytes_data, mime_type

    raise FileNotFoundError(
        "No file uploaded. Please upload an image file."
    )


st.set_page_config(
    page_title="Gemini OCR Invoice Extractor",
    page_icon="📷",
    layout="wide"
)

st.header("📷 Gemini OCR Invoice Extractor")

user_query = st.text_input(
    "Ask something about the invoice:",
    key="input"
)


uploaded_file = st.file_uploader(
    "Upload an invoice image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


submit_button = st.button("Tell me about the invoice")


input_prompt = """
You are an expert in understanding invoices.

You will receive an invoice image as input.
Your task is to carefully analyze the invoice and answer any user questions based on it.

Extract important details such as:
- Invoice Number
- Invoice Date
- Vendor Name
- Customer Name
- Total Amount
- Tax Amount
- Items Purchased

Answer the user's question accurately based on the invoice content.
"""


if submit_button:

    if uploaded_file is None:
        st.warning("Please upload an invoice image.")
    else:
        try:
            image_data = input_image_setup(uploaded_file)

            response = get_gemini_response(
                input_prompt,
                image_data,
                user_query
            )

            st.subheader("Extracted Information")
            st.write(response)

        except Exception as e:
            st.error(f"Error: {e}")