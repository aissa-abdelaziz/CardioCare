import streamlit as st
import time
from PIL import Image
from transformers import pipeline
import torch
from io import BytesIO
from mistralai import Mistral
import base64
from model import get_ocr_results, get_analysis, get_tips

MISTRAL_KEY = "8ZCYeJ1qgVGhlHjYP5pCNSfZbFaoGGUK"

model = "pixtral-12b-2409"
client = Mistral(api_key=MISTRAL_KEY)


# Function to simulate blood analysis
def analyze_image(image):
    status_text = st.empty()
    status_text.info("Starting analysis...")
    extracted_text = get_ocr_results(image)
    print(extracted_text)
    status_text.info("Completing...")
    return extracted_text


# Streamlit app
def main():
    st.set_page_config(page_title="CardioCare➕", layout="centered")
    st.markdown(
        "<h1 style='text-align: center;' class='header'> CardioCare➕</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h5 style='text-align: center;' class='subheader'>A blend of care and technology for personalized heart health support.</h5>",
        unsafe_allow_html=True,
    )
    st.sidebar.image(r"C:\Users\abdel\OneDrive\Desktop\hackathon\logo.png")
    st.sidebar.markdown("---")
    with st.sidebar.expander("Instructions"):
        st.write("1. 🖼️ Upload an image of your blood analysis.")
        st.write("2. 🔍 Click the 'Analyze' button.")
        st.write("3. 📈 View the results on the screen.")
        st.write("4. 📥 Download your report.")
        st.write("5. ✅ Enjoy your personalized health insights!")
    with st.sidebar.expander("🚨 Emergency Contact Information"):
        st.markdown(
            """
        In case of a medical emergency, dial **15** to reach Emergency Services immediately.  
        Your safety is our priority.
        """
        )
    st.sidebar.info(
        """
        ❗ **Note:** This AI-based tool provides insights and advice based on your blood analysis. It is not a substitute for professional medical advice.  
        If you experience any urgent or serious health issues, please contact a healthcare provider or emergency services immediately.
        """
    )

    # Main content
    with st.container():
        st.markdown("<div class='main'>", unsafe_allow_html=True)

        image_input = st.file_uploader(
            "Upload Blood Analysis Image", type=["jpg", "png", "jpeg"]
        )
        if image_input is not None:
            image = Image.open(image_input)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        analysis_complete = False
        tips_complete = False

        if st.button("Analyze", key="analyze"):
            if image_input is not None:
                progress = st.progress(0)
                with st.spinner("Extracting Information... Please wait."):
                    extracted_text = analyze_image(image)
                    print(extracted_text)
                st.success(f"{extracted_text}")
                with st.spinner("Analyzing... Please wait."):
                    analysis = get_analysis(image)
                    print(analysis)
                st.success(f"{analysis}")
                analysis_complete = True
                with st.spinner("Generating Tips... Please wait."):
                    Tips = get_tips(image)
                    print(Tips)
                st.success(f"{Tips}")
                tips_complete = True
            else:
                st.error("Please upload an image.")

        # Download button
        if analysis_complete:
            if st.button("Download Report"):
                st.success("Report downloaded! (mock functionality)")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='footer' style='text-align: center;'>Powered by CardioCare➕</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
