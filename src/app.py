import streamlit as st
import boto3
import json
import numpy as np
import cv2
import os
from streamlit_drawable_canvas import st_canvas
from dotenv import load_dotenv

# Load config
load_dotenv()
ENDPOINT_NAME = 'mnist-digit-recognizer'
REGION = os.getenv('AWS_REGION')

st.title("Draw a Digit (0-9)")
st.write("Draw a number in the box below and ask the AI to identify it.")


canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=15,                      # Thicker stroke is better for resizing
    stroke_color="#FFFFFF",               # White pen
    background_color="#000000",           # Black background
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

def process_image(image_data):
    gray_image = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)
    
    resized = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)
    
    normalized = resized / 255.0
    
    flattened = normalized.flatten().tolist()
    
    return flattened, resized

if st.button('PREDICT DRAWING'):
    if canvas_result.image_data is not None:
        
        pixels, debug_img = process_image(canvas_result.image_data)
        
        st.write("What the AI sees (28x28 pixels):")
        st.image(debug_img, width=100, clamp=True)
        
        # Send to AWS
        with st.spinner('Sending to AWS SageMaker...'):
            try:
                runtime = boto3.client('sagemaker-runtime', region_name=REGION)
                
                payload = json.dumps(pixels)
                
                response = runtime.invoke_endpoint(
                    EndpointName=ENDPOINT_NAME,
                    ContentType='application/json',
                    Body=payload
                )
                result = json.loads(response['Body'].read().decode())
                
                # Display Result
                st.markdown(f"## Prediction: **{result}**")
                
            except Exception as e:
                st.error(f"Error connecting to AWS: {e}")
    else:
        st.warning("Please draw something first!")

st.info(f"Connected to Endpoint: {ENDPOINT_NAME}")