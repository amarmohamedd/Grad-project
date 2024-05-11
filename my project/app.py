import streamlit as st
from inference_sdk import InferenceHTTPClient
import base64
from PIL import Image
import io
import numpy as np
import cv2

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="6K1nI6qRJCtsRAzSIR1N"
)

# Define a dictionary mapping class names to colors
class_colors = {
    "Brain SOL For assessment": [0, 255, 255],  # Yellow
    "Craniopharyngioma": [255, 0, 0],    # Blue
    "Extradural-hemorrhage": [0, 255, 0],    # Green
    "Intraparenchymal-hemorrhage": [255, 0, 255],  # Magenta
    "Intraventricular-hemorrhage": [255, 255, 0],  # Cyan
    "Meningioma": [255, 165, 0],  # Orange
    "Subarachnoid hemorrhage": [0, 0, 255],    # Red
    "Subdural-hemorrhage": [128, 0, 128]   # Purple
    # Add more classes and colors as needed
}

# Function to resize an image while maintaining aspect ratio
def resize_image(image, target_size):
    return image.resize(target_size, Image.LANCZOS)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(io.BytesIO(uploaded_file.read()))

    # Resize the image to a fixed size
    target_size = (800, 600)  # Set the target size here
    resized_image = resize_image(image, target_size)

    # Convert the resized image to RGB format
    resized_image = resized_image.convert("RGB")

    # Convert the resized image to base64
    buffered = io.BytesIO()
    resized_image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Perform inference on the resized image
    result = CLIENT.infer(encoded_image, model_id="medical-5/2")

    # Display inference results
    st.write("Inference Results:")
    st.json(result)
    
    # Check if segmentation parameters are available in the result
    if "predictions" in result:
        # Open the original image
        original_image = np.array(resized_image)

        # Ensure the original image is 3-channel
        if original_image.ndim == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

        # Create an empty colored mask
        colored_mask = np.zeros_like(original_image)

        for prediction in result["predictions"]:
            # Get the points
            points = [(int(point["x"]), int(point["y"])) for point in prediction["points"]]

            # Create an empty mask
            mask = np.zeros_like(original_image[:, :, 0])

            # Draw the segmentation mask
            cv2.fillPoly(mask, [np.array(points)], color=(255))

            # Get the class color
            class_color = class_colors.get(prediction["class"], [0, 0, 0])  # Default to black if class not found

            # Apply the class color to the mask
            colored_mask[mask == 255] = class_color

        # Overlay the colored mask on the original image
        segmented_image = cv2.addWeighted(original_image, 1, colored_mask, 0.5, 0)

        # Display segmentation mask
        st.write("Segmentation Mask:")
        st.image(segmented_image, caption='Segmented Image.', use_column_width=True)
    else:
        st.write("Segmentation parameters not found in inference results.")
