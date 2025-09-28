import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from keras.models import load_model
import io

# Page configuration
st.set_page_config(
    page_title="Document Forgery Detection",
    page_icon="📄",
    layout="wide"
)

# Header with GitHub link
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📄 Document Forgery Detection")
    st.write("Upload a document image to detect if it has been tampered with using Error Level Analysis and CNN.")

with col2:
    st.markdown("### 🔗 Links")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Complete_Project-blue?logo=github)](https://github.com/rahulpmishra/document-forgery-detection)")
    st.markdown("[![Kaggle](https://img.shields.io/badge/Kaggle-Training_Notebook-orange?logo=kaggle)](https://www.kaggle.com/code/rahulprakashmishra/image-forgery-detection-cnn-training)")

st.markdown("---")

# Sidebar for settings
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider(
    "Classification Threshold", 
    min_value=0.1, 
    max_value=0.9, 
    value=0.5, 
    step=0.05,
    help="Threshold for classifying as Forged (≤) vs Authentic (>)"
)

st.sidebar.markdown("**Threshold Guide:**")
st.sidebar.markdown("• **Lower (0.3-0.4):** More sensitive to forgery")
st.sidebar.markdown("• **Default (0.5):** Balanced detection") 
st.sidebar.markdown("• **Higher (0.6-0.7):** More conservative")

@st.cache_resource
def load_detection_model():
    """Load the trained model (cached for performance)"""
    return load_model("./models/trained_model.h5")

def convert_to_ela_image_memory(image, quality=90):
    """Convert image to ELA without saving to disk"""
    # Convert to RGB if not already
    original_image = image.convert("RGB")
    
    # Save to memory buffer with specified quality
    buffer = io.BytesIO()
    original_image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    
    # Reload from buffer
    resaved_image = Image.open(buffer)
    
    # Calculate pixel difference
    ela_image = ImageChops.difference(original_image, resaved_image)
    
    # Calculate scaling factors
    extrema = ela_image.getextrema()
    max_difference = max([pix[1] for pix in extrema])
    if max_difference == 0:
        max_difference = 1
    scale = 350.0 / max_difference
    
    # Enhance brightness
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image

def prepare_image_for_prediction(image):
    """Prepare image for CNN prediction"""
    image_size = (128, 128)
    ela_image = convert_to_ela_image_memory(image, 90)
    return np.array(ela_image.resize(image_size)).flatten() / 255.0

def predict_forgery(image, threshold=0.5):
    """Predict if image is forged or authentic"""
    model = load_detection_model()
    class_names = ["Forged", "Authentic"]
    
    test_image = prepare_image_for_prediction(image)
    test_image = test_image.reshape(-1, 128, 128, 3)
    
    y_pred = model.predict(test_image, verbose=0)
    raw_score = float(y_pred[0][0])
    
    # Use dynamic threshold
    prediction = class_names[0] if raw_score <= threshold else class_names[1]
    
    if raw_score <= threshold:
        confidence = f"{(1-raw_score) * 100:.2f}"
    else:
        confidence = f"{raw_score * 100:.2f}"
    
    return prediction, confidence, raw_score

# Main interface
uploaded_file = st.file_uploader(
    "Choose a document image file", 
    type=['png', 'jpg', 'jpeg'],
    help="Upload a PNG, JPG, or JPEG document image to analyze"
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    
    # Run prediction first
    with st.spinner("Analyzing document for forgery detection..."):
        try:
            prediction, confidence, raw_score = predict_forgery(image, threshold)
            
            # Display results first
            st.subheader("🎯 Detection Results")
            
            # Create result container with color coding
            col1, col2, col3 = st.columns([2, 2, 2])
            
            with col1:
                if prediction == "Authentic":
                    st.success(f"**Prediction:** {prediction}")
                else:
                    st.error(f"**Prediction:** {prediction}")
                    
            with col2:
                if prediction == "Authentic":
                    st.success(f"**Confidence:** {confidence}%")
                else:
                    st.error(f"**Confidence:** {confidence}%")
                    
            with col3:
                st.info(f"**Raw Score:** {raw_score:.4f}")
                st.caption(f"Threshold: {threshold}")
                
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.info("Please make sure the trained_model.h5 file is present in the project directory.")
    
    # Then show images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Document")
        st.image(image, caption="Uploaded Document", use_container_width=True)
    
    # Generate ELA image
    ela_image = convert_to_ela_image_memory(image)
    
    with col2:
        st.subheader("ELA Analysis")
        st.image(ela_image, caption="Error Level Analysis", use_container_width=True)

else:
    st.info("👆 Please upload a document image file to begin analysis")
    
    # Show example info when no image is uploaded
    st.markdown("---")
    st.subheader("About this tool")
    st.write("""
    This application uses **Error Level Analysis (ELA)** combined with a **Convolutional Neural Network** 
    to detect document forgery and tampering.
    
    - **Upload** any PNG, JPG, or JPEG document image
    - **Adjust** the classification threshold in the sidebar
    - **View** the original document and ELA analysis side by side
    - **Get** instant prediction results with confidence percentage
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p style='color: #888; font-size: 14px;'>
            👨‍💻 <strong>Connect with me:</strong> <br> •
            <a href="https://rpmdev.me" target="_blank">Portfolio</a> • 
            <a href="https://github.com/rahulpmishra" target="_blank">GitHub</a> • 
            <a href="https://www.kaggle.com/rahulprakashmishra" target="_blank">Kaggle</a> 
        </p>
        <p style='color: #aaa; font-size: 12px; margin-top: 15px;'>
            💡 Explore more of my work and reach out for collaborations!
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)