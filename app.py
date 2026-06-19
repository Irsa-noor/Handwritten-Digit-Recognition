import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2

# Set page configuration
st.set_page_config(
    page_title="Digit Recognition AI",
    page_icon="🔢",
    layout="centered"
)

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #FFF1B5 !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #FFF1B5 !important;
    }
    
    .main {
        background-color: #FFF1B5 !important;
    }
    
    h1 {
        color: #43302E !important;
        text-align: center !important;
    }
    
    h2 {
        color: #43302E !important;
        border-bottom: 2px solid #C1DBE8 !important;
        padding-bottom: 10px !important;
    }
    
    p {
        color: #43302E !important;
    }
    
    [data-testid="stMarkdownContainer"] {
        color: #43302E !important;
    }
    
    div[data-testid="column"] {
        background-color: #C1DBE8 !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title(" Handwritten Digit Recognition AI")
st.markdown("---")
st.write("Draw a number (0-9) below and the AI will predict what you drew!")

# Try to load the model
try:
    model = keras.models.load_model("models/digit_recognition_model.h5")
    model_loaded = True
except:
    model_loaded = False
    st.error("Model not found! Please run train_model.py first.")
    st.info("Steps:\n1. Open Command Prompt in your project folder\n2. Type: `python train_model.py`\n3. Wait for training to complete\n4. Then refresh this page")

if model_loaded:
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Draw Here")
        
        # Use Streamlit's native drawing capability
        from streamlit_drawable_canvas import st_canvas
        
        # Create drawing canvas
        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=5,
            stroke_color="black",
            background_color="white",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas"
        )
    
    with col2:
        st.subheader(" AI Prediction ")
        
        if canvas_result.image_data is not None:
            # Get the drawn image
            img = canvas_result.image_data.astype(np.uint8)
            
            # Convert RGBA to RGB if needed
            if img.shape[2] == 4:
                img = img[:, :, :3]
            
            # Convert to grayscale using OpenCV
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Resize to 28x28 using OpenCV
            img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
            
            # Invert colors (MNIST expects white on black)
            img_resized = 255 - img_resized
            
            # Normalize (0-1)
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Flatten to 784 dimensions
            img_flat = img_normalized.reshape(1, 784)
            
            # Make prediction
            prediction = model.predict(img_flat, verbose=0)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            # Display results
            st.metric("Predicted Digit", predicted_digit)
            st.metric("Confidence", f"{confidence:.1f}%")
            
            # Show all probabilities
            st.subheader("Prediction Breakdown")
            for i in range(10):
                prob = prediction[0][i] * 100
                bar = "█" * int(prob / 5) + "░" * (20 - int(prob / 5))
                st.write(f"{i}: {bar} {prob:.1f}%")

# Instructions section
st.markdown("---")
st.subheader(" How to Use:")
st.write("""
1. **Draw** a number (0-9) in the white canvas on the left
2. The **AI will instantly predict** what you drew on the right
3. **Confidence** shows how sure the AI is (higher = more certain)
4. Keep drawing to test more numbers!
""")

st.markdown("---")
st.subheader("How It Works:")
st.write("""
- **Neural Network**: Artificial "brain" with multiple layers
- **Trained on**: 70,000+ handwritten digit images
- **Accuracy**: ~98% on test data
- **Processing**: Your drawing is resized to 28×28 pixels, then analyzed by the AI
""")

