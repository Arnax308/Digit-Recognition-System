import streamlit as st

# --- Page Configuration - MUST BE FIRST ---
st.set_page_config(
    page_title="Magic Number Reader 🌟", 
    page_icon="🌟", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Now import everything else
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import cv2
import time
import matplotlib.pyplot as plt
import os
from scipy import ndimage
import tensorflow as tf

# Initialize session state for canvas clearing
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0

# Load pre-trained CNN digit recognition model
@st.cache_resource
def load_digit_model():
    model_path = "digit_model_cnn.h5"
    
    if os.path.exists(model_path):
        try:
            # Load existing pre-trained CNN model
            model = tf.keras.models.load_model(model_path)
            print("✅ CNN Model loaded successfully")
            return model
        except Exception as e:
            st.error(f"❌ Error loading CNN model: {e}")
            st.stop()
    else:
        st.error("""
        ⚠️ **CNN Model file not found!**
        
        Please run the CNN model trainer first to create the digit_model_cnn.h5 file:
        
        1. Install TensorFlow: `pip install tensorflow`
        2. Run: `python model-cnn.py`
        3. This will create the `digit_model_cnn.h5` file
        4. Then run this app again!
        """)
        st.stop()

model = load_digit_model()

# --- Custom CSS Styling (same as before) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka+One:wght@400&family=Nunito:wght@400;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .title {
        font-family: 'Fredoka One', cursive;
        font-size: 3rem;
        color: #4A90E2;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-10px);
        }
        60% {
            transform: translateY(-5px);
        }
    }
    
    .subtitle {
        font-family: 'Nunito', sans-serif;
        font-size: 1.3rem;
        color: #450;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 600;
    }
    
    .canvas-container {
        background: linear-gradient(45deg, #FFE0B2, #FFF3E0);
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.1);
        border: 3px dashed #FF9800;
        text-align: center;
        display: none;
    }
    
    .canvas-title {
        font-family: 'Nunito', sans-serif;
        font-size: 1.5rem;
        color: #FF9800;
        text-align: center;
        margin-bottom: 15px;
        font-weight: 700;
    }
    
    .stButton>button {
        font-family: 'Nunito', sans-serif;
        color: white;
        background: linear-gradient(45deg, #FF6B6B, #FF8E53);
        border: none;
        border-radius: 25px;
        padding: 15px 30px;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        margin: 10px auto;
        width: 100%;
        height: 60px;
        display: block;
    }
    
    .stButton>button:hover {
        background: linear-gradient(45deg, #FF8E53, #FF6B6B);
        transform: translateY(-3px);
        box-shadow: 0 12px 20px rgba(0,0,0,0.3);
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #A8E6CF, #7FCDCD);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        font-family: 'Fredoka One', cursive;
        font-size: 2rem;
        color: #2C3E50;
        margin: 20px 0;
        box-shadow: 0 15px 25px rgba(0,0,0,0.2);
        border: 3px solid #4ECDC4;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    
    .prediction-digit {
        font-size: 4rem;
        color: #E74C3C;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.2);
        display: inline-block;
        animation: wiggle 1s ease-in-out;
    }
    
    @keyframes wiggle {
        0%, 7% {
            transform: rotateZ(0);
        }
        15% {
            transform: rotateZ(-15deg);
        }
        20% {
            transform: rotateZ(10deg);
        }
        25% {
            transform: rotateZ(-10deg);
        }
        30% {
            transform: rotateZ(6deg);
        }
        35% {
            transform: rotateZ(-4deg);
        }
        40%, 100% {
            transform: rotateZ(0);
        }
    }
    
    .fun-facts {
        background: linear-gradient(135deg, #FFD93D, #FF9F1C);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        font-family: 'Nunito', sans-serif;
        font-weight: 600;
        color: #2C3E50;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    
    .instructions {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        font-family: 'Nunito', sans-serif;
        color: #34495E;
        text-align: center;
        border-left: 5px solid #3498DB;
    }
    
    .debug-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        border: 2px dashed #3498DB;
    }
    
    .model-info {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 15px;
        padding: 15px;
        margin: 20px 0;
        text-align: center;
        font-family: 'Nunito', sans-serif;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# --- IMPROVED: CNN-Optimized Preprocessing Function ---
def preprocess_image_for_cnn(image_data, show_steps=False):
    """
    Enhanced preprocessing specifically optimized for CNN models.
    CNNs expect 28x28 images, so we'll process to that size directly.
    """
    if image_data is None or np.sum(image_data) == 0:
        return None, None

    # Convert from RGBA to Grayscale (take alpha into account)
    if len(image_data.shape) == 3 and image_data.shape[2] == 4:  # RGBA
        # For RGBA, we need to handle transparency properly
        rgb = image_data[:, :, :3]
        alpha = image_data[:, :, 3] / 255.0
        
        # Convert to grayscale considering alpha
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # Apply alpha channel - where alpha is 0 (transparent), make it white (255)
        gray = (gray * alpha + 255 * (1 - alpha)).astype(np.uint8)
    elif len(image_data.shape) == 3 and image_data.shape[2] == 3:  # RGB
        gray = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_data.copy()

    # Step 1: Invert to match MNIST training data (black background, white digits)
    inverted = 255 - gray

    # Step 2: Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)

    # Step 3: Find contours to get the digit region
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest_contour) < 200:
        return None, None

    # Step 4: Get bounding box and extract digit
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add some padding
    padding = 20
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(cleaned.shape[1], x + w + padding)
    y_end = min(cleaned.shape[0], y + h + padding)
    
    digit_roi = cleaned[y_start:y_end, x_start:x_end]

    # Step 5: Make the image square by padding the shorter dimension
    h, w = digit_roi.shape
    if h > w:
        # Height is greater, pad width
        pad_width = (h - w) // 2
        digit_roi = cv2.copyMakeBorder(digit_roi, 0, 0, pad_width, pad_width, cv2.BORDER_CONSTANT, value=0)
    elif w > h:
        # Width is greater, pad height
        pad_height = (w - h) // 2
        digit_roi = cv2.copyMakeBorder(digit_roi, pad_height, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=0)

    # Step 6: Resize to 28x28 (MNIST standard)
    digit_28x28 = cv2.resize(digit_roi, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Step 7: Apply Gaussian blur to smooth like MNIST
    blurred = cv2.GaussianBlur(digit_28x28, (3, 3), 1.0)
    
    # Step 8: Normalize to 0-1 range (as expected by CNN)
    normalized = blurred.astype('float32') / 255.0
    
    # Step 9: Reshape for CNN input (add batch and channel dimensions)
    # Shape should be (1, 28, 28, 1) for the CNN
    final_image = normalized.reshape(1, 28, 28, 1)
    
    steps_info = None
    if show_steps:
        steps_info = {
            'original_gray': gray,
            'inverted': inverted,
            'cleaned': cleaned,
            'digit_roi': digit_roi,
            'digit_28x28': digit_28x28,
            'blurred': blurred,
            'normalized': normalized
        }

    return final_image, steps_info

def get_prediction_probabilities_cnn(image_data, show_debug=False):
    """Get prediction probabilities using CNN model with debug info"""
    processed_img, steps_info = preprocess_image_for_cnn(image_data, show_steps=show_debug)
    
    if processed_img is None:
        return None, None, None
    
    # Get prediction probabilities from CNN
    probabilities = model.predict(processed_img, verbose=0)[0]
    
    # Debug info
    debug_info = {
        'processed_shape': processed_img.shape,
        'processed_min': processed_img.min(),
        'processed_max': processed_img.max(),
        'processed_mean': processed_img.mean(),
        'processed_std': processed_img.std(),
        'raw_image_shape': image_data.shape if image_data is not None else None,
        'raw_image_sum': np.sum(image_data) if image_data is not None else None
    }
    
    return probabilities, debug_info, steps_info

def get_fun_message(digit, confidence):
    messages = {
        0: "Zero! Like the number of times you'll be wrong! 🎯",
        1: "One! You're number one at drawing! 🏆",
        2: "Two! It takes two to tango! 💃",
        3: "Three! Third time's the charm! ✨",
        4: "Four! May the force be with you! ⭐",
        5: "Five! Give me five! 🙏",
        6: "Six! You're sick at drawing numbers! 😄",
        7: "Seven! Lucky number seven! 🍀",
        8: "Eight! Don't be late, that's an eight! ⏰",
        9: "Nine! You're doing just fine! 😊"
    }
    return messages.get(digit, "Amazing number!")

# --- Main Content ---
# Title with animation
st.markdown('<div class="title">🌟 Magic Number Reader 🌟</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Draw a number and watch the AI magic happen! ✨</div>', unsafe_allow_html=True)

# Model info
st.markdown("""
<div class="model-info">
    🧠 <strong>Powered by Deep Learning!</strong><br>
    Using a Convolutional Neural Network trained on 70,000 digit images!
</div>
""", unsafe_allow_html=True)

# Fun instructions
st.markdown("""
<div class="instructions">
    <h3>🎯 How to Play:</h3>
    <p>1. Draw any digit (0-9) in the magic box below</p>
    <p>2. Click the "Predict My Number!" button</p>
    <p>3. Watch the AI guess your number!</p>
    <p>4. Try different numbers and see how smart the AI is!</p>
</div>
""", unsafe_allow_html=True)

# Debug toggle
show_debug = False
show_processing_steps = False

# Canvas section
st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
st.markdown('<div class="canvas-title">🎨 Draw Your Number Here!</div>', unsafe_allow_html=True)

# Canvas with better settings
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=20,
    stroke_color="#000000",
    background_color="#FFFFFF",
    background_image=None,
    update_streamlit=True,
    height=300,
    width=300,
    drawing_mode="freedraw",
    point_display_radius=0,
    display_toolbar=False,
    key=f"canvas_{st.session_state.canvas_key}",
)

st.markdown('</div>', unsafe_allow_html=True)

# --- Button Section ---
col1, col2 = st.columns([1, 1])

with col1:
    predict_btn = st.button("🤖 Predict My Number!", key="predict")
    
with col2:
    if st.button("🧹 Clear Canvas", key="clear"):
        st.session_state.canvas_key += 1
        st.rerun()

# --- Prediction Logic ---
if predict_btn and canvas_result.image_data is not None:
    st.markdown("---")
    
    # Check if canvas is empty
    if np.sum(canvas_result.image_data) == 0:
        st.warning("🎨 Oops! Draw something first before I can make a prediction!")
    else:
        # Show debug info if enabled
        if show_debug:
            st.markdown('<div class="debug-container">', unsafe_allow_html=True)
            st.write("**Debug - Canvas Data Info:**")
            st.write(f"Canvas shape: {canvas_result.image_data.shape}")
            st.write(f"Canvas data type: {canvas_result.image_data.dtype}")
            st.write(f"Canvas sum (total pixel values): {np.sum(canvas_result.image_data)}")
            st.write(f"Canvas min/max: {canvas_result.image_data.min()} / {canvas_result.image_data.max()}")
            
            # Show raw canvas image
            st.write("**Raw Canvas Image:**")
            st.image(canvas_result.image_data, caption="Raw canvas data", width=150)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Thinking animation
        with st.spinner("🧠 The deep learning AI is analyzing your drawing..."):
            probabilities, debug_info, steps_info = get_prediction_probabilities_cnn(
                canvas_result.image_data, show_debug=show_processing_steps
            )
            time.sleep(1)
        
        if show_debug and debug_info:
            st.markdown('<div class="debug-container">', unsafe_allow_html=True)
            st.write("**Debug - CNN Preprocessing Info:**")
            for key, value in debug_info.items():
                st.write(f"{key}: {value}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show processing steps if enabled
        if show_processing_steps and steps_info:
            st.markdown('<div class="debug-container">', unsafe_allow_html=True)
            st.write("**Image Processing Steps for CNN:**")
            
            cols = st.columns(3)
            step_names = ['original_gray', 'inverted', 'cleaned', 'digit_roi', 'digit_28x28', 'normalized']
            
            for i, step_name in enumerate(step_names):
                if step_name in steps_info:
                    with cols[i % 3]:
                        st.image(steps_info[step_name], caption=step_name.replace('_', ' ').title(), width=100)
            
            # Show the final 28x28 result as a heatmap
            st.write("**Final 28x28 image as seen by the CNN:**")
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(steps_info['normalized'], cmap='gray', interpolation='nearest')
            ax.set_title('Final 28x28 Image for CNN')
            plt.colorbar(im)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if probabilities is None:
            st.error("❌ Could not process the image. Please try drawing something bigger and clearer!")
        else:
            digit = np.argmax(probabilities)
            confidence = np.max(probabilities) * 100
            
            # Show all probabilities in debug mode
            if show_debug:
                st.markdown('<div class="debug-container">', unsafe_allow_html=True)
                st.write("**Debug - CNN Probabilities:**")
                for i, prob in enumerate(probabilities):
                    st.write(f"Digit {i}: {prob:.4f} ({prob*100:.1f}%)")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Confidence check
            if confidence < 40:
                st.warning(f"🤔 Hmm, I'm only {confidence:.0f}% sure, but I think it's a {digit}. Can you try drawing it a bit clearer?")
            else:
                # Success animations
                st.balloons()
                
                # Main prediction display
                fun_message = get_fun_message(digit, confidence)
                st.markdown(f"""
                <div class="prediction-box">
                    🎉 I think it's a <span class="prediction-digit">{digit}</span>! 🎉<br>
                    <div style="font-size: 1.2rem; margin-top: 10px;">{fun_message}</div>
                    <div style="font-size: 1rem; margin-top: 10px; color: #7F8C8D;">
                        CNN Confidence: {confidence:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Fun facts
                fun_facts = {
                    0: "Zero is the only number that is neither positive nor negative!",
                    1: "One is the first positive integer and the basis of our counting system!",
                    2: "Two is the only even prime number!",
                    3: "Three is considered a lucky number in many cultures!",
                    4: "Four is the number of sides in a square!",
                    5: "Five is the number of fingers on a human hand!",
                    6: "Six is the first perfect number (1+2+3=6)!",
                    7: "Seven is often considered the luckiest number!",
                    8: "Eight turned sideways is the infinity symbol ∞!",
                    9: "Nine is the highest single-digit number!"
                }
                
                st.markdown(f"""
                <div class="fun-facts">
                    <h4>🤓 Fun Fact about {digit}:</h4>
                    <p>{fun_facts.get(digit, "Numbers are amazing!")}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence chart
                st.subheader("📊 CNN Confidence Analysis")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#FF6B6B' if i == digit else '#A8E6CF' for i in range(10)]
                bars = ax.bar(range(10), probabilities * 100, color=colors, alpha=0.8)
                
                # Highlight the predicted digit
                bars[digit].set_color('#E74C3C')
                bars[digit].set_alpha(1.0)
                
                ax.set_xticks(range(10))
                ax.set_xlabel("Digit", fontsize=12, fontweight='bold')
                ax.set_ylabel("Confidence (%)", fontsize=12, fontweight='bold')
                ax.set_ylim(0, 100)
                ax.set_title("Deep Learning CNN Confidence for Each Digit", fontsize=14, fontweight='bold')
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)

# Footer
st.markdown('<br><br>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #450; font-family: 'Nunito', sans-serif;">
    <p>🎨 Powered by Deep Learning & CNN! Much more accurate than traditional models! 🚀</p>
</div>
""", unsafe_allow_html=True)