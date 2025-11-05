import streamlit as st
import os
import subprocess
from PIL import Image
import sys
import cv2
import numpy as np
import render_utils
from streamlit_image_comparison import image_comparison 

WALLPAPER_FOLDER = "wallpapers"
CAMERA_IMAGE_FILE = "captured_room.jpg"
MASK_OUTPUT_FILE = "wall_mask.png" 
st.set_page_config(page_title="AI Wallpaper Visualizer", layout="wide")
st.title("AI Wallpaper Visualizer")

if "room_path" not in st.session_state:
    st.session_state.room_path = None
if "selected_wallpaper_path" not in st.session_state:
    st.session_state.selected_wallpaper_path = None
if "mask_path" not in st.session_state:
    st.session_state.mask_path = None
if "room_image_cv" not in st.session_state:
    st.session_state.room_image_cv = None
if "texture_image_cv" not in st.session_state:
    st.session_state.texture_image_cv = None
if "camera_is_open" not in st.session_state:
    st.session_state.camera_is_open = False

st.subheader("Step 1: Upload Your Room Image")

col1, col2 = st.columns(2)
with col1:
    uploaded_room = st.file_uploader("Upload a room image", type=["jpg", "jpeg", "png"])
with col2:
    st.write("Capture now")
    if st.session_state.camera_is_open:
        if st.button("Close Camera"):
            st.session_state.camera_is_open = False
            st.rerun()
    else:
        if st.button("Open Camera"):
            st.session_state.camera_is_open = True
            st.rerun()

camera_capture = None
if st.session_state.camera_is_open:
    camera_capture = st.camera_input(
        "Capture from webcam", 
        label_visibility="collapsed"
    )

if uploaded_room:
    with open("uploaded_room.jpg", "wb") as f:
        f.write(uploaded_room.getbuffer())
    st.session_state.room_path = "uploaded_room.jpg"
    st.session_state.camera_is_open = False
elif camera_capture:
    with open(CAMERA_IMAGE_FILE, "wb") as f:
        f.write(camera_capture.getbuffer())
    st.session_state.room_path = CAMERA_IMAGE_FILE
    st.session_state.camera_is_open = False

if st.session_state.room_path:
    st.image(st.session_state.room_path, caption="Current Room", use_container_width=True)
else:
    st.info("Please upload or capture an image to begin.")

# wallpaper part
if st.session_state.room_path:
    st.markdown("---")
    st.subheader("Step 2: Select a Wallpaper")

    wallpapers = [
        os.path.join(WALLPAPER_FOLDER, f)
        for f in os.listdir(WALLPAPER_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", "png"))
    ]

    selected_wallpaper = None

    if wallpapers:
        st.write("Click on a wallpaper below to select it:")
        cols = st.columns(4)
        for i, wp_path in enumerate(wallpapers):
            with cols[i % 4]:
                wp_name = os.path.basename(wp_path)
                try:
                    img = Image.open(wp_path)
                    img.thumbnail((250, 250))
                    st.image(img, caption=wp_name, use_container_width=True) # Fixed use_column_width
                except Exception as e:
                    st.warning(f"Couldn't load {wp_name}: {e}")

                if st.button(f"Select {wp_name}", key=f"select_{i}"):
                    st.session_state.selected_wallpaper_path = wp_path # Fixed session state key
                    st.success(f"Selected: {wp_name}")

    else:
        st.warning("No wallpapers found. Please add some first.")

    if "selected_wallpaper_path" in st.session_state and st.session_state.selected_wallpaper_path: # Check not None
        selected_wallpaper = st.session_state.selected_wallpaper_path
        st.markdown("---")
        st.subheader("Selected Wallpaper Preview")
        img = Image.open(selected_wallpaper)
        img.thumbnail((400, 400))
        st.image(img, caption=os.path.basename(selected_wallpaper))
    
    if st.session_state.room_path and st.session_state.selected_wallpaper_path:
        st.markdown("---")
        st.subheader("Step 3: Find Walls")
        
        multi_wall = st.checkbox("Apply to all detected walls (multi-wall mode)", value=True)

        if st.button("Find Walls (This takes ~1 minute)"):
            with st.spinner("Loading AI models and analyzing room... Please wait."):
                cmd = [
                    sys.executable, "run_pipeline2.py",
                    "--room", st.session_state.room_path,
                    "--output_mask", MASK_OUTPUT_FILE
                ]
                if not multi_wall:
                    cmd.append("--single_wall")
                
                print(f"Running command: {' '.join(cmd)}")
                try:
                    result = subprocess.run(
                        cmd, check=True, capture_output=True, text=True, timeout=300
                    )
                    print("Pipeline STDOUT:", result.stdout)
                    
                    if os.path.exists(MASK_OUTPUT_FILE):
                        st.success("Wall analysis complete!")
                        st.session_state.mask_path = MASK_OUTPUT_FILE
                        st.session_state.room_image_cv = cv2.imread(st.session_state.room_path)
                        st.session_state.texture_image_cv = cv2.imread(st.session_state.selected_wallpaper_path)
                        
                        st.rerun()
                    else:
                        st.error("AI Pipeline ran but failed to create a mask file.")
                        st.code(result.stdout)
                        st.code(result.stderr)
                
                except subprocess.CalledProcessError as e:
                    st.error(f"Error finding walls (Exit Code {e.returncode}):")
                    st.code(e.stderr, language="bash")
                except subprocess.TimeoutExpired:
                    st.error("Error: The AI process timed out.")

if st.session_state.mask_path:
    st.markdown("---")
    st.subheader("Step 4: Live Adjustments")
    st.info("All adjustments below are real-time.")
    mask_image = cv2.imread(st.session_state.mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        st.error("Failed to load the wall mask.")
        st.session_state.mask_path = None 
    elif st.session_state.room_image_cv is None or st.session_state.texture_image_cv is None:
        st.warning("Image data lost. Re-loading images...")
        st.session_state.room_image_cv = cv2.imread(st.session_state.room_path)
        st.session_state.texture_image_cv = cv2.imread(st.session_state.selected_wallpaper_path)
        if st.session_state.room_image_cv is None or st.session_state.texture_image_cv is None:
            st.error("Failed to reload images. Please start over from Step 1.")
            st.session_state.clear() 
            st.stop()
    else:
        mask_h, mask_w = mask_image.shape[:2]
        room_h, room_w, _ = st.session_state.room_image_cv.shape
        if (room_h, room_w) != (mask_h, mask_w):
            print(f"Resizing source image from {room_w}x{room_h} to match mask {mask_w}x{mask_h}")
            st.session_state.room_image_cv = cv2.resize(
                st.session_state.room_image_cv, 
                (mask_w, mask_h), 
                interpolation=cv2.INTER_AREA
            )
        zoom = st.slider("Texture Zoom (Pattern Size)", 0.2, 3.0, 1.0, 0.1)
        brightness = st.slider("Brightness", 0, 100, 50, 1)
        contrast = st.slider("Contrast", 0, 100, 50, 1)
        saturation = st.slider("Saturation", 0, 100, 50, 1)

        final_image_bgr = render_utils.apply_design(
            st.session_state.room_image_cv,
            st.session_state.texture_image_cv,
            mask_image,
            zoom,
            brightness,
            contrast,
            saturation
        )
        
        final_image_rgb = cv2.cvtColor(final_image_bgr, cv2.COLOR_BGR2RGB)
        room_image_rgb = cv2.cvtColor(st.session_state.room_image_cv, cv2.COLOR_BGR2RGB)
        
        st.markdown("---")
        st.subheader("Final Preview")
        st.image(final_image_rgb, caption="Adjusted Preview", use_container_width=True)

        st.markdown("---")
        st.subheader("Before / After Comparison")
        image_comparison(
            img1=room_image_rgb,
            img2=final_image_rgb,
            label1="Before",
            label2="After"
        )

