import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Set page configuration
st.set_page_config(
    page_title="Advanced Image Processing App",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main headers and content styling */
    .main-header {
        font-size: 2.5rem;
        color: #1f66c2;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .category-header {
        font-size: 1.5rem;
        color: #0d47a1;
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: 600;
    }
    
    .result-header {
        font-size: 1.5rem;
        color: #1b5e20;
        margin-top: 1rem;
        font-weight: 600;
    }
    
    /* Sidebar specific styling for dark mode */
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    
    /* Make sidebar headers light colored */
    .sidebar h1, .sidebar h2, .sidebar h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
    }
    
    /* Make sidebar labels and text light colored */
    .sidebar label, .sidebar p, .sidebar .stMarkdown, .sidebar text {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* Style dropdown text in sidebar */
    .sidebar .stSelectbox > div > div > div {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* Style dropdown options */
    .sidebar .stSelectbox [data-baseweb="select"] {
        color: #ffffff !important;
    }
    
    /* Improve the contrast of dropdown boxes */
    .sidebar .stSelectbox [data-baseweb="select"] {
        background-color: rgba(151, 166, 195, 0.15) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Make slider text visible */
    .sidebar .stSlider label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* Style buttons */
    .stButton button {
        background-color: #4c8bf5 !important;
        color: white !important;
        font-weight: bold !important;
        width: 100%;
        border: none !important;
    }
    
    .stButton button:hover {
        background-color: #3a70d6 !important;
    }
    
    .info-box {
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #1976d2;
        margin-bottom: 10px;
        font-weight: 500;
    }
    
    /* Make section titles in sidebar more prominent */
    .sidebar-section-header {
        color: #4c8bf5 !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        margin-top: 1.5rem !important;
        border-bottom: 1px solid #4c8bf5;
        padding-bottom: 0.5rem;
    }
    
    /* Make file uploader text visible */
    [data-testid="stFileUploader"] label {
        color: #ffffff !important;
    }
    
    /* Make radio buttons and checkboxes visible */
    .sidebar [data-testid="stRadio"] label, .sidebar [data-testid="stCheckbox"] label {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Functions for Image Processing Operations
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_color_space(image, target_space):
    if target_space == "RGB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif target_space == "Grayscale":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif target_space == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif target_space == "Lab":
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return image

def apply_filter(image, filter_type, params=None):
    if filter_type == "Gaussian":
        ksize = params.get("ksize", 5)
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif filter_type == "Median":
        ksize = params.get("ksize", 5)
        return cv2.medianBlur(image, ksize)
    return image

def detect_edges(image, edge_detector, threshold1=100, threshold2=200):
    if edge_detector == "Canny":
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(image, threshold1, threshold2)
    elif edge_detector == "Sobel":
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        return cv2.convertScaleAbs(sobel_combined)
    return image

def apply_thresholding(image, threshold_type, threshold_value=127):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if threshold_type == "Binary":
        _, thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    elif threshold_type == "Adaptive":
        thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
    else:
        thresholded = image
    
    return thresholded

def detect_corners(image, corner_detector):
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Convert to float32
    gray = np.float32(gray)
    
    if corner_detector == "Harris":
        corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        corners = cv2.dilate(corners, None)
        
        if len(image.shape) > 2:
            result = image.copy()
        else:
            result = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
        result[corners > 0.01 * corners.max()] = [0, 0, 255]
        corner_count = np.sum(corners > 0.01 * corners.max())
        
        return result, corner_count
        
    elif corner_detector == "Shi-Tomasi":
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.astype(np.uint8)
            
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        
        if len(image.shape) > 2:
            result = image.copy()
        else:
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(result, (int(x), int(y)), 3, [0, 0, 255], -1)
                
            corner_count = len(corners)
        else:
            corner_count = 0
            
        return result, corner_count
    
    return image, 0

def extract_features(image, feature_extractor):
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if feature_extractor == "ORB":
        orb = cv2.ORB_create(nfeatures=500)
        keypoints = orb.detect(gray, None)
        keypoints, descriptors = orb.compute(gray, keypoints)
        img_with_keypoints = cv2.drawKeypoints(image.copy() if len(image.shape) > 2 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 
                                              keypoints, None, color=(0, 255, 0), 
                                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        return img_with_keypoints, len(keypoints), descriptors
    
    elif feature_extractor == "SIFT":
        try:
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            img_with_keypoints = cv2.drawKeypoints(image.copy() if len(image.shape) > 2 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 
                                                keypoints, None, color=(0, 255, 0),
                                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            return img_with_keypoints, len(keypoints), descriptors
        except:
            st.error("SIFT is not available in your OpenCV version. Try using ORB instead.")
            return image, 0, None
            
    return image, 0, None

def segment_image(image, segmentation_type):
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if segmentation_type == "Contours":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        if len(image.shape) > 2:
            result = image.copy()
        else:
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        return result, len(contours)
        
    return image, 0

def apply_pca(descriptors, n_components=10):
    if descriptors is None or len(descriptors) < n_components:
        return None, None
    
    # Fit PCA
    pca = PCA(n_components=min(n_components, descriptors.shape[1]))
    pca_result = pca.fit_transform(descriptors)
    
    # Create a figure for the explained variance
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Explained Variance')
    plt.tight_layout()
    
    return fig, pca_result

def process_and_display(operation, image, params, col1, col2):
    """Process image based on operation and display results in the given columns"""
    
    # For Color Space operations
    if operation == "Color Spaces":
        color_space = params["color_space"]
        processed_image = convert_color_space(image, color_space)
        
        # Display result
        with col2:
            st.markdown(f"<h3 class='result-header'>Image in {color_space} Color Space</h3>", unsafe_allow_html=True)
            
            # For RGB, HSV and Lab spaces
            if color_space in ["RGB", "HSV", "Lab"]:
                if color_space == "RGB":
                    st.image(processed_image, channels="RGB", use_container_width=True)
                else:
                    st.image(processed_image, use_container_width=True)
                
                # Split channels and show histograms
                st.markdown("<h3>Channel Histograms</h3>", unsafe_allow_html=True)
                channel_names = {
                    "RGB": ["Red", "Green", "Blue"],
                    "HSV": ["Hue", "Saturation", "Value"],
                    "Lab": ["Lightness", "a* (green-red)", "b* (blue-yellow)"]
                }
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                
                for i, (channel, name) in enumerate(zip(cv2.split(processed_image), channel_names[color_space])):
                    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
                    axes[i].plot(hist)
                    axes[i].set_title(name)
                    axes[i].set_xlim([0, 256])
                
                plt.tight_layout()
                st.pyplot(fig)
                
            # For grayscale
            else:
                st.image(processed_image, use_container_width=True)
                
                # Show histogram
                st.markdown("<h3>Grayscale Histogram</h3>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 4))
                hist = cv2.calcHist([processed_image], [0], None, [256], [0, 256])
                ax.plot(hist)
                ax.set_xlim([0, 256])
                st.pyplot(fig)
                
    # For Filtering operations
    elif operation == "Filters":
        filter_type = params["filter_type"]
        if filter_type == "Sobel":
            processed_image = apply_filter(image, filter_type)
        else:
            processed_image = apply_filter(image, filter_type, {"ksize": params["ksize"]})
        
        # Display results
        with col2:
            st.markdown(f"<h3 class='result-header'>Image with {filter_type} Filter</h3>", unsafe_allow_html=True)
            st.image(processed_image, use_container_width=True)
    
    # For Edge Detection
    elif operation == "Edge Detection":
        edge_detector = params["edge_detector"]
        threshold1 = params.get("threshold1", 100)
        threshold2 = params.get("threshold2", 200)
        
        processed_image = detect_edges(image, edge_detector, threshold1, threshold2)
        
        # Display results
        with col2:
            st.markdown(f"<h3 class='result-header'>Edge Detection using {edge_detector}</h3>", unsafe_allow_html=True)
            st.image(processed_image, use_container_width=True)
    
    # For Thresholding
    elif operation == "Binary Image":
        threshold_type = params["threshold_type"]
        threshold_value = params.get("threshold_value", 127)
        
        processed_image = apply_thresholding(image, threshold_type, threshold_value)
        
        # Display results
        with col2:
            st.markdown(f"<h3 class='result-header'>Image after {threshold_type} Thresholding</h3>", unsafe_allow_html=True)
            st.image(processed_image, use_container_width=True)
    
    # For Corner Detection
    elif operation == "Corner Detection":
        corner_detector = params["corner_detector"]
        
        processed_image, corner_count = detect_corners(image, corner_detector)
        
        # Display results
        with col2:
            st.markdown(f"<h3 class='result-header'>Corner Detection using {corner_detector}</h3>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown(f"<div class='info-box'><span style='color: #212121; font-weight: 600;'>Number of corners detected: {corner_count}</span></div>", unsafe_allow_html=True)
    
    # For Feature Extraction
    elif operation == "Feature Extraction":
        feature_extractor = params["feature_extractor"]
        
        processed_image, keypoint_count, descriptors = extract_features(image, feature_extractor)
        
        # Display results
        with col2:
            st.markdown(f"<h3 class='result-header'>Feature Extraction using {feature_extractor}</h3>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown(f"<div class='info-box'><span style='color: #212121; font-weight: 600;'>Number of keypoints detected: {keypoint_count}</span></div>", unsafe_allow_html=True)
        
        # Return descriptors for potential PCA use
        return descriptors
    
    # For Segmentation
    elif operation == "Segmentation":
        segmentation_type = params["segmentation_type"]
        
        processed_image, segment_count = segment_image(image, segmentation_type)
        
        # Display results
        with col2:
            st.markdown(f"<h3 class='result-header'>Image Segmentation using {segmentation_type}</h3>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown(f"<div class='info-box'><span style='color: #212121; font-weight: 600;'>Number of segments detected: {segment_count}</span></div>", unsafe_allow_html=True)
    
    # For PCA
    elif operation == "PCA":
        feature_extractor = params["feature_extractor"]
        n_components = params["n_components"]
        
        with st.spinner("Extracting features and applying PCA..."):
            # Extract features first
            _, keypoint_count, descriptors = extract_features(image, feature_extractor)
            
            if descriptors is not None and keypoint_count > 0:
                # Apply PCA
                pca_fig, pca_result = apply_pca(descriptors, n_components)
                
                # Display results
                with col2:
                    st.markdown(f"<h3 class='result-header'>PCA on {feature_extractor} Features</h3>", unsafe_allow_html=True)
                    st.markdown(f"<div class='info-box'><span style='color: #212121; font-weight: 600;'>Original feature dimensionality: {descriptors.shape[1]}</span><br><span style='color: #212121; font-weight: 600;'>Reduced to {n_components} principal components</span></div>", unsafe_allow_html=True)
                    
                    if pca_fig is not None:
                        st.pyplot(pca_fig)
                        
                    if pca_result is not None:
                        st.markdown("<h3>First few samples in PCA space</h3>", unsafe_allow_html=True)
                        st.write(pca_result[:5])
            else:
                with col2:
                    st.error(f"Could not extract features with {feature_extractor}. Try a different extractor.")
    
    return None  # Default return if not Feature Extraction

def main():
    # Header with custom styling
    st.markdown("<h1 class='main-header'>Advanced Image Processing App</h1>", unsafe_allow_html=True)
    
    # Sidebar for image upload and operation selection
    with st.sidebar:
        # Add sidebar section headers with improved styling
        st.markdown("<div class='sidebar-section-header'>Upload Image</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            st.markdown("<div class='sidebar-section-header'>Operation Categories</div>", unsafe_allow_html=True)
            
            # Group operations into categories
            categories = {
                "Color Operations": ["Color Spaces"],
                "Filtering & Edges": ["Filters", "Edge Detection"],
                "Binarization": ["Binary Image"],
                "Feature Analysis": ["Corner Detection", "Feature Extraction"],
                "Advanced": ["Segmentation", "PCA"]
            }
            
            # Create a dropdown for categories
            selected_category = st.selectbox("Select Category", list(categories.keys()))
            
            # Get operations for the selected category
            operations = categories[selected_category]
            
            # Only show operation dropdown if there are multiple operations in the category
            if len(operations) > 1:
                selected_operation = st.selectbox("Select Operation", operations)
            else:
                # If only one operation, just display it as text and use it
                selected_operation = operations[0]
                st.markdown(f"<div style='padding: 5px 10px; background-color: rgba(151, 166, 195, 0.15); border-radius: 4px; margin-bottom: 10px; color: white;'><strong>Operation:</strong> {selected_operation}</div>", unsafe_allow_html=True)
            
            # Additional parameters for the selected operation
            st.markdown(f"<div class='sidebar-section-header'>Parameters for {selected_operation}</div>", unsafe_allow_html=True)
            
            # Dynamic parameters based on operation
            params = {}
            if selected_operation == "Color Spaces":
                params["color_space"] = st.selectbox("Select Color Space", ["RGB", "Grayscale", "HSV", "Lab"])
                
            elif selected_operation == "Filters":
                params["filter_type"] = st.selectbox("Select Filter", ["Gaussian", "Median"])
                if params["filter_type"] in ["Gaussian", "Median"]:
                    params["ksize"] = st.slider("Kernel Size", 1, 25, 5, step=2)  # Ensure odd kernel size
            
            elif selected_operation == "Edge Detection":
                params["edge_detector"] = st.selectbox("Select Edge Detector", ["Canny", "Sobel"])
                params["threshold1"] = st.slider("Threshold 1", 0, 255, 100)
                params["threshold2"] = st.slider("Threshold 2", 0, 255, 200)
                
            elif selected_operation == "Binary Image":
                params["threshold_type"] = st.selectbox("Select Thresholding Method", ["Binary", "Adaptive"])
                if params["threshold_type"] == "Binary":
                    params["threshold_value"] = st.slider("Threshold Value", 0, 255, 127)
                
            elif selected_operation == "Corner Detection":
                params["corner_detector"] = st.selectbox("Select Corner Detector", ["Harris", "Shi-Tomasi"])
                
            elif selected_operation == "Feature Extraction":
                params["feature_extractor"] = st.selectbox("Select Feature Extractor", ["ORB", "SIFT"])
                
            elif selected_operation == "Segmentation":
                params["segmentation_type"] = st.selectbox("Select Segmentation Method", ["Contours"])
                
            elif selected_operation == "PCA":
                params["feature_extractor"] = st.selectbox("Select Feature Extractor for PCA", ["ORB", "SIFT"])
                params["n_components"] = st.slider("Number of PCA Components", 2, 20, 10)
            
            # Process button with improved visibility
            st.markdown("<br>", unsafe_allow_html=True)
            process_button = st.button("Process Image", key="process_image")

    # Main content area
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)
        
        # Display original image in the first column
        with col1:
            st.markdown("<h3 class='category-header'>Original Image</h3>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Display image info
            height, width, channels = image.shape if len(image.shape) > 2 else (*image.shape, 1)
            st.markdown(
                f"<div class='info-box'>"
                f"<b style='color: #0d47a1; font-size: 1.1rem;'>Image Information</b><br>"
                f"<span style='color: #212121; font-weight: 500;'>Width: {width} px</span><br>"
                f"<span style='color: #212121; font-weight: 500;'>Height: {height} px</span><br>"
                f"<span style='color: #212121; font-weight: 500;'>Channels: {channels}</span><br>"
                f"<span style='color: #212121; font-weight: 500;'>Size: {width * height * channels / 1024:.2f} KB</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        # Process image based on selected operation when button is clicked
        if 'process_button' in locals() and process_button:
            try:
                with st.spinner(f"Processing with {selected_operation}..."):
                    process_and_display(selected_operation, image, params, col1, col2)
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
    else:
        # Display instructions when no image is uploaded
        st.markdown("""
        <div style='padding: 2rem; background-color: #e3f2fd; border-radius: 10px; text-align: center; border: 2px solid #1976d2;'>
            <h2 style='color: #0d47a1; font-weight: bold;'>👋 Welcome to the Advanced Image Processing App!</h2>
            <p style='color: #0d47a1; font-size: 1.1rem; font-weight: 500;'>To get started, upload an image using the sidebar on the left.</p>
            <p style='color: #0d47a1; font-size: 1.1rem; font-weight: 500;'>Then select an operation category and specific operation to apply to your image.</p>
            <p style='color: #0d47a1; font-size: 1.1rem; font-weight: 500;'>You'll see the original and processed images side by side for easy comparison.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display feature overview with improved text visibility
        st.markdown("""
        <h2 style='margin-top: 2rem; color: #0d47a1; font-weight: bold;'>Features Available</h2>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
            <div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px; border: 1px solid #1976d2;'>
                <h3 style='color: #0d47a1; font-weight: bold;'>Color Operations</h3>
                <ul style='color: #212121; font-weight: 500;'>
                    <li>RGB, Grayscale, HSV, and Lab color spaces</li>
                    <li>Channel histograms</li>
                </ul>
            </div>
            <div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px; border: 1px solid #1976d2;'>
                <h3 style='color: #0d47a1; font-weight: bold;'>Filtering & Edges</h3>
                <ul style='color: #212121; font-weight: 500;'>
                    <li>Gaussian, Median, and Sobel filters</li>
                    <li>Canny edge detection</li>
                </ul>
            </div>
            <div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px; border: 1px solid #1976d2;'>
                <h3 style='color: #0d47a1; font-weight: bold;'>Binarization</h3>
                <ul style='color: #212121; font-weight: 500;'>
                    <li>Binary thresholding</li>
                    <li>Adaptive thresholding</li>
                </ul>
            </div>
            <div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px; border: 1px solid #1976d2;'>
                <h3 style='color: #0d47a1; font-weight: bold;'>Feature Analysis</h3>
                <ul style='color: #212121; font-weight: 500;'>
                    <li>Harris and Shi-Tomasi corner detection</li>
                    <li>ORB and SIFT feature extraction</li>
                </ul>
            </div>
            <div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px; border: 1px solid #1976d2;'>
                <h3 style='color: #0d47a1; font-weight: bold;'>Advanced Operations</h3>
                <ul style='color: #212121; font-weight: 500;'>
                    <li>Contour Segmentation</li>
                    <li>PCA dimensionality reduction</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
