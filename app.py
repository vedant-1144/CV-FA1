import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt

st.title("Interactive Image Processing App")

uploaded_file = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])
if uploaded_file:
    # Read image as numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display original image
    st.image(img, caption='Original Image', use_container_width=True)

    operation = st.selectbox("Select Operation", [
        'Convert Color Space',
        'Apply Filter',
        'Edge Detection',
        'Binary Thresholding',
        'Corner Detection',
        'Feature Extraction',
        'Segmentation',
        'PCA Demo'
    ])

    if st.button("Run Operation"):
        result_img = None
        info_text = ""

        if operation == 'Convert Color Space':
            option = st.radio("Choose Color Space", ['Grayscale', 'HSV', 'Lab'])
            if option == 'Grayscale':
                result_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif option == 'HSV':
                result_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif option == 'Lab':
                result_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        elif operation == 'Apply Filter':
            option = st.radio("Choose Filter", ['Gaussian', 'Median', 'Sobel'])
            if option == 'Gaussian':
                result_img = cv2.GaussianBlur(img, (7,7), 0)
            elif option == 'Median':
                result_img = cv2.medianBlur(img, 7)
            elif option == 'Sobel':
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
                result_img = cv2.convertScaleAbs(sobelx)

        elif operation == 'Edge Detection':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            result_img = edges

        elif operation == 'Binary Thresholding':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result_img = thresh

        elif operation == 'Corner Detection':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            corners = cv2.cornerHarris(gray, 2, 3, 0.04)
            result_img = img.copy()
            result_img[corners > 0.01 * corners.max()] = [0,0,255]
            info_text = f"Number of corners detected: {np.sum(corners > 0.01 * corners.max())}"

        elif operation == 'Feature Extraction':
            # ORB example (SIFT/SURF need contrib)
            orb = cv2.ORB_create()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints = orb.detect(gray, None)
            result_img = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0), flags=0)
            info_text = f"Number of keypoints detected: {len(keypoints)}"

        elif operation == 'Segmentation':
            # Simple contour extraction demo
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result_img = img.copy()
            cv2.drawContours(result_img, contours, -1, (0,255,0), 2)
            info_text = f"Number of contours found: {len(contours)}"

        elif operation == 'PCA Demo':
            from sklearn.decomposition import PCA
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pca = PCA(n_components=2)
            h, w = gray.shape
            flattened = gray.reshape(h*w,1)
            pca_features = pca.fit_transform(flattened)
            info_text = f"Explained variance ratio: {pca.explained_variance_ratio_}"
            st.write(info_text)
            result_img = gray  # just show grayscale for simplicity

        if result_img is not None:
            # Display processed image
            st.image(result_img, caption=f"Result of {operation}", use_container_width=True)
        if info_text:
            st.write(info_text)