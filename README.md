# Advanced Image Processing App

An interactive web application for image processing and computer vision operations using Streamlit.

## Features

- **Color Space Conversion**: RGB, Grayscale, HSV, Lab with channel histograms
- **Image Filtering**: Gaussian blur, Median blur, Sobel filter
- **Edge Detection**: Canny edge detector
- **Binary Image Processing**: Binary, Otsu, and Adaptive thresholding
- **Corner Detection**: Harris and Shi-Tomasi corner detectors
- **Feature Extraction**: ORB and SIFT keypoint detection
- **Image Segmentation**: Contour-based and Watershed segmentation
- **PCA Visualization**: Dimensionality reduction of image features

## Requirements

- Python 3.8+ 
- OpenCV
- NumPy
- Streamlit
- scikit-learn
- Matplotlib
- Pillow

## Setup and Usage

### Option 1: Using the run script

```bash
# Make the script executable if not already
chmod +x run_advanced_app.sh

# Run the application
./run_advanced_app.sh
```

### Option 2: Manual setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements_advanced.txt

# Run the application
streamlit run advanced_app.py
```

## How to Use

1. Upload an image using the file uploader
2. Select an operation type from the dropdown menu
3. Configure any additional parameters if available
4. Click the action button to process the image
5. View the results and analysis on the right side

## Notes

- Some operations like SIFT feature extraction might not be available in all OpenCV versions
- For large images, some operations might take a while to complete
- For PCA visualization, a sufficient number of features must be detected in the image

## Example Use Cases

1. **Education**: Demonstrating image processing concepts
2. **Image Analysis**: Quick visual analysis of different image characteristics
3. **Feature Visualization**: Understanding feature extraction algorithms
4. **Computer Vision Prototyping**: Testing different approaches before implementation
