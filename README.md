# Advanced Image Processing App

An interactive Streamlit application for applying various image processing techniques to images using OpenCV and Python.

## Features

- **Color Space Conversion**: RGB, Grayscale, HSV, and Lab color spaces with channel histograms
- **Filtering**: Gaussian, Median, and Sobel filters
- **Edge Detection**: Canny edge detector with customizable thresholds
- **Binarization**: Binary, Otsu, and Adaptive thresholding techniques
- **Corner Detection**: Harris and Shi-Tomasi corner detection methods
- **Feature Extraction**: ORB and SIFT keypoint detection and visualization
- **Image Segmentation**: Contour-based and Watershed segmentation algorithms
- **PCA Analysis**: Principal Component Analysis on extracted image features

## Getting Started

### Prerequisites

- Python 3.6+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run advanced_app_improved.py
```

## Usage

1. Upload an image using the sidebar file uploader
2. Select an operation category from the dropdown menu
3. Choose a specific operation to perform
4. Configure any parameters for the selected operation
5. Click the "Process Image" button to apply the operation
6. View the results displayed side-by-side with the original image

## License

[MIT License](LICENSE)
