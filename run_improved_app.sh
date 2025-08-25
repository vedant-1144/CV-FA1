#!/bin/bash

# Run the advanced image processing app with improved UI

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements_advanced.txt

# Run the improved Streamlit app
echo "Starting Improved Image Processing App..."
streamlit run advanced_app_improved.py
