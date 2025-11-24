GNR 607 — Satellite Image Processing (RGB & HSI Component Enhancement)

A lightweight Streamlit app that applies seven types of component-level enhancements to a single RGB image. 
You can select from three enhancement functions (Linear, Logarithmic, Exponential) and apply them to either 
all RGB channels, a single RGB channel, or one of the HSI components (Hue, Saturation, Intensity). 
The enhanced output is displayed in the UI and can be downloaded.

------------------------------------------------------------

Features

- Upload one RGB image (JPEG/PNG).
- Choose one of 7 enhancement targets:
  - RGB: All Channels
  - RGB: Red Only
  - RGB: Green Only
  - RGB: Blue Only
  - HSI: Hue
  - HSI: Saturation
  - HSI: Intensity
- Choose one of 3 enhancement functions:
  - Linear
  - Logarithmic
  - Exponential
- Preview original and enhanced images.
- Download the enhanced image.

------------------------------------------------------------

Requirements

Python 3.8+

Install required packages:

streamlit
numpy
opencv-python
pillow
matplotlib

Using pip:

pip install streamlit numpy opencv-python pillow matplotlib

------------------------------------------------------------

Run the App

Place your code into `app.py`, then run:

streamlit run app.py

This opens the UI in your browser at http://localhost:8501.

------------------------------------------------------------

Usage Steps

1. Upload one RGB image (.jpg, .jpeg, .png)
2. Original image appears on the screen
3. Select enhancement type (RGB/HSI component)
4. Select enhancement method (Linear/Log/Exponential)
5. Click Run Enhancement
6. Enhanced output image appears
7. Click Download Enhanced Image

A copy is also saved inside the folder: `enhanced_output/enhanced_output.png`

------------------------------------------------------------

Suggested File Structure

project-folder/
│
├─ app.py
├─ enhanced_output/
└─ README.txt

------------------------------------------------------------
Notes

- RGB enhancements operate by splitting and modifying channels.
- HSI conversion is done manually, so slight variations from library implementations may occur.
- Very large images may slow processing.

------------------------------------------------------------

License

Free to use and modify for educational purposes.

