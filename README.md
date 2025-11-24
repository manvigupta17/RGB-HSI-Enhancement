# GNR 607 — Satellite Image Processing (RGB & HSI Component Enhancement)

### **Created by**
- Mandakini Dalwee (23B0418)  
- Aman Khatri (23B3961)  
- Manvi Gupta (24B0697)

---

## 📌 Overview

This project is a **Streamlit app** that applies **seven types of component-level enhancements** to a single RGB image.  
Users can choose from **three enhancement functions** (Linear, Logarithmic, Exponential) and apply them to:

- All RGB channels  
- Individual RGB channels (R / G / B)  
- Individual HSI components (Hue / Saturation / Intensity)

The enhanced output is displayed in the UI and can be downloaded.

---

## ⭐ Features

### **Upload**
- Upload one RGB image (JPEG/PNG)

### **Select Enhancement Target (7 options)**
- RGB: All Channels  
- RGB: Red Only  
- RGB: Green Only  
- RGB: Blue Only  
- HSI: Hue  
- HSI: Saturation  
- HSI: Intensity  

### **Select Enhancement Function (3 options)**
- Linear  
- Logarithmic  
- Exponential  

### **Output**
- View original and enhanced images  
- Download the enhanced image  

---

## 📦 Requirements

**Python 3.8+**

Install required packages:

streamlit numpy opencv-python pillow matplotlib

Using pip:

pip install streamlit numpy opencv-python pillow matplotlib

---

## ▶️ Run the App
streamlit run app.py

This opens the UI in your browser at:  
**http://localhost:8501**

---

## 📝 Usage Steps

1. Upload one RGB image (.jpg, .jpeg, .png)  
2. Original image appears on the screen  
3. Select enhancement target (RGB/HSI component)  
4. Select enhancement method (Linear / Log / Exponential)  
5. Click **Run Enhancement**  
6. Enhanced output is displayed  
7. Click **Download Enhanced Image**  
8. A copy is saved  
   
---

## 📁 Suggested File Structure

project-folder/

app.py

enhanced_output/

README.txt

---






