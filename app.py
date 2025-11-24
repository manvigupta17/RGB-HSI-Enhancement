# Updated Streamlit app supporting single-image selection and choosing 1 of 7 enhancements

import os
import shutil
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import io
from PIL import Image

DEFAULT_OUT_DIR = "enhanced_output"
os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)

# --- Enhancement functions (unchanged) ---
def linear_enhancement(img):
    img = img.astype(np.float32) / 255.0
    a, b = 1.2, 0.05
    enhanced = np.clip(a * img + b, 0, 1)
    return (enhanced * 255).astype(np.uint8)

def log_enhancement(img):
    img = img.astype(np.float32) / 255.0
    c = 1.0 / (np.log(1 + np.max(img)) + 1e-12)
    enhanced = c * np.log(1 + img)
    return (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)

def exp_enhancement(img):
    img = img.astype(np.float32) / 255.0
    e = np.exp(img) - 1
    e /= (np.max(e) + 1e-12)
    return (np.clip(e, 0, 1) * 255).astype(np.uint8)

# --- HSI conversion functions ---
def rgb_to_hsi(rgb):
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]

    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g)**2 + (r - b)*(g - b)) + 1e-12
    theta = np.arccos(np.clip(num/den, -1, 1))
    h = np.where(b <= g, theta, 2*np.pi - theta) / (2*np.pi)

    min_rgb = np.minimum(np.minimum(r, g), b)
    s = 1 - (3/(r+g+b + 1e-12))*min_rgb
    i = (r+g+b)/3

    return np.stack([h, np.clip(s,0,1), np.clip(i,0,1)], axis=-1)

def hsi_to_rgb(hsi):
    h = hsi[...,0] * 2*np.pi
    s = hsi[...,1]
    i = hsi[...,2]

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    idx = h < 2*np.pi/3
    b[idx] = i[idx] * (1 - s[idx])
    r[idx] = i[idx] * (1 + s[idx] * np.cos(h[idx])/(np.cos(np.pi/3 - h[idx]) + 1e-12))
    g[idx] = 3*i[idx] - (r[idx] + b[idx])

    idx = (h >= 2*np.pi/3) & (h < 4*np.pi/3)
    h2 = h[idx] - 2*np.pi/3
    r[idx] = i[idx] * (1 - s[idx])
    g[idx] = i[idx] * (1 + s[idx] * np.cos(h2)/(np.cos(np.pi/3 - h2) + 1e-12))
    b[idx] = 3*i[idx] - (r[idx] + g[idx])

    idx = h >= 4*np.pi/3
    h3 = h[idx] - 4*np.pi/3
    g[idx] = i[idx] * (1 - s[idx])
    b[idx] = i[idx] * (1 + s[idx] * np.cos(h3)/(np.cos(np.pi/3 - h3) + 1e-12))
    r[idx] = 3*i[idx] - (g[idx] + b[idx])

    rgb = np.stack([r,g,b], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

# --- Enhancement applications ---
def apply_rgb_single(img, channel_idx, func):
    ch = list(cv2.split(img))
    ch[channel_idx] = func(ch[channel_idx])
    return cv2.merge(ch)

def apply_rgb_all(img, func):
    r,g,b = cv2.split(img)
    return cv2.merge([func(r), func(g), func(b)])

def apply_hsi_enh(img, comp_idx, func):
    hsi = rgb_to_hsi(img)
    comp = (hsi[...,comp_idx] * 255).astype(np.uint8)
    compE = func(comp).astype(np.float32) / 255.0
    hsi_mod = hsi.copy()
    hsi_mod[...,comp_idx] = compE
    return hsi_to_rgb(hsi_mod)

# --- Streamlit UI update ---
st.set_page_config(page_title="GNR 607", layout="wide")
st.title("Satellite Image Processing — RGB & HSI Component Enhancement ")

uploaded = st.file_uploader("Upload ONE RGB Image", type=["jpg","jpeg","png"])

enh_list = {
    "RGB: All Channels" : ("rgb_all", None),
    "RGB: Red Only" : ("rgb_single", 0),
    "RGB: Green Only" : ("rgb_single", 1),
    "RGB: Blue Only" : ("rgb_single", 2),
    "HSI: Hue" : ("hsi", 0),
    "HSI: Saturation" : ("hsi", 1),
    "HSI: Intensity" : ("hsi", 2)
}

enhancement_choice = st.selectbox("Select enhancement type", list(enh_list.keys()))
method_choice = st.selectbox("Select enhancement function", ["Linear","Logarithmic","Exponential"])

method_map = {
    "Linear": linear_enhancement,
    "Logarithmic": log_enhancement,
    "Exponential": exp_enhancement
}

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)
    st.image(img_np, caption="Original", use_column_width=True)

    if st.button("Run Enhancement"):
        enh_type, idx = enh_list[enhancement_choice]
        func = method_map[method_choice]

        if enh_type == "rgb_all":
            out = apply_rgb_all(img_np, func)
        elif enh_type == "rgb_single":
            out = apply_rgb_single(img_np, idx, func)
        else:
            out = apply_hsi_enh(img_np, idx, func)

        save_path = os.path.join(DEFAULT_OUT_DIR, "enhanced_output.png")
        cv2.imwrite(save_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        st.image(out, caption="Enhanced Output", use_column_width=True)

        with open(save_path, "rb") as f:
            st.download_button("Download Enhanced Image", data=f, file_name="enhanced.png", mime="image/png")
