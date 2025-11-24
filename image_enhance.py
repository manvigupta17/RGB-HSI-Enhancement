import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration
DEFAULT_COMPARE_DIR = "comparison_collages" 
os.makedirs(DEFAULT_COMPARE_DIR, exist_ok=True)

# Enhancement functions
def linear_enhancement(img):
    img = img.astype(np.float32) / 255.0
    a, b = 1.2, 0.05
    enhanced = np.clip(a * img + b, 0, 1)
    return (enhanced * 255).astype(np.uint8)

def log_enhancement(img):
    img = img.astype(np.float32) / 255.0
    # c * log(1 + r)
    c = 1.0 / (np.log(1 + np.max(img)) + 1e-12)
    enhanced = c * np.log(1 + img)
    enhanced = np.clip(enhanced, 0, 1)
    return (enhanced * 255).astype(np.uint8)

def exp_enhancement(img):
    img = img.astype(np.float32) / 255.0
    # c * (exp(r) - 1)
    c = np.exp(img) - 1
    c = c / (np.max(c) + 1e-12)
    enhanced = np.clip(c, 0, 1)
    return (enhanced * 255).astype(np.uint8)

# RGB <-> HSI conversion
def rgb_to_hsi(rgb):
    rgb = rgb.astype(np.float32) / 255.0
    R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B)*(G - B)) + 1e-6
    theta = np.arccos(np.clip(num / den, -1, 1))
    
    H = np.where(B <= G, theta, 2*np.pi - theta)
    H = H / (2*np.pi) # normalize H 
    
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-6)) * min_rgb
    I = (R + G + B) / 3
    
    return np.stack([H, S, I], axis=-1)

def hsi_to_rgb(hsi):
    H, S, I = hsi[:,:,0]*2*np.pi, hsi[:,:,1], hsi[:,:,2]
    R, G, B = np.zeros_like(H), np.zeros_like(H), np.zeros_like(H)
    
    # 0 <= H < 120
    idx = (H < 2*np.pi/3)
    B[idx] = I[idx]*(1 - S[idx])
    R[idx] = I[idx]*(1 + (S[idx]*np.cos(H[idx]))/(np.cos(np.pi/3 - H[idx]) + 1e-12))
    G[idx] = 3*I[idx] - (R[idx] + B[idx])
    
    # 120 <= H < 240
    idx = (H >= 2*np.pi/3) & (H < 4*np.pi/3)
    H2 = H[idx] - 2*np.pi/3
    R[idx] = I[idx]*(1 - S[idx])
    G[idx] = I[idx]*(1 + (S[idx]*np.cos(H2))/(np.cos(np.pi/3 - H2) + 1e-12))
    B[idx] = 3*I[idx] - (R[idx] + G[idx])
    
    # 240 <= H <= 360
    idx = (H >= 4*np.pi/3)
    H3 = H[idx] - 4*np.pi/3
    G[idx] = I[idx]*(1 - S[idx])
    B[idx] = I[idx]*(1 + (S[idx]*np.cos(H3))/(np.cos(np.pi/3 - H3) + 1e-12))
    R[idx] = 3*I[idx] - (G[idx] + B[idx])
    
    rgb = np.clip(np.stack([R, G, B], axis=-1), 0, 1)
    return (rgb * 255).astype(np.uint8)

def apply_hsi_enhancement(img, comp_idx, func):
    """
    Converts RGB to HSI, enhances ONE component, converts back.
    comp_idx: 0=H, 1=S, 2=I
    """
    hsi = rgb_to_hsi(img)
    hsi_mod = hsi.copy()
    comp = (hsi[:,:,comp_idx] * 255).astype(np.uint8)
    compE = func(comp)
    
    # Normalize back to 0-1 float 
    hsi_mod[:,:,comp_idx] = compE.astype(np.float32)/255.0
    
    return hsi_to_rgb(hsi_mod)

def apply_rgb_enhancement(img, func):
    """
    Enhances R, G, and B components individually and merges them.
    """
    r, g, b = cv2.split(img)
    rE = func(r)
    gE = func(g)
    bE = func(b)
    return cv2.merge((rE, gE, bE))

# main block
def process_images_to_collages(image_paths, image_names):
    """
    Generates 4 comparison collages:
    1. RGB All Components
    2. HSI Hue
    3. HSI Saturation
    4. HSI Intensity
    """
    os.makedirs(DEFAULT_COMPARE_DIR, exist_ok=True)

    original_images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None: raise FileNotFoundError(f"Could not read {path}")
        original_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    collages_paths = []
    enhancement_methods = {
        'Linear': linear_enhancement,
        'Logarithmic': log_enhancement,
        'Exponential': exp_enhancement
    }
    
    method_names = list(enhancement_methods.keys())
    rows = len(method_names) + 1 
    cols = 3 
    tasks = [
        {"name": "RGB_Full_Enhancement", "title": "RGB Enhancement (R, G, B Modified)", "type": "rgb"},
        {"name": "HSI_Hue_Enhancement", "title": "HSI: Hue Modified (S, I Fixed)", "type": "hsi", "idx": 0},
        {"name": "HSI_Sat_Enhancement", "title": "HSI: Saturation Modified (H, I Fixed)", "type": "hsi", "idx": 1},
        {"name": "HSI_Int_Enhancement", "title": "HSI: Intensity Modified (H, S Fixed)", "type": "hsi", "idx": 2},
    ]

    for task in tasks:
        fig, axes = plt.subplots(rows, cols, figsize=(15, 20))
        fig.suptitle(task["title"], fontsize=20, y=0.95)

        # Original
        for i in range(cols):
            axes[0, i].imshow(original_images[i])
            axes[0, i].set_title(f"{image_names[i]}\n(Original)", fontsize=12)
            axes[0, i].axis('off')

        # Rows 1-3
        for row_idx, method_name in enumerate(method_names):
            func = enhancement_methods[method_name]
            
            for img_idx in range(cols):
                img = original_images[img_idx]
                if task["type"] == "rgb":
                    res_img = apply_rgb_enhancement(img, func)
                elif task["type"] == "hsi":
                    res_img = apply_hsi_enhancement(img, task["idx"], func)
                
                # Plot
                ax = axes[row_idx + 1, img_idx]
                ax.imshow(res_img)
                ax.set_title(f"{method_name} Enhancement", fontsize=10)
                ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        save_path = os.path.join(DEFAULT_COMPARE_DIR, f"{task['name']}.png")
        plt.savefig(save_path)
        plt.close(fig)
        collages_paths.append(save_path)

    return collages_paths