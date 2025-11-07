
import io, sys, subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import streamlit as st

# Try to import drawable-canvas (needs GitHub dependency)
try:
    from streamlit_drawable_canvas import st_canvas
except Exception as e:
    st.error("This live-drawing version requires 'streamlit-drawable-canvas' (GitHub install). "
             "Use the click-tools app if your platform blocks Git installs.")
    st.stop()

st.set_page_config(page_title="Bone Ninja — Live Canvas", layout="wide")

# Colors
CYAN = "#00FFFF"; BLUE="#4285F4"; MAGENTA="#DD00DD"; RED="#FF0000"; GREEN="#00C800"; ORANGE="#FFA500"

@dataclass
class Line2D:
    p1: Tuple[float,float]
    p2: Tuple[float,float]

def pil_from_bytes(file_bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGBA")

def polygon_mask(size: Tuple[int,int], points: List[Tuple[float,float]]) -> Image.Image:
    mask = Image.new("L", size, 0)
    if len(points) >= 3:
        ImageDraw.Draw(mask).polygon(points, fill=255, outline=255)
    return mask

def apply_affine(img: Image.Image, dx: float, dy: float, rot_deg: float, center: Tuple[float,float]) -> Image.Image:
    rotated = img.rotate(rot_deg, resample=Image.BICUBIC, center=center, expand=False)
    canvas = Image.new("RGBA", img.size, (0,0,0,0))
    canvas.alpha_composite(rotated, (int(round(dx)), int(round(dy))))
    return canvas

def paste_with_mask(base: Image.Image, overlay: Image.Image, mask: Image.Image) -> Image.Image:
    out = base.copy()
    out.paste(overlay, (0,0), mask)
    return out

st.sidebar.title("Live drawing (requires drawable-canvas)")
uploaded = st.sidebar.file_uploader("Upload image", type=["png","jpg","jpeg","tif","tiff"])
if uploaded is None:
    st.info("Upload an image to begin."); st.stop()
img = pil_from_bytes(uploaded.getvalue())
W,H = img.size
disp_w = st.sidebar.slider("Canvas width", 700, 1600, min(1100, W))
scale = disp_w / W
disp_h = int(H*scale)

stroke_w = st.sidebar.slider("Stroke width", 1, 4, 2)
dx = st.sidebar.slider("ΔX (px)", -1000, 1000, 0, 1)
dy = st.sidebar.slider("ΔY (px)", -1000, 1000, 0, 1)
theta = st.sidebar.slider("Rotate (deg)", -180, 180, 0, 1)

mode = st.sidebar.selectbox("Tool", ["draw","transform"], index=0)

bg = img.resize((disp_w, disp_h), Image.BILINEAR)
c = st_canvas(background_image=bg, stroke_color=CYAN, stroke_width=stroke_w,
              update_streamlit=True, height=disp_h, width=disp_w,
              drawing_mode="freedraw" if mode=="draw" else "transform",
              display_toolbar=True, key="live")

st.warning("This demo only shows the live-drawing canvas; full tool parity can be added once deploy allows GitHub dependency.")
