
import os
import io
import types
import tempfile
from uuid import uuid4
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Import the original code WITHOUT modification
import engine  # engine.py placed alongside this file

st.set_page_config(page_title="Ilizarov 2D (Streamlit Port)", layout="wide")

# --- Session State: persistent simulator instance ---
if "sim" not in st.session_state:
    st.session_state.sim = engine.IlizarovSim2D()
if "last_obj_count" not in st.session_state:
    st.session_state.last_obj_count = 0

sim = st.session_state.sim

# --- Sidebar: File loader and controls ---
with st.sidebar:
    st.markdown("### Load image")
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        tmp_dir = Path(tempfile.gettempdir()) / "st_ilizarov_uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ext = (uploaded.name.split(".")[-1].lower() if "." in uploaded.name else "png")
        tmp_path = tmp_dir / f"{uuid4().hex}.{ext}"
        img = Image.open(uploaded).convert("RGB")
        img.save(tmp_path.as_posix())
        sim.load_img(tmp_path.as_posix())

    st.markdown("---")
    st.markdown("### Mode buttons")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Scale"):
            sim.btn_scale_cb(None)
        if st.button("Proximal line"):
            sim.btn_prox_cb(None)
        if st.button("CORA point"):
            sim.btn_cora_cb(None)
        if st.button("Theta mode"):
            sim.btn_theta_cb(None)
    with col2:
        if st.button("Hinge"):
            sim.btn_hinge_cb(None)
        if st.button("Distal line"):
            sim.btn_dist_cb(None)
        if st.button("Osteotomy poly"):
            sim.btn_poly_cb(None)
        if st.button("Reset"):
            sim.btn_reset_cb(None)

    st.markdown("---")
    st.markdown("### Angle slider")
    theta = st.slider("θ", min_value=-180.0, max_value=180.0, step=0.1, value=0.0, key="theta_slider")
    sim.on_slide(theta)

# --- Single interactive canvas that mirrors the current view ---
canvas_size = 900  # large interactive area

# Redraw using the engine and convert the Matplotlib figure to an image used as canvas background
sim.redraw()
canvas = FigureCanvas(sim.fig)
canvas.draw()
w, h = sim.fig.canvas.get_width_height()
buf = np.frombuffer(sim.fig.canvas.buffer_rgba(), dtype=np.uint8)
rgba = buf.reshape(h, w, 4)
bg_img = Image.fromarray(rgba).convert("RGB")

# Pad the figure to square for the canvas while preserving aspect ratio
if w >= h:
    new_w = canvas_size
    new_h = int(h * canvas_size / w)
else:
    new_h = canvas_size
    new_w = int(w * canvas_size / h)
bg_resized = bg_img.resize((new_w, new_h))
pad_w = canvas_size - new_w
pad_h = canvas_size - new_h
padded = Image.new("RGB", (canvas_size, canvas_size), (0,0,0))
padded.paste(bg_resized, (pad_w//2, pad_h//2))
bg_for_canvas = padded
disp_w, disp_h = new_w, new_h
off_x, off_y = pad_w//2, pad_h//2

st.markdown("#### Click directly on the image below")
canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=2,
    stroke_color="#00FFFF",
    background_image=bg_for_canvas,
    update_streamlit=True,
    height=canvas_size,
    width=canvas_size,
    drawing_mode="point",
    key="canvas",
    point_display_radius=2,
)

class _Evt:
    def __init__(self, x, y):
        self.xdata = x
        self.ydata = y
        self.inaxes = True

obj_count = 0
if canvas_result.json_data is not None:
    obj_count = len(canvas_result.json_data.get("objects", []))

# When a new point is clicked on the canvas, map it back to image coords and forward to engine handlers
if obj_count > st.session_state.last_obj_count and obj_count > 0:
    try:
        obj = canvas_result.json_data["objects"][-1]
        cx = obj.get("left", canvas_size/2) + obj.get("radius", 0)
        cy = obj.get("top", canvas_size/2) + obj.get("radius", 0)

        # Remove padding and scale back to the original figure size
        cx_unpad = max(0, min(canvas_size, cx - off_x))
        cy_unpad = max(0, min(canvas_size, cy - off_y))
        x_fig = cx_unpad / (disp_w if disp_w else 1) * w
        y_fig = cy_unpad / (disp_h if disp_h else 1) * h

        # Convert figure pixel to image pixel using sim.img dimensions (engine expects image space)
        if hasattr(sim, "img") and sim.img is not None:
            H, W = sim.img.shape[:2]
            x_img = x_fig / (w if w else 1) * W
            y_img = y_fig / (h if h else 1) * H
        else:
            x_img, y_img = x_fig, y_fig

        evt = _Evt(x_img, y_img)
        sim.on_move(evt)
        sim.on_click(evt)
    except Exception as e:
        st.info(f"Click mapping error: {e}")

    st.session_state.last_obj_count = obj_count

# Optional status text from engine
if hasattr(sim, "status_txt"):
    st.caption(getattr(sim, "status_txt", ""))
