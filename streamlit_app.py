
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
        # Create a safe temp directory and write the upload
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

# --- Main area: canvas and figure display ---
left, right = st.columns([1,1], gap="large")

# Draw to matplotlib figure using the existing redraw
with left:
    st.markdown("#### View")
    sim.redraw()
    st.pyplot(sim.fig, use_container_width=True)

# Canvas for pointer input (clicks)
with right:
    st.markdown("#### Click on image")
    canvas_size = 768  # bigger target for precision
    bg = None
    H = W = None
    if hasattr(sim, "img") and sim.img is not None:
        try:
            arr = sim.img
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            arr = arr.astype("uint8")
            H, W = arr.shape[:2]
            bg = Image.fromarray(arr)
            # Fit within square while preserving aspect ratio
            if W >= H:
                new_w = canvas_size
                new_h = int(H * canvas_size / W)
            else:
                new_h = canvas_size
                new_w = int(W * canvas_size / H)
            bg = bg.resize((new_w, new_h))
            pad_w = canvas_size - new_w
            pad_h = canvas_size - new_h
            # Create a padded background so coordinates stay simple
            padded = Image.new("RGB", (canvas_size, canvas_size), (0,0,0))
            padded.paste(bg, (pad_w//2, pad_h//2))
            bg = padded
            disp_w, disp_h = new_w, new_h
            off_x, off_y = pad_w//2, pad_h//2
        except Exception:
            bg = None
            disp_w = disp_h = off_x = off_y = 0
    else:
        disp_w = disp_h = off_x = off_y = 0

    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=2,
        stroke_color="#00FFFF",
        background_image=bg,
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

    # Auto-dispatch click when a new point is added
    obj_count = 0
    if canvas_result.json_data is not None:
        obj_count = len(canvas_result.json_data.get("objects", []))

    if obj_count > st.session_state.last_obj_count and obj_count > 0:
        try:
            obj = canvas_result.json_data["objects"][-1]
            # Canvas coordinates
            cx = obj.get("left", canvas_size/2) + obj.get("radius", 0)
            cy = obj.get("top", canvas_size/2) + obj.get("radius", 0)

            # Convert from displayed padded canvas back to original image coords
            if H and W and disp_w and disp_h is not None:
                # Remove padding offset
                cx_unpad = max(0, min(canvas_size, cx - off_x))
                cy_unpad = max(0, min(canvas_size, cy - off_y))
                # Scale to image space
                x_img = cx_unpad / (disp_w if disp_w else 1) * W
                y_img = cy_unpad / (disp_h if disp_h else 1) * H
            else:
                x_img, y_img = cx, cy

            evt = _Evt(x_img, y_img)
            sim.on_move(evt)
            sim.on_click(evt)
        except Exception as e:
            st.info(f"Click mapping error: {e}")

        st.session_state.last_obj_count = obj_count

if hasattr(sim, "status_txt"):
    st.caption(getattr(sim, "status_txt", ""))
