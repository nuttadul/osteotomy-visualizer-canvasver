
import os
import io
import types
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

sim = st.session_state.sim

# --- Sidebar: File loader and controls ---
with st.sidebar:
    st.markdown("### Load image")
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        # Save to a temp file and pass path to the original `load_img`
        tmp_path = os.path.join(st.experimental_user, "tmp_upload_image.png") if hasattr(st, "experimental_user") else "tmp_upload_image.png"
        img = Image.open(uploaded).convert("RGB")
        img.save(tmp_path)
        sim.load_img(tmp_path)

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
    st.markdown("#### Pointer input")
    canvas_size = 512
    bg = None
    if hasattr(sim, "img") and sim.img is not None:
        # Build a background image for the canvas from the loaded image
        try:
            import matplotlib.pyplot as plt
            # sim.img is likely a numpy array (h,w,3) or (h,w); normalize to RGB
            arr = sim.img
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            arr = arr.astype("uint8")
            bg = Image.fromarray(arr)
            if bg.width != canvas_size or bg.height != canvas_size:
                bg = bg.resize((canvas_size, canvas_size))
        except Exception:
            bg = None

    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=2,
        stroke_color="#00FFFF",
        background_image=bg,
        update_streamlit=True,
        height=canvas_size,
        width=canvas_size,
        drawing_mode="transform",
        key="canvas",
    )

    # Map last pointer position to image coordinates and send to original callbacks
    # We emulate Matplotlib events with a tiny object exposing xdata/ydata
    class _Evt:
        def __init__(self, x, y):
            self.xdata = x
            self.ydata = y
            # For compatibility with potential attributes in the original handlers
            self.inaxes = True

    # Try to infer a single click from the canvas JSON (drag/transform center)
    if canvas_result.json_data is not None and len(canvas_result.json_data.get("objects", [])) > 0:
        # Use the first object's center as a "move" pointer demo (best-effort)
        try:
            obj = canvas_result.json_data["objects"][-1]
            cx = obj.get("left", canvas_size/2) + obj.get("width", 0)/2
            cy = obj.get("top", canvas_size/2) + obj.get("height", 0)/2
            # Convert canvas coords to image coords
            if hasattr(sim, "img") and sim.img is not None:
                H, W = sim.img.shape[:2]
                x_img = cx / canvas_size * W
                y_img = cy / canvas_size * H
            else:
                x_img, y_img = cx, cy

            evt = _Evt(x_img, y_img)
            # Forward as both move and click to approximate interaction
            sim.on_move(evt)
            if st.button("Send as click at ({:.1f}, {:.1f})".format(x_img, y_img)):
                sim.on_click(evt)
        except Exception as e:
            st.info(f"Pointer mapping error: {e}")

# Small status line
if hasattr(sim, "status_txt"):
    st.caption(getattr(sim, "status_txt", ""))
