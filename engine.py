#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ilizarov 2D planning and simulation (padded canvas)
- Load X-ray/photo
- Scale (px/mm)
- Proximal & Distal lines (+ live preview while placing)
- CORA point
- Osteotomy polygon: crops a cut-piece RGBA layer (with fallback if skimage missing)
- Rotate distal segment & cut-piece around hinge using slider
- Light drawing UI for polygon; distal line in yellow
- Expanded canvas so rotated segment doesn’t clip out of view
"""

import os, sys, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.widgets import Slider, Button, PolygonSelector
from matplotlib import path as mpath  # fallback polygon mask

# Optional deps
try:
    import tkinter as tk
    from tkinter import filedialog, simpledialog
    TK_OK = True
except Exception:
    TK_OK = False

try:
    from skimage.draw import polygon2mask
    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False


# ---------- Utilities ----------
def macos_choose_file():
    """macOS Finder-native file picker (fallback if Tk not available)."""
    if sys.platform != 'darwin':
        return None
    try:
        import subprocess
        script = (
            'try\n'
            'set f to (choose file with prompt "Choose image" of type {"public.png","public.jpeg","public.tiff","public.bmp"})\n'
            'POSIX path of f\n'
            'on error\n'
            'return ""\n'
            'end try'
        )
        res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        if res.returncode == 0:
            p = res.stdout.strip()
            return p if p else None
    except Exception:
        pass
    return None


def rotate_point(P, H, theta):
    px, py = P
    hx, hy = H
    c, s = math.cos(theta), math.sin(theta)
    x = c * (px - hx) - s * (py - hy) + hx
    y = s * (px - hx) + c * (py - hy) + hy
    return (x, y)


def polygon_mask_numpy(verts, h, w):
    """
    Boolean mask for polygon verts (x,y in image coords) WITHOUT skimage.
    Uses matplotlib.path.Path.contains_points on pixel centers.
    """
    poly = mpath.Path(np.asarray(verts, dtype=np.float32))
    xs = np.arange(w) + 0.5
    ys = np.arange(h) + 0.5
    gx, gy = np.meshgrid(xs, ys)  # (h, w)
    pts = np.vstack([gx.ravel(), gy.ravel()]).T
    inside = poly.contains_points(pts)
    return inside.reshape(h, w)


# ---------- App ----------
class IlizarovSim2D:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal', adjustable='box')
        # Start wide; real limits set after image load
        self.ax.set_xlim(-200, 1400)
        self.ax.set_ylim(1000, -200)
        plt.subplots_adjust(bottom=0.15)

        # Images
        self.bg_float = None          # original background (H,W,3) float 0..1
        self.base_bg = None           # background with a "hole"
        self.dist_rgba = None         # cut piece (H,W,4 RGBA)
        self.bg_artist = None         # imshow for background
        self.dist_img_artist = None   # imshow for moving cut-piece

        # Planning data
        self.px_per_mm = 1.0
        self.prox_line = None         # ((x1,y1),(x2,y2))
        self.dist_line = None         # ((x1,y1),(x2,y2))
        self.cora = None              # (x,y)
        self.hinge = None             # (x,y)

        # Osteotomy polygon
        self.osteotomy_poly = None    # list[(x,y)]
        self.osteotomy_poly0 = None   # frozen copy for outline rotation
        self.theta = 0.0              # radians

        # Artists for lines/markers/outline
        self.prox_artist = None
        self.dist_artist = None
        self.hinge_artist = None
        self.cora_artist = None
        self.ost_outline_artist = None
        self.preview_artist = None    # live guide while placing prox/dist

        # UI state
        self.theta_slider = None
        self.poly_selector = None
        self.tmp_pts = []
        self.mode = None

        # Events
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_move = self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

        # Status
        self.status = self.ax.text(0.02, 0.02, '', transform=self.ax.transAxes,
                                   fontsize=9, va='bottom', ha='left',
                                   bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # Right-side buttons
        right = 0.86
        w, h, pad = 0.12, 0.045, 0.008
        y = 0.86
        def add_btn(label, cb):
            nonlocal y
            axb = self.fig.add_axes([right, y, w, h])
            b = Button(axb, label)
            b.on_clicked(cb)
            y -= (h + pad)
            return b

        self.btn_load  = add_btn('Load [i]', self.btn_load_cb)
        self.btn_scale = add_btn('Scale [s]', self.btn_scale_cb)
        self.btn_hinge = add_btn('Hinge [h]', self.btn_hinge_cb)
        self.btn_prox  = add_btn('Prox line [p]', self.btn_prox_cb)
        self.btn_dist  = add_btn('Dist line [d]', self.btn_dist_cb)
        self.btn_cora  = add_btn('CORA [c]', self.btn_cora_cb)
        self.btn_poly  = add_btn('Osteotomy [o]', self.btn_poly_cb)
        self.btn_theta = add_btn('Slider [t]', self.btn_theta_cb)
        self.btn_reset = add_btn('Reset [r]', self.btn_reset_cb)

        self.set_status("i=load, o=osteotomy, h=hinge, s=scale, p=prox, d=dist, c=cora, t=slider")
        self.ax.set_autoscale_on(False)

    # -------- View helper --------
    def _apply_view_padding(self, h, w, pad_frac=0.25):
        """
        Expand axes limits beyond image extent by a fraction of the max dimension.
        Keeps y inverted (image coords).
        """
        pad = float(max(h, w)) * float(pad_frac)
        self.ax.set_xlim(-pad, w + pad)
        self.ax.set_ylim(h + pad, -pad)

    # -------- Buttons --------
    def btn_load_cb(self, _=None): self.open_image_dialog()
    def btn_scale_cb(self, _=None): self.mode='scale1'; self.tmp_pts=[]; self.set_status('Scale: click 2 points')
    def btn_hinge_cb(self, _=None): self.mode='hinge'; self.set_status('Click hinge')
    def btn_prox_cb(self, _=None):  self.mode='prox1'; self.tmp_pts=[]; self.set_status('Prox line: click 2 points')
    def btn_dist_cb(self, _=None):  self.mode='dist1'; self.tmp_pts=[]; self.set_status('Dist line: click 2 points')
    def btn_cora_cb(self, _=None):  self.mode='cora'; self.set_status('Click CORA')
    def btn_poly_cb(self, _=None):  self.start_poly()
    def btn_theta_cb(self, _=None): self.toggle_slider()
    def btn_reset_cb(self, _=None):
        self.prox_line = self.dist_line = self.cora = None
        self.osteotomy_poly = self.osteotomy_poly0 = None
        self.base_bg = self.dist_rgba = None
        self.theta = 0.0
        # remove artists
        for name in ['prox_artist','dist_artist','hinge_artist','cora_artist','ost_outline_artist']:
            a = getattr(self, name, None)
            if a is not None:
                try: a.remove()
                except Exception: pass
                setattr(self, name, None)
        if self.dist_img_artist is not None:
            try: self.dist_img_artist.remove()
            except Exception: pass
            self.dist_img_artist = None
        if self.preview_artist is not None:
            try: self.preview_artist.remove()
            except Exception: pass
            self.preview_artist = None
        if self.theta_slider is not None:
            try: self.theta_slider.set_val(0)
            except Exception: pass
        self.redraw()
        # re-apply padding if image already loaded
        if self.bg_float is not None:
            h, w = self.bg_float.shape[:2]
            self._apply_view_padding(h, w, pad_frac=0.25)
        self.set_status('Cleared.')

    # -------- UI helpers --------
    def set_status(self, msg):
        self.status.set_text(msg)
        self.fig.canvas.draw_idle()
        print(msg)

    def open_image_dialog(self):
        path = macos_choose_file()
        if not path and TK_OK:
            root = tk.Tk(); root.withdraw()
            try:
                path = filedialog.askopenfilename(title='Choose image',
                    filetypes=[('Image', '*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp')])
            finally:
                root.destroy()
        if path:
            self.load_img(path)

    def load_img(self, path):
        img = plt.imread(path)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = np.clip(img.astype(np.float32), 0.0, 1.0)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        self.bg_float = img
        self.base_bg = self.dist_rgba = None
        self.osteotomy_poly = self.osteotomy_poly0 = None
        self.theta = 0.0

        # clear artists
        for name in ['prox_artist','dist_artist','hinge_artist','cora_artist','ost_outline_artist']:
            a = getattr(self, name, None)
            if a is not None:
                try: a.remove()
                except Exception: pass
                setattr(self, name, None)
        if self.dist_img_artist is not None:
            try: self.dist_img_artist.remove()
            except Exception: pass
            self.dist_img_artist = None
        if self.preview_artist is not None:
            try: self.preview_artist.remove()
            except Exception: pass
            self.preview_artist = None
        if self.theta_slider is not None:
            try: self.theta_slider.set_val(0)
            except Exception: pass

        h, w = img.shape[:2]
        # Extra canvas so the rotated cut piece stays visible
        self._apply_view_padding(h, w, pad_frac=0.25)
        if self.bg_artist is None:
            self.bg_artist = self.ax.imshow(img, origin='upper', extent=(0, w, h, 0), zorder=1)
        else:
            self.bg_artist.set_data(img)
            self.bg_artist.set_extent((0, w, h, 0))

        self.set_status(f'Loaded {os.path.basename(path)}')

    # -------- Events --------
    def on_key(self, e):
        if e.key == 'i': self.open_image_dialog()
        elif e.key == 'o': self.start_poly()
        elif e.key == 'h': self.mode='hinge'; self.set_status('Click hinge')
        elif e.key == 't': self.toggle_slider()
        elif e.key == 's': self.mode='scale1'; self.tmp_pts=[]; self.set_status('Scale: click 2 points')
        elif e.key == 'p': self.mode='prox1'; self.tmp_pts=[]; self.set_status('Prox: click 2 points')
        elif e.key == 'd': self.mode='dist1'; self.tmp_pts=[]; self.set_status('Dist: click 2 points')
        elif e.key == 'c': self.mode='cora'; self.set_status('Click CORA')
        elif e.key == 'r': self.btn_reset_cb()

    def on_click(self, e):
        if e.button == 3 or e.inaxes != self.ax:
            self.mode = None
            self.fig.canvas.draw_idle()
            return
        x, y = e.xdata, e.ydata
        mode = self.mode

        # Hinge
        if mode == 'hinge':
            self.hinge = (x, y)
            self.mode = None
            self.set_status(f'Hinge=({x:.1f},{y:.1f})')
            self.redraw(); return

        # Scale
        if mode == 'scale1':
            self.tmp_pts = [(x,y)]; self.mode='scale2'; self.set_status('Click point 2'); return
        elif mode == 'scale2':
            p1 = self.tmp_pts[0]; p2 = (x,y)
            px = float(np.hypot(p1[0]-p2[0], p1[1]-p2[1]))
            mm = None
            if TK_OK:
                root = tk.Tk(); root.withdraw()
                try:
                    mm = simpledialog.askfloat('Scale', 'Enter real distance (mm):', initialvalue=100.0)
                finally:
                    root.destroy()
            if mm and mm>0:
                self.px_per_mm = px/mm
                self.set_status(f'Scale set {self.px_per_mm:.4f} px/mm')
            self.mode=None; self.tmp_pts=[]; return

        # Prox line
        if mode == 'prox1':
            self.tmp_pts=[(x,y)]; self.mode='prox2'; return
        elif mode == 'prox2':
            p1 = self.tmp_pts[0]; self.prox_line = (p1, (x,y))
            self.mode=None; self.tmp_pts=[]
            if self.preview_artist is not None:
                try: self.preview_artist.remove()
                except Exception: pass
                self.preview_artist=None
            self.set_status('Prox line set'); self.redraw(); return

        # Dist line
        if mode == 'dist1':
            self.tmp_pts=[(x,y)]; self.mode='dist2'; return
        elif mode == 'dist2':
            p1 = self.tmp_pts[0]; self.dist_line = (p1, (x,y))
            self.mode=None; self.tmp_pts=[]
            if self.preview_artist is not None:
                try: self.preview_artist.remove()
                except Exception: pass
                self.preview_artist=None
            self.set_status('Dist line set'); self.redraw(); return

        # CORA
        if mode == 'cora':
            self.cora = (x,y)
            self.mode=None
            self.set_status(f'CORA=({x:.1f},{y:.1f})')
            self.redraw(); return

    def on_move(self, e):
        mode = self.mode
        if e.inaxes != self.ax or mode not in ('prox2','dist2'):
            if self.preview_artist is not None and mode not in ('prox2','dist2'):
                try: self.preview_artist.remove()
                except Exception: pass
                self.preview_artist = None
            return
        x, y = e.xdata, e.ydata
        p1 = self.tmp_pts[0]
        xs, ys = [p1[0], x], [p1[1], y]
        if self.preview_artist is None:
            (self.preview_artist,) = self.ax.plot(xs, ys, linestyle='-', linewidth=1.5, color='0.8', zorder=8)
        else:
            self.preview_artist.set_data(xs, ys)
        self.fig.canvas.draw_idle()

    # -------- Polygon / slider --------
    def _destroy_polyselector(self):
        """Disconnect and remove any artists from an existing PolygonSelector so the
        original outline does not remain visible at the original location."""
        ps = self.poly_selector
        if ps is None:
            return
        # Try public disconnect
        try:
            ps.disconnect_events()
        except Exception:
            pass
        # Aggressively remove any artists the selector may have drawn
        try:
            for attr in (
                '_selection_artist', '_line', '_polygon', '_poly', '_axline',
                '_handles', '_polygon_handles', '_marker_handles', '_markers', '_artists'
            ):
                obj = getattr(ps, attr, None)
                if obj is None:
                    continue
                if isinstance(obj, (list, tuple)):
                    for a in list(obj):
                        try:
                            a.remove()
                        except Exception:
                            pass
                else:
                    try:
                        obj.remove()
                    except Exception:
                        pass
        except Exception:
            pass
        self.poly_selector = None

    def start_poly(self):
        # Kill existing selector completely (and any leftover artists)
        self._destroy_polyselector()
        # Remove old outline
        if self.ost_outline_artist is not None:
            try: self.ost_outline_artist.remove()
            except Exception: pass
            self.ost_outline_artist = None
        self.set_status('Draw osteotomy polygon (double-click to close)')
        # Create selector (no unsupported kwargs)
        self.poly_selector = PolygonSelector(self.ax, self.on_poly_complete, useblit=False)

        # Primary styling API (newer Matplotlib)
        try:
            self.poly_selector.set_props(dict(color='#00FFFF', linewidth=1.5, zorder=10000, alpha=0.85))
            self.poly_selector.set_handle_props(dict(marker='o',
                                                     markersize=8,
                                                     mec='#00FFFF', mew=2.0,
                                                     mfc='#00FFFF',
                                                     zorder=10001, alpha=1.0))
        except Exception:
            pass

        # Older Matplotlib variants expose dicts to mutate
        try:
            if hasattr(self.poly_selector, 'lineprops') and isinstance(self.poly_selector.lineprops, dict):
                self.poly_selector.lineprops.update(dict(color='#00FFFF', linewidth=1.5, zorder=10000))
            if hasattr(self.poly_selector, 'markerprops') and isinstance(self.poly_selector.markerprops, dict):
                self.poly_selector.markerprops.update(dict(marker='o', markersize=8,
                                                           mec='#00FFFF', mew=2.0, mfc='#00FFFF', zorder=10001))
        except Exception:
            pass

        # Final hard fallback: touch internal artists to *force* the cyan while drawing
        try:
            ln = getattr(self.poly_selector, '_line', None)
            if ln is not None:
                ln.set_color('#00FFFF'); ln.set_linewidth(1.5); ln.set_alpha(0.85); ln.set_zorder(10000)
            for attr in ('_handles', '_polygon_handles', '_marker_handles', '_markers'):
                hs = getattr(self.poly_selector, attr, None)
                if hs is None:
                    continue
                if not isinstance(hs, (list, tuple)):
                    hs = [hs]
                for h in hs:
                    for setter, val in (('set_markerfacecolor', '#00FFFF'),
                                        ('set_markeredgecolor', '#00FFFF'),
                                        ('set_color', '#00FFFF')):
                        try:
                            getattr(h, setter)(val)
                        except Exception:
                            pass
                    try: h.set_markersize(8)
                    except Exception: pass
                    try: h.set_zorder(10001)
                    except Exception: pass
        except Exception:
            pass

        # Extra: forcibly ensure all styling is bright cyan above everything
        self._force_polyselector_cyan(self.poly_selector)

    def _force_polyselector_cyan(self, ps):
        """Brutally force the PolygonSelector drawing phase to bright cyan above everything."""
        CY = '#00FFFF'
        try:
            # Newer Matplotlib public APIs
            try:
                ps.set_props(dict(color=CY, linewidth=1.5, zorder=10000, alpha=0.85))
            except Exception:
                pass
            try:
                ps.set_handle_props(dict(marker='o', markersize=8, mec=CY, mew=2.0, mfc=CY, zorder=10001, alpha=1.0))
            except Exception:
                pass
            # Older API dicts
            try:
                if hasattr(ps, 'lineprops') and isinstance(ps.lineprops, dict):
                    ps.lineprops.update(dict(color=CY, linewidth=1.5, zorder=10000))
                if hasattr(ps, 'markerprops') and isinstance(ps.markerprops, dict):
                    ps.markerprops.update(dict(marker='o', markersize=8, mec=CY, mew=2.0, mfc=CY, zorder=10001))
            except Exception:
                pass
            # Hard fallbacks: touch internal artists if present
            for attr in ('_selection_artist', '_line', '_polygon', '_poly', '_axline'):
                ln = getattr(ps, attr, None)
                if ln is None:
                    continue
                for setter, val in (('set_color', CY), ('set_alpha', 0.85), ('set_linewidth', 1.5), ('set_zorder', 10000)):
                    try:
                        getattr(ln, setter)(val)
                    except Exception:
                        pass
            for attr in ('_handles', '_polygon_handles', '_marker_handles', '_markers', '_artists'):
                hs = getattr(ps, attr, None)
                if hs is None:
                    continue
                if not isinstance(hs, (list, tuple)):
                    hs = [hs]
                for h in hs:
                    for setter, val in (('set_markerfacecolor', CY), ('set_markeredgecolor', CY), ('set_color', CY), ('set_zorder', 10001)):
                        try:
                            getattr(h, setter)(val)
                        except Exception:
                            pass
                    try:
                        h.set_markersize(8)
                    except Exception:
                        pass
        except Exception:
            pass

    def on_poly_complete(self, verts):
        if not verts or len(verts) < 3 or self.bg_float is None:
            self.redraw(); return

        self.osteotomy_poly = [(float(x), float(y)) for x, y in verts]
        self.osteotomy_poly0 = [tuple(p) for p in self.osteotomy_poly]

        h, w = self.bg_float.shape[:2]
        # Ensure canvas is spacious for rotation
        self._apply_view_padding(h, w, pad_frac=0.25)

        # Build mask (skimage preferred, fallback otherwise)
        if SKIMAGE_OK:
            verts_xy = np.array(self.osteotomy_poly, dtype=np.float32)
            verts_yx = np.stack([verts_xy[:,1], verts_xy[:,0]], axis=1)
            mask = polygon2mask((h, w), verts_yx)
        else:
            mask = polygon_mask_numpy(self.osteotomy_poly, h, w)

        mask_f = mask.astype(np.float32)

        # Background with a hole where the cut piece was
        self.base_bg = self.bg_float.copy()
        self.base_bg[mask] = 0.0

        # The cut piece as RGBA (RGB where mask, transparent elsewhere)
        rgb = self.bg_float.copy()
        rgb[~mask] = 0.0
        alpha = mask_f[..., None]
        self.dist_rgba = np.concatenate([rgb, alpha], axis=-1)

        # Create/update separate RGBA layer for the cut piece
        if self.dist_img_artist is None:
            self.dist_img_artist = self.ax.imshow(self.dist_rgba, origin='upper',
                                                  extent=(0, w, h, 0), zorder=9)
        else:
            self.dist_img_artist.set_data(self.dist_rgba)
            self.dist_img_artist.set_extent((0, w, h, 0))

        # Remove the original drawing outline so only the rotated cyan outline remains
        self._destroy_polyselector()

        self.redraw()

    def toggle_slider(self):
        if self.theta_slider is None:
            ax_slider = plt.axes([0.2, 0.03, 0.6, 0.03])
            self.theta_slider = Slider(ax_slider, 'θ (deg)', -60, 60, valinit=0)
            self.theta_slider.on_changed(self.on_slide)
        else:
            self.theta_slider.ax.remove(); self.theta_slider = None
        self.fig.canvas.draw_idle()
        # keep fixed limits; prevent autoscale on slider interaction
        self.ax.set_autoscale_on(False)

    def on_slide(self, val):
        self.theta = math.radians(val)
        self.redraw()

    # -------- Drawing helpers --------
    def _update_or_create_line(self, name, xs, ys, **kwargs):
        artist_attr = f"{name}_artist"
        artist = getattr(self, artist_attr, None)
        if artist is None or not plt.fignum_exists(self.fig.number):
            line, = self.ax.plot(xs, ys, **kwargs)
            setattr(self, artist_attr, line)
        else:
            artist.set_data(xs, ys)
            # try to update style if provided
            for k, v in kwargs.items():
                try:
                    getattr(artist, f"set_{k}")(v)
                except Exception:
                    pass
        return getattr(self, artist_attr)

    # -------- Redraw --------
    def redraw(self):
        # Background
        if self.base_bg is not None:
            h, w = self.base_bg.shape[:2]
            if self.bg_artist is None:
                self.bg_artist = self.ax.imshow(self.base_bg, origin='upper', extent=(0, w, h, 0), zorder=1)
            else:
                self.bg_artist.set_data(self.base_bg)
                self.bg_artist.set_extent((0, w, h, 0))
        elif self.bg_float is not None:
            h, w = self.bg_float.shape[:2]
            if self.bg_artist is None:
                self.bg_artist = self.ax.imshow(self.bg_float, origin='upper', extent=(0, w, h, 0), zorder=1)
            else:
                self.bg_artist.set_data(self.bg_float)
                self.bg_artist.set_extent((0, w, h, 0))

        # Moving cut-piece layer
        if self.dist_img_artist is not None and self.dist_rgba is not None:
            if self.hinge is not None:
                hx, hy = self.hinge
                t = transforms.Affine2D().rotate_around(hx, hy, self.theta) + self.ax.transData
                self.dist_img_artist.set_transform(t)
            else:
                self.dist_img_artist.set_transform(self.ax.transData)
            self.dist_img_artist.set_zorder(9)

        # Proximal line (fixed)
        if self.prox_line is not None:
            (x1, y1), (x2, y2) = self.prox_line
            self._update_or_create_line('prox', [x1, x2], [y1, y2],
                                        linestyle='-', linewidth=2, color='0.8', alpha=0.9, zorder=10)

        # Distal line (rotates)
        if self.dist_line is not None:
            (x1, y1), (x2, y2) = self.dist_line
            if self.hinge is not None:
                p1 = rotate_point((x1, y1), self.hinge, self.theta)
                p2 = rotate_point((x2, y2), self.hinge, self.theta)
            else:
                p1, p2 = (x1, y1), (x2, y2)
            self._update_or_create_line('dist', [p1[0], p2[0]], [p1[1], p2[1]],
                                        linestyle='--', linewidth=2, color='y', alpha=0.9, zorder=11)

        # Hinge / CORA markers
        if self.hinge is not None:
            self._update_or_create_line('hinge', [self.hinge[0]], [self.hinge[1]],
                                        marker='x', linestyle='None', markersize=8, color='r', zorder=12)
        if self.cora is not None:
            self._update_or_create_line('cora', [self.cora[0]], [self.cora[1]],
                                        marker='o', linestyle='None', markersize=6, color='b', zorder=12)

        # Cyan osteotomy outline (rotates with distal)
        if self.osteotomy_poly0 is not None:
            verts = np.array(self.osteotomy_poly0, dtype=float)
            if self.hinge is not None and abs(self.theta) > 1e-12:
                hx, hy = self.hinge
                c, s = math.cos(self.theta), math.sin(self.theta)
                x = verts[:, 0] - hx
                y = verts[:, 1] - hy
                xr = c * x - s * y + hx
                yr = s * x + c * y + hy
                verts = np.stack([xr, yr], axis=1)
            xs, ys = verts[:, 0], verts[:, 1]
            xs = np.r_[xs, xs[0]]; ys = np.r_[ys, ys[0]]
            self._update_or_create_line('ost_outline', xs, ys,
                                        linestyle='-', linewidth=1.5, color='c', alpha=0.85, zorder=13)

        self.fig.canvas.draw_idle()


# ---------- Main ----------
if __name__ == '__main__':
    IlizarovSim2D()
    plt.show()