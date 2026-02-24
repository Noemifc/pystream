"""
Live Difference + SSIM + Live SSIM Plot + TXT report

- Select a folder containing reconstructed slice images
- Compares consecutive images: i vs i-1
- Live diff viewer + SSIM value + SSIM plot (SSIM vs slice index)
- Logs pairs where SSIM <= threshold
- Saves a TXT report at the end (or when you click Save)

Supported formats (via Pillow): png, tif/tiff, jpg, bmp
"""

import os
import re
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

try:
    from PIL import Image
except ImportError as e:
    raise ImportError("Missing dependency: Pillow. Install with: pip install pillow") from e


# -----------------------------
# Config
# -----------------------------
_MIN_FRAME_SIZE = 32          # nxn pixel size
_DEFAULT_THRESHOLD = 0.80     # SSIM warning threshold
_DEFAULT_INTERVAL_MS = 60     # playback speed: 60 ms ~ 16.6 fps
_MAX_PLOT_POINTS = 6000


# -----------------------------
# Utilities
# -----------------------------
def natural_key(s: str):
    """Natural sort key: 'img_2.png' < 'img_10.png'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_image_gray(path: str) -> Optional[np.ndarray]:
    """Load image as 2D float32 (grayscale). Returns None if load fails."""
    try:
        im = Image.open(path)
        # convert to grayscale (L) and keep as float32
        im = im.convert("F")  # 32-bit float pixels
        arr = np.array(im, dtype=np.float32)
        if arr.ndim != 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            return None
        if arr.shape[0] < _MIN_FRAME_SIZE or arr.shape[1] < _MIN_FRAME_SIZE:
            return None
        return arr
    except Exception:
        return None


def similarity_ssim(img_a: np.ndarray, img_b: np.ndarray, eps: float = 1e-6) -> float:
    """
    Simple global SSIM-like measure (no windows, no SciPy).
    Returns value in [0,1].
    """
    if img_a is None or img_b is None:
        return 1.0
    if img_a.shape != img_b.shape:
        return 1.0

    A = img_a.astype(np.float32, copy=False)
    B = img_b.astype(np.float32, copy=False)

    muA = float(A.mean())
    muB = float(B.mean())

    sigmaA = float(A.var())
    sigmaB = float(B.var())

    sigmaAB = float(np.mean((A - muA) * (B - muB)))

    # constants (dimensionless here, since we didn't normalize intensity)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    numerator = (2 * muA * muB + C1) * (2 * sigmaAB + C2)
    denominator = (muA * muA + muB * muB + C1) * (sigmaA + sigmaB + C2)

    ssim = numerator / (denominator + eps)
    return float(max(0.0, min(1.0, ssim)))


@dataclass
class FlagEvent:
    idx_prev: int
    idx_curr: int
    prev_name: str
    curr_name: str
    ssim: float


# -----------------------------
# UI
# -----------------------------
class SliceCompareWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Slice Comparator — Live Diff + SSIM")
        self.setGeometry(80, 80, 920, 980)

        # state
        self.folder: Optional[str] = None
        self.files: List[str] = []
        self.index: int = 0  # current index (compares index vs index-1)
        self.prev_img: Optional[np.ndarray] = None
        self.prev_name: str = ""
        self.flagged: List[FlagEvent] = []

        self.running = False
        self.paused = False

        self.threshold = _DEFAULT_THRESHOLD
        self.interval_ms = _DEFAULT_INTERVAL_MS

        # plot buffers
        self._x = []    # slice index
        self._y = []    # ssim
        self._max_points = _MAX_PLOT_POINTS

        # timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._step_once)

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- Top controls ---
        top = QtWidgets.QHBoxLayout()

        self.btn_folder = QtWidgets.QPushButton("Choose folder…")
        self.btn_folder.clicked.connect(self._choose_folder)
        top.addWidget(self.btn_folder)

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_start.clicked.connect(self._on_start)
        top.addWidget(self.btn_start)

        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_pause.setEnabled(False)
        top.addWidget(self.btn_pause)

        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.clicked.connect(self._on_reset)
        top.addWidget(self.btn_reset)

        self.btn_save = QtWidgets.QPushButton("Save TXT")
        self.btn_save.clicked.connect(self._save_txt_dialog)
        top.addWidget(self.btn_save)

        top.addStretch()

        # threshold
        top.addWidget(QtWidgets.QLabel("SSIM threshold:"))
        self.spin_th = QtWidgets.QDoubleSpinBox()
        self.spin_th.setRange(0.0, 1.0)
        self.spin_th.setSingleStep(0.01)
        self.spin_th.setDecimals(3)
        self.spin_th.setValue(self.threshold)
        self.spin_th.valueChanged.connect(self._on_threshold_changed)
        top.addWidget(self.spin_th)

        # speed
        top.addWidget(QtWidgets.QLabel("Interval (ms):"))
        self.spin_ms = QtWidgets.QSpinBox()
        self.spin_ms.setRange(1, 2000)
        self.spin_ms.setValue(self.interval_ms)
        self.spin_ms.valueChanged.connect(self._on_interval_changed)
        top.addWidget(self.spin_ms)

        layout.addLayout(top)

        # status label
        self.lbl = QtWidgets.QLabel("Pick a folder to begin.")
        layout.addWidget(self.lbl)

        # --- Diff image view ---
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        layout.addWidget(self.image_view, stretch=3)

        # --- Plot ---
        self.plot = pg.PlotWidget()
        self.plot.setTitle("SSIM vs slice index")
        self.plot.setLabel("bottom", "Slice index (i)")
        self.plot.setLabel("left", "SSIM (0..1)")
        self.plot.setYRange(0.0, 1.0, padding=0.0)
        self.plot.showGrid(x=True, y=True, alpha=0.25)

        self.curve = self.plot.plot([], [], pen=pg.mkPen(width=2))

        self.th_line = pg.InfiniteLine(
            pos=self.threshold,
            angle=0,
            pen=pg.mkPen(color='r', width=2, style=QtCore.Qt.DashLine)
        )
        self.plot.addItem(self.th_line)

        layout.addWidget(self.plot, stretch=2)

        # --- Bottom info ---
        bottom = QtWidgets.QHBoxLayout()
        self.lbl_count = QtWidgets.QLabel("Flagged: 0")
        bottom.addWidget(self.lbl_count)
        bottom.addStretch()
        layout.addLayout(bottom)

    # -----------------------------
    # Controls
    # -----------------------------
    def _choose_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder with slices")
        if not folder:
            return
        self.folder = folder
        self._load_file_list()
        self._on_reset()
        self._update_status(f"Loaded {len(self.files)} images from: {self.folder}")

    def _load_file_list(self):
        assert self.folder is not None
        entries = []
        for fn in os.listdir(self.folder):
            path = os.path.join(self.folder, fn)
            if not os.path.isfile(path):
                continue
            # Let Pillow decide, but filter out obvious non-images a bit:
            if fn.lower().endswith((".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")):
                entries.append(fn)
        entries.sort(key=natural_key)
        self.files = entries

    def _on_threshold_changed(self, v: float):
        self.threshold = float(v)
        self.th_line.setValue(self.threshold)

    def _on_interval_changed(self, ms: int):
        self.interval_ms = int(ms)
        if self.timer.isActive():
            self.timer.setInterval(self.interval_ms)

    def _on_start(self):
        if not self.folder or len(self.files) < 2:
            self._update_status("Need a folder with at least 2 images.")
            return
        self.running = True
        self.paused = False
        self.btn_pause.setEnabled(True)
        self.btn_start.setEnabled(False)
        self.btn_start.setText("Running")
        self.btn_pause.setText("Pause")
        self.timer.start(self.interval_ms)

    def _on_pause(self):
        if not self.running:
            return
        self.paused = not self.paused
        if self.paused:
            self.timer.stop()
            self.btn_pause.setText("Paused")
            self.btn_start.setEnabled(True)
            self.btn_start.setText("Resume")
        else:
            self.timer.start(self.interval_ms)
            self.btn_pause.setText("Pause")
            self.btn_start.setEnabled(False)
            self.btn_start.setText("Running")

    def _on_reset(self):
        self.timer.stop()
        self.running = False
        self.paused = False

        self.index = 0
        self.prev_img = None
        self.prev_name = ""
        self.flagged.clear()

        self._x.clear()
        self._y.clear()
        self.curve.setData([], [])
        self.image_view.clear()

        self.btn_start.setEnabled(True)
        self.btn_start.setText("Start")
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")

        self._update_flagged_label()

    # -----------------------------
    # Processing loop
    # -----------------------------
    def _step_once(self):
        if not self.folder or len(self.files) < 2:
            self.timer.stop()
            return

        # first load prev
        if self.prev_img is None:
            self.index = 0
            p0 = os.path.join(self.folder, self.files[0])
            img0 = load_image_gray(p0)
            if img0 is None:
                self._update_status(f"Failed to load first image: {self.files[0]}")
                self.timer.stop()
                self._finish_and_autosave()
                return
            self.prev_img = img0
            self.prev_name = self.files[0]
            self.index = 1
            self._update_status(f"First slice captured: {self.prev_name}. Waiting next…")
            return

        # stop at end
        if self.index >= len(self.files):
            self.timer.stop()
            self._finish_and_autosave()
            return

        curr_name = self.files[self.index]
        curr_path = os.path.join(self.folder, curr_name)
        curr_img = load_image_gray(curr_path)
        if curr_img is None:
            self._update_status(f"Skipped (load failed): {curr_name}")
            self.index += 1
            return

        # if shape mismatch -> skip but advance
        if curr_img.shape != self.prev_img.shape:
            self._update_status(
                f"Skipped (shape mismatch): {self.prev_name} {self.prev_img.shape} vs {curr_name} {curr_img.shape}"
            )
            self.prev_img = curr_img
            self.prev_name = curr_name
            self.index += 1
            return

        # compute diff + ssim
        diff = np.abs(curr_img - self.prev_img)
        ssim_val = similarity_ssim(curr_img, self.prev_img)

        # plot update (x = current index, y = ssim)
        self._x.append(self.index)
        self._y.append(ssim_val)
        if len(self._x) > self._max_points:
            self._x = self._x[-self._max_points:]
            self._y = self._y[-self._max_points:]
        self.curve.setData(self._x, self._y)

        # warning / log
        warn = (ssim_val <= self.threshold)
        if warn:
            QtWidgets.QApplication.beep()
            self.lbl.setStyleSheet("color: red; font-weight: bold;")
            self.flagged.append(
                FlagEvent(
                    idx_prev=self.index - 1,
                    idx_curr=self.index,
                    prev_name=self.prev_name,
                    curr_name=curr_name,
                    ssim=ssim_val,
                )
            )
            self._update_flagged_label()
        else:
            self.lbl.setStyleSheet("")

        # update image view + label
        stats = f"[{self.index-1}→{self.index}]  {self.prev_name}  vs  {curr_name}   |  SSIM={ssim_val:.4f}"
        if warn:
            stats += f"   ⚠ SSIM ≤ {self.threshold:.4f}"
        self.image_view.setImage(diff, autoLevels=True)
        self._update_status(stats)

        # advance
        self.prev_img = curr_img
        self.prev_name = curr_name
        self.index += 1

    # -----------------------------
    # TXT save
    # -----------------------------
    def _default_report_path(self) -> Optional[str]:
        if not self.folder:
            return None
        return os.path.join(self.folder, "ssim_threshold_exceeded_pairs.txt")

    def _finish_and_autosave(self):
        # When done, auto-save report in the same folder (if possible)
        if self.folder:
            out = self._default_report_path()
            self._save_txt(out)
            self._update_status(f"Done. Auto-saved report: {os.path.basename(out)}")
        else:
            self._update_status("Done.")

        # reset buttons to allow re-run
        self.running = False
        self.paused = False
        self.btn_start.setEnabled(True)
        self.btn_start.setText("Start")
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")

    def _save_txt_dialog(self):
        default = self._default_report_path() or "ssim_threshold_exceeded_pairs.txt"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save TXT report", default, "Text files (*.txt);;All files (*)"
        )
        if not path:
            return
        self._save_txt(path)
        self._update_status(f"Saved report: {path}")

    def _save_txt(self, path: Optional[str]):
        if not path:
            return
        lines = []
        lines.append("SSIM threshold exceeded report (SSIM <= threshold)\n")
        lines.append(f"Folder: {self.folder}\n")
        lines.append(f"Threshold: {self.threshold:.6f}\n")
        lines.append(f"Total images: {len(self.files)}\n")
        lines.append(f"Flagged pairs: {len(self.flagged)}\n")
        lines.append("-" * 80 + "\n")
        if not self.flagged:
            lines.append("No pairs exceeded the threshold.\n")
        else:
            for ev in self.flagged:
                lines.append(
                    f"[{ev.idx_prev} -> {ev.idx_curr}]  "
                    f"{ev.prev_name}  |  {ev.curr_name}  |  SSIM={ev.ssim:.6f}\n"
                )

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        except Exception as e:
            self._update_status(f"Failed saving report: {e}")

    # -----------------------------
    # UI helpers
    # -----------------------------
    def _update_status(self, msg: str):
        self.lbl.setText(msg)

    def _update_flagged_label(self):
        self.lbl_count.setText(f"Flagged: {len(self.flagged)}")


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = SliceCompareWindow()
    win.show()
    win.raise_()
    print("Slice Comparator opened. Choose a folder, then Start.")
    app.exec_()


if __name__ == "__main__":
    main()
