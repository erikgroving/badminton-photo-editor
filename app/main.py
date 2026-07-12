"""
Badminton AI Photo Editor — desktop app.

Two-phase workflow:
  Phase 1  Cull + Player Coverage  →  user reviews bursts in browser
  Phase 2  Crop + Color Correction →  final results

Run:
    python app/main.py
"""
import sys
import threading
import webbrowser
from pathlib import Path

if not getattr(sys, 'frozen', False):
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QProgressBar, QPushButton, QScrollArea, QSlider,
    QVBoxLayout, QWidget,
)


# ─── Worker threads ─────────────────────────────────────────────────────────────

class CullingWorker(QThread):
    stage_progress = Signal(int, int, int, str)
    finished       = Signal(dict)
    error          = Signal(str)

    def __init__(self, input_dir: str, output_dir: str, selection_target: float,
                 player_coverage: bool = True):
        super().__init__()
        self.input_dir        = input_dir
        self.output_dir       = output_dir
        self.selection_target = selection_target
        self.player_coverage  = player_coverage

    def run(self):
        try:
            from inference.run import run_culling_stage
            summary = run_culling_stage(
                self.input_dir,
                self.output_dir,
                selection_target=self.selection_target,
                player_coverage=self.player_coverage,
                progress_cb=self._cb,
            )
            self.finished.emit(summary)
        except Exception as exc:
            self.error.emit(str(exc))

    def _cb(self, stage: int, done: int, total: int, msg: str):
        self.stage_progress.emit(stage, done, total, msg)


class ProcessingWorker(QThread):
    stage_progress = Signal(int, int, int, str)
    finished       = Signal(dict)
    error          = Signal(str)

    def __init__(self, output_dir: str, jpeg_quality: int = 95):
        super().__init__()
        self.output_dir   = output_dir
        self.jpeg_quality = jpeg_quality

    def run(self):
        try:
            from inference.run import run_processing_stage
            summary = run_processing_stage(
                self.output_dir,
                jpeg_quality=self.jpeg_quality,
                progress_cb=self._cb,
            )
            self.finished.emit(summary)
        except Exception as exc:
            self.error.emit(str(exc))

    def _cb(self, stage: int, done: int, total: int, msg: str):
        self.stage_progress.emit(stage, done, total, msg)


# ─── Review server launchers ────────────────────────────────────────────────────

_review_server_started = False
_burst_server_started  = False


def _start_review_server(output_dir: str, port: int = 8765):
    global _review_server_started
    if _review_server_started:
        webbrowser.open(f"http://127.0.0.1:{port}")
        return
    _review_server_started = True

    def _serve():
        import review.server as srv
        from http.server import HTTPServer
        srv._output_dir = Path(output_dir)
        server = HTTPServer(("127.0.0.1", port), srv.ReviewHandler)
        server.serve_forever()

    threading.Thread(target=_serve, daemon=True).start()
    threading.Timer(0.6, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()


def _start_burst_review(output_dir: str, port: int = 8767):
    global _burst_server_started
    if _burst_server_started:
        # Server already running — reload state for the new (or same) output_dir.
        def _reload():
            import review.burst_review as br
            from review.burst_review import ReviewState
            br._state = None
            try:
                br._state = ReviewState(Path(output_dir))
            except Exception as exc:
                print(f"[burst_review] state reload failed: {exc}")
        threading.Thread(target=_reload, daemon=True).start()
        return
    _burst_server_started = True

    def _serve():
        from review.burst_review import BurstReviewHandler, ReviewState
        import review.burst_review as br
        from http.server import HTTPServer

        # Bind first so the browser can connect immediately while state loads.
        server = HTTPServer(("127.0.0.1", port), BurstReviewHandler)

        def _load():
            try:
                br._state = ReviewState(Path(output_dir))
            except Exception as exc:
                print(f"[burst_review] state load failed: {exc}")

        threading.Thread(target=_load, daemon=True).start()
        server.serve_forever()

    threading.Thread(target=_serve, daemon=True).start()


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _make_sep():
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    return sep


def _make_bar_row(name: str, label_width: int = 170):
    row   = QHBoxLayout()
    lbl   = QLabel(name)
    lbl.setFixedWidth(label_width)
    lbl.setStyleSheet("color: #aaa; font-size: 12px;")
    bar   = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(False)
    bar.setFixedHeight(14)
    count = QLabel("")
    count.setFixedWidth(110)
    count.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    count.setStyleSheet("color: #777; font-size: 11px;")
    row.addWidget(lbl)
    row.addWidget(bar)
    row.addWidget(count)
    return row, lbl, bar, count


# ─── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Badminton AI Photo Editor")
        self.setWindowIcon(_app_icon())
        self.setMinimumSize(680, 780)
        self._cull_worker:    CullingWorker    | None = None
        self._proc_worker:    ProcessingWorker | None = None
        self._output_dir:     str = ""
        self._setup_ui()

    def _setup_ui(self):
        # Scroll area so nothing is clipped on small screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.setCentralWidget(scroll)

        root   = QWidget()
        scroll.setWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(10)
        layout.setContentsMargins(28, 20, 28, 20)

        # ── Title ──────────────────────────────────────────────────────────────
        title = QLabel("Badminton AI Photo Editor")
        title.setFont(QFont("-apple-system", 17, QFont.Weight.Bold))
        layout.addWidget(title)

        subtitle = QLabel("Select a folder of RAW photos to cull, crop, and colour-correct.")
        subtitle.setStyleSheet("color: #888; font-size: 13px;")
        layout.addWidget(subtitle)

        layout.addSpacing(4)

        # ── Input folder ───────────────────────────────────────────────────────
        in_row  = QHBoxLayout()
        in_lbl  = QLabel("RAW folder:")
        in_lbl.setFixedWidth(90)
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Source folder of CR3 files…")
        self.folder_edit.setReadOnly(True)
        in_btn  = QPushButton("Browse…")
        in_btn.setFixedWidth(80)
        in_btn.clicked.connect(self._browse_input)
        in_row.addWidget(in_lbl)
        in_row.addWidget(self.folder_edit)
        in_row.addWidget(in_btn)
        layout.addLayout(in_row)

        # ── Output folder ──────────────────────────────────────────────────────
        out_row = QHBoxLayout()
        out_lbl = QLabel("Output folder:")
        out_lbl.setFixedWidth(90)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Where to write results (separate from source)…")
        self.output_edit.setReadOnly(True)
        out_btn = QPushButton("Browse…")
        out_btn.setFixedWidth(80)
        out_btn.clicked.connect(self._browse_output)
        out_row.addWidget(out_lbl)
        out_row.addWidget(self.output_edit)
        out_row.addWidget(out_btn)
        layout.addLayout(out_row)

        # ── Keep top % slider ──────────────────────────────────────────────────
        sel_row = QHBoxLayout()
        sel_lbl = QLabel("Keep top:")
        sel_lbl.setFixedWidth(90)
        self.sel_slider = QSlider(Qt.Orientation.Horizontal)
        self.sel_slider.setRange(5, 100)
        self.sel_slider.setValue(30)
        self.sel_slider.setTickInterval(5)
        self.sel_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sel_value = QLabel("30%")
        self.sel_value.setFixedWidth(38)
        self.sel_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.sel_slider.valueChanged.connect(lambda v: self.sel_value.setText(f"{v}%"))
        sel_row.addWidget(sel_lbl)
        sel_row.addWidget(self.sel_slider)
        sel_row.addWidget(self.sel_value)
        layout.addLayout(sel_row)

        sel_hint = QLabel("Percentage of photos to pass.")
        sel_hint.setStyleSheet("color: #999; font-size: 11px;")
        sel_hint.setWordWrap(True)
        layout.addWidget(sel_hint)

        self.coverage_chk = QCheckBox("Player coverage guarantee (promote one photo per unrepresented player)")
        self.coverage_chk.setChecked(True)
        self.coverage_chk.setStyleSheet("font-size: 12px; color: #222;")
        layout.addWidget(self.coverage_chk)

        layout.addSpacing(2)

        # ── Phase 1: Culling ───────────────────────────────────────────────────
        layout.addWidget(_make_sep())

        self.cull_btn = QPushButton("Run Culling")
        self.cull_btn.setFixedHeight(36)
        self.cull_btn.setFont(QFont("-apple-system", 13, QFont.Weight.Medium))
        self.cull_btn.clicked.connect(self._run_culling)
        layout.addWidget(self.cull_btn)

        r1, self.lbl1, self.bar1, self.cnt1 = _make_bar_row("Stage 1  Culling")
        r2, self.lbl2, self.bar2, self.cnt2 = _make_bar_row("Stage 2  Player Coverage")
        layout.addLayout(r1)
        layout.addLayout(r2)

        self.cull_count_lbl = QLabel("")
        self.cull_count_lbl.setStyleSheet("color: #5cb85c; font-size: 12px; font-weight: 600;")
        layout.addWidget(self.cull_count_lbl)

        self.coverage_count_lbl = QLabel("")
        self.coverage_count_lbl.setStyleSheet("color: #2b7bd0; font-size: 12px; font-weight: 600;")
        layout.addWidget(self.coverage_count_lbl)

        # Review Bursts + View Culled Photos side by side
        review_row = QHBoxLayout()
        self.burst_btn = QPushButton("Review Bursts")
        self.burst_btn.setFixedHeight(34)
        self.burst_btn.setFont(QFont("-apple-system", 12, QFont.Weight.Bold))
        self.burst_btn.setStyleSheet("background-color: #4a90d9; color: white; border-radius: 6px;")
        self.burst_btn.setEnabled(False)
        self.burst_btn.clicked.connect(self._open_burst_review)

        self.culled_btn = QPushButton("View Culled Photos")
        self.culled_btn.setFixedHeight(34)
        self.culled_btn.setFont(QFont("-apple-system", 12, QFont.Weight.Medium))
        self.culled_btn.setStyleSheet("background-color: #555; color: #ddd; border-radius: 6px;")
        self.culled_btn.setEnabled(False)
        self.culled_btn.clicked.connect(self._open_culled_review)

        review_row.addWidget(self.burst_btn)
        review_row.addWidget(self.culled_btn)
        layout.addLayout(review_row)

        # ── Phase 2: Crop + Color ──────────────────────────────────────────────
        layout.addWidget(_make_sep())

        proc_hint = QLabel("Review your burst selection, then run Crop && Color to generate final JPEGs.")
        proc_hint.setStyleSheet("color: #888; font-size: 11px;")
        proc_hint.setWordWrap(True)
        layout.addWidget(proc_hint)

        self.proc_btn = QPushButton("Run Crop && Color")
        self.proc_btn.setFixedHeight(36)
        self.proc_btn.setFont(QFont("-apple-system", 13, QFont.Weight.Medium))
        self.proc_btn.setEnabled(False)
        self.proc_btn.clicked.connect(self._run_processing)
        layout.addWidget(self.proc_btn)

        r3, self.lbl3, self.bar3, self.cnt3 = _make_bar_row("Stage 3  Cropping")
        r4, self.lbl4, self.bar4, self.cnt4 = _make_bar_row("Stage 4  Color Correction")
        layout.addLayout(r3)
        layout.addLayout(r4)

        self.proc_stats_lbl = QLabel("")
        self.proc_stats_lbl.setStyleSheet("color: #5cb85c; font-size: 12px; font-weight: 600;")
        layout.addWidget(self.proc_stats_lbl)

        self.review_btn = QPushButton("Score Overview")
        self.review_btn.setFixedHeight(34)
        self.review_btn.setFont(QFont("-apple-system", 12, QFont.Weight.Medium))
        self.review_btn.setEnabled(False)
        self.review_btn.clicked.connect(self._open_review)
        layout.addWidget(self.review_btn)

        # ── Status label ───────────────────────────────────────────────────────
        layout.addSpacing(4)
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color: #888; font-size: 11px;")
        self.status_lbl.setWordWrap(True)
        layout.addWidget(self.status_lbl)

        layout.addStretch()

    # ── Folder pickers ─────────────────────────────────────────────────────────

    def _browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select RAW Photos Folder")
        if folder:
            self.folder_edit.setText(folder)
            if not self.output_edit.text().strip():
                import os
                name    = Path(folder).name
                desktop = Path(os.path.expanduser("~")) / "Desktop" / "Badminton AI Photo Editor" / name
                self.output_edit.setText(str(desktop))

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_edit.setText(folder)

    # ── Phase 1: Culling ───────────────────────────────────────────────────────

    def _run_culling(self):
        input_dir  = self.folder_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        if not input_dir:
            self.status_lbl.setText("Please select a RAW photos folder first.")
            return
        if not output_dir:
            self.status_lbl.setText("Please select an output folder first.")
            return

        self._output_dir = output_dir
        self.cull_btn.setEnabled(False)
        self.burst_btn.setEnabled(False)
        self.culled_btn.setEnabled(False)
        self.proc_btn.setEnabled(False)
        self.review_btn.setEnabled(False)
        self.cull_count_lbl.setText("")
        self.coverage_count_lbl.setText("")
        self.proc_stats_lbl.setText("")
        self.status_lbl.setText("Starting culling…")
        for bar in (self.bar1, self.bar2):
            bar.setValue(0)
        for lbl in (self.lbl1, self.lbl2):
            lbl.setStyleSheet("color: #aaa; font-size: 12px;")
        for cnt in (self.cnt1, self.cnt2):
            cnt.setText("")
        if not self.coverage_chk.isChecked():
            self.lbl2.setStyleSheet("color: #555; font-size: 12px;")
            self.cnt2.setText("skipped")

        self._cull_worker = CullingWorker(
            input_dir, output_dir, self.sel_slider.value() / 100.0,
            player_coverage=self.coverage_chk.isChecked(),
        )
        self._cull_worker.stage_progress.connect(self._on_cull_progress)
        self._cull_worker.finished.connect(self._on_cull_finished)
        self._cull_worker.error.connect(self._on_error)
        self._cull_worker.start()

    def _on_cull_progress(self, stage: int, done: int, total: int, msg: str):
        bar, lbl, cnt = {
            0: (self.bar1, self.lbl1, self.cnt1),
            1: (self.bar2, self.lbl2, self.cnt2),
        }.get(stage, (None, None, None))

        if bar and total > 0:
            bar.setValue(int(done / total * 100))
            cnt.setText(f"{done:,} / {total:,}")

        for s, (b, l, c) in enumerate([(self.bar1, self.lbl1, self.cnt1),
                                        (self.bar2, self.lbl2, self.cnt2)]):
            l.setStyleSheet(
                "color: #4a90d9; font-size: 12px; font-weight: 600;" if s == stage
                else "color: #aaa; font-size: 12px;"
            )

        if stage == 0 and done == total and "passed" in msg and "culled" in msg:
            self.cull_count_lbl.setText(msg)

        self.status_lbl.setText(msg)

    def _on_cull_finished(self, summary: dict):
        self.bar1.setValue(100)
        self.bar2.setValue(100)
        for lbl in (self.lbl1, self.lbl2):
            lbl.setStyleSheet("color: #5cb85c; font-size: 12px;")

        n_total       = summary.get("total",       0)
        n_passed_cull = summary.get("passed_cull", summary.get("passed", 0))
        n_passed      = summary.get("passed",      0)
        # coverage_stats is authoritative; passed-count delta is the fallback
        n_promoted    = summary.get("coverage_stats", {}).get(
            "n_promoted", n_passed - n_passed_cull)

        self.cull_count_lbl.setText(
            f"After culling: {n_passed_cull:,} passed / {n_total - n_passed_cull:,} culled"
        )
        if n_promoted > 0:
            self.coverage_count_lbl.setText(
                f"After player coverage: {n_passed:,} passed (+{n_promoted:,} promoted)"
            )
        elif self.coverage_chk.isChecked():
            self.coverage_count_lbl.setText("Player coverage: no additional photos promoted")
        else:
            self.coverage_count_lbl.setText("")

        self.status_lbl.setText("Culling complete. Review bursts then click Run Crop && Color.")
        self.cull_btn.setEnabled(True)
        self.burst_btn.setEnabled(True)
        self.culled_btn.setEnabled(True)
        self.proc_btn.setEnabled(True)

        # Pre-start the burst review server now so it's ready when user clicks
        _start_burst_review(self._output_dir)

    # ── Phase 2: Crop + Color ──────────────────────────────────────────────────

    def _run_processing(self):
        if not self._output_dir:
            self.status_lbl.setText("Run culling first.")
            return

        self.proc_btn.setEnabled(False)
        self.review_btn.setEnabled(False)
        self.proc_stats_lbl.setText("")
        self.bar3.setValue(0)
        self.bar4.setValue(0)
        self.cnt3.setText("")
        self.cnt4.setText("")
        for lbl in (self.lbl3, self.lbl4):
            lbl.setStyleSheet("color: #aaa; font-size: 12px;")
        self.status_lbl.setText("Starting crop & color…")

        self._proc_worker = ProcessingWorker(self._output_dir)
        self._proc_worker.stage_progress.connect(self._on_proc_progress)
        self._proc_worker.finished.connect(self._on_proc_finished)
        self._proc_worker.error.connect(self._on_error)
        self._proc_worker.start()

    def _on_proc_progress(self, stage: int, done: int, total: int, msg: str):
        bar, lbl, cnt = {
            2: (self.bar3, self.lbl3, self.cnt3),
            3: (self.bar4, self.lbl4, self.cnt4),
        }.get(stage, (None, None, None))

        if bar and total > 0:
            bar.setValue(int(done / total * 100))
            cnt.setText(f"{done:,} / {total:,}")

        for s, (b, l, c) in enumerate([(self.bar3, self.lbl3, self.cnt3),
                                        (self.bar4, self.lbl4, self.cnt4)], start=2):
            l.setStyleSheet(
                "color: #4a90d9; font-size: 12px; font-weight: 600;" if s == stage
                else "color: #aaa; font-size: 12px;"
            )

        self.status_lbl.setText(msg)

    def _on_proc_finished(self, summary: dict):
        self.bar3.setValue(100)
        self.bar4.setValue(100)
        for lbl in (self.lbl3, self.lbl4):
            lbl.setStyleSheet("color: #5cb85c; font-size: 12px;")
        n = summary.get("processed", 0)
        self.proc_stats_lbl.setText(f"Done — {n:,} photos processed.")
        self.status_lbl.setText(f"Output: {self._output_dir}")
        self.proc_btn.setEnabled(True)
        self.review_btn.setEnabled(True)

    # ── Error ──────────────────────────────────────────────────────────────────

    def _on_error(self, msg: str):
        self.status_lbl.setText(f"Error: {msg}")
        self.cull_btn.setEnabled(True)
        self.proc_btn.setEnabled(True)

    # ── Review launchers ───────────────────────────────────────────────────────

    def _open_burst_review(self):
        _start_burst_review(self._output_dir)
        threading.Thread(target=lambda: webbrowser.open("http://127.0.0.1:8767/"), daemon=True).start()

    def _open_culled_review(self):
        _start_burst_review(self._output_dir)
        threading.Thread(target=lambda: webbrowser.open("http://127.0.0.1:8767/culled"), daemon=True).start()

    def _open_review(self):
        _start_review_server(self._output_dir)


# ─── Entry point ───────────────────────────────────────────────────────────────

def _app_icon() -> QIcon:
    if getattr(sys, 'frozen', False):
        base = Path(sys.executable).parent
    else:
        base = Path(__file__).parent.parent
    for name in ("icon.ico", "icon.png"):
        p = base / "assets" / name
        if p.exists():
            return QIcon(str(p))
    return QIcon()


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(_app_icon())
    if sys.platform == "darwin":
        app.setStyle("macos")
    elif sys.platform == "win32":
        app.setStyle("windowsvista")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
