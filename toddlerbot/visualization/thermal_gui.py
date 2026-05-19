"""Tkinter side panel for the MuJoCo thermal sandbox.

Runs in the main thread alongside the MuJoCo passive viewer (which owns its
own GLFW window). The simulation loop runs on a background daemon thread, so
the GUI talks to `MuJoCoThermalSim` via thread-safe setters
(`set_mode`, `request_reset_*`) and a read-only snapshot getter.

Features
--------
- Mode radios: Basic / Thermal Only / Thermal + Derate
- Reset Pose
- Reset Temps (uniform sample inside `domain_rand.temp_range`); seed entry
- Motor selection list (group toggles + per-motor checkboxes)
- 4x4 live grid of selected motors. Each cell shows ambient temp, max torque,
  commanded torque and derated torque. tau_cmd / tau_der turn red while the
  derate envelope clips them.
- "Open Graphs" popup: matplotlib grid covering only the **selected**
  motors. Housing+winding temperatures on the left axis, motor current (mA)
  on a right axis, and derated torque (Nm) on a second offset right axis.
  History grows unbounded — the X axis spans 0..elapsed and the live plot
  downsamples to a fixed budget of points per panel so long runs stay
  responsive. The history resets to t=0 whenever Reset Pose is pressed.
- "Export Graphs" button: writes a one-shot PNG with all 30 motors in a
  fixed 5x6 grid using the full history (no downsampling).
"""

from __future__ import annotations

import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from heat2torque import ThermalMode

GRID_COLS = 4
GRID_ROWS = 4
REFRESH_MS = 100
GRAPH_REFRESH_MS = 250
GRAPH_DT_SEC = REFRESH_MS / 1000.0
# τmax / τmax_ref update period in seconds. The dynamic max torque jitters
# with the winding temperature; throttle to 1 Hz for readability.
TMAX_REFRESH_SEC = 1.0
# Maximum points drawn per panel in the live popup. The full history is
# kept in memory; only the rendered series is downsampled.
GRAPH_RENDER_BUDGET = 1000
EXPORT_PANEL_ROWS = 5
EXPORT_PANEL_COLS = 6


class ThermalGUI:
    def __init__(
        self,
        sim,  # MuJoCoThermalSim
        motor_ordering: List[str],
        motor_groups: Dict[str, str],
        on_close: Optional[Callable[[], None]] = None,
    ):
        self.sim = sim
        self.motor_ordering = list(motor_ordering)
        self.motor_groups = dict(motor_groups)
        self.on_close = on_close

        self.root = tk.Tk()
        self.root.title("ToddlerBot Thermal GUI")
        self.root.geometry("1080x720")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.mode_var = tk.StringVar(value=self._mode_to_label(sim.mode))
        self.seed_var = tk.StringVar(value="")
        self.selected: Dict[str, tk.BooleanVar] = {
            name: tk.BooleanVar(value=(name in motor_ordering[: GRID_COLS * GRID_ROWS]))
            for name in self.motor_ordering
        }
        self.grid_cells: Dict[int, Dict[str, tk.Widget]] = {}

        # Unbounded rolling history. We keep the full series in memory so
        # the X axis can span 0..elapsed regardless of run length; only the
        # rendered series is downsampled (see `_downsample`). Reset Pose
        # wipes everything back to t=0.
        self._history_lock = threading.Lock()
        self._t_history: List[float] = []
        self._h_history: Dict[str, List[float]] = {
            name: [] for name in self.motor_ordering
        }
        self._w_history: Dict[str, List[float]] = {
            name: [] for name in self.motor_ordering
        }
        self._i_history: Dict[str, List[float]] = {
            name: [] for name in self.motor_ordering
        }
        self._tau_history: Dict[str, List[float]] = {
            name: [] for name in self.motor_ordering
        }
        self._elapsed = 0.0

        self._graph_window: Optional[tk.Toplevel] = None
        self._graph_canvas: Optional[FigureCanvasTkAgg] = None
        self._graph_fig: Optional[Figure] = None
        # name -> (ax_T, ax_I, ax_tau, line_h, line_w, line_i, line_tau)
        self._graph_axes: Dict[str, tuple] = {}
        # selection snapshot used to lay out the live popup. The popup is
        # rebuilt when this differs from the current selection.
        self._graph_selection: List[str] = []

        # 1 Hz throttle clock for τmax / τmax_ref labels.
        self._last_tmax_refresh: float = 0.0

        self._build_layout()
        self._refresh_loop()

    # -------------------------------------------------------------- layout
    def _build_layout(self):
        root = self.root
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        # --- top control bar --------------------------------------------
        ctrl = ttk.Frame(root, padding=(8, 6))
        ctrl.grid(row=0, column=0, columnspan=2, sticky="ew")

        ttk.Label(ctrl, text="Mode:").pack(side="left")
        for label in ("Basic", "Thermal Only", "Thermal + Derate"):
            ttk.Radiobutton(
                ctrl,
                text=label,
                variable=self.mode_var,
                value=label,
                command=self._on_mode_change,
            ).pack(side="left", padx=(4, 0))

        ttk.Separator(ctrl, orient="vertical").pack(
            side="left", fill="y", padx=10
        )
        ttk.Button(ctrl, text="Reset Pose", command=self._on_reset_pose).pack(
            side="left"
        )
        ttk.Label(ctrl, text="Seed:").pack(side="left", padx=(10, 2))
        ttk.Entry(ctrl, textvariable=self.seed_var, width=10).pack(side="left")
        ttk.Button(ctrl, text="Reset Temps", command=self._on_reset_temps).pack(
            side="left", padx=4
        )
        ttk.Separator(ctrl, orient="vertical").pack(
            side="left", fill="y", padx=10
        )
        ttk.Button(ctrl, text="Open Graphs", command=self._open_graph_window).pack(
            side="left"
        )
        ttk.Button(ctrl, text="Export Graphs", command=self._export_graphs).pack(
            side="left", padx=4
        )

        self.status_var = tk.StringVar(value="ready")
        ttk.Label(ctrl, textvariable=self.status_var, foreground="#666").pack(
            side="right"
        )

        # --- left motor selector ----------------------------------------
        selector = ttk.LabelFrame(root, text="Motor Selection", padding=6)
        selector.grid(row=1, column=0, sticky="nsw", padx=(8, 4), pady=(0, 8))

        groups = sorted(set(self.motor_groups.values()))
        group_bar = ttk.Frame(selector)
        group_bar.pack(fill="x", pady=(0, 4))
        ttk.Label(group_bar, text="Group:").pack(side="left")
        for g in groups:
            ttk.Button(
                group_bar,
                text=g,
                width=6,
                command=lambda g=g: self._toggle_group(g),
            ).pack(side="left", padx=2)
        ttk.Button(group_bar, text="all", width=4, command=self._select_all).pack(
            side="left", padx=2
        )
        ttk.Button(group_bar, text="none", width=4, command=self._select_none).pack(
            side="left", padx=2
        )

        # scrollable checkbox list
        list_canvas = tk.Canvas(selector, width=240, highlightthickness=0)
        list_scroll = ttk.Scrollbar(
            selector, orient="vertical", command=list_canvas.yview
        )
        list_canvas.configure(yscrollcommand=list_scroll.set)
        list_canvas.pack(side="left", fill="both", expand=True)
        list_scroll.pack(side="right", fill="y")

        list_inner = ttk.Frame(list_canvas)
        list_canvas.create_window((0, 0), window=list_inner, anchor="nw")
        list_inner.bind(
            "<Configure>",
            lambda e: list_canvas.configure(scrollregion=list_canvas.bbox("all")),
        )
        for name in self.motor_ordering:
            grp = self.motor_groups.get(name, "?")
            ttk.Checkbutton(
                list_inner,
                text=f"{name}  [{grp}]",
                variable=self.selected[name],
                command=self._on_selection_change,
            ).pack(anchor="w")

        # --- right grid -------------------------------------------------
        grid_outer = ttk.LabelFrame(root, text="Live Motor State", padding=6)
        grid_outer.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=(0, 8))
        grid_outer.rowconfigure(0, weight=1)
        grid_outer.columnconfigure(0, weight=1)

        grid_canvas = tk.Canvas(grid_outer, highlightthickness=0)
        grid_scroll = ttk.Scrollbar(
            grid_outer, orient="vertical", command=grid_canvas.yview
        )
        grid_canvas.configure(yscrollcommand=grid_scroll.set)
        grid_canvas.grid(row=0, column=0, sticky="nsew")
        grid_scroll.grid(row=0, column=1, sticky="ns")

        self.grid_inner = ttk.Frame(grid_canvas)
        grid_canvas.create_window((0, 0), window=self.grid_inner, anchor="nw")
        self.grid_inner.bind(
            "<Configure>",
            lambda e: grid_canvas.configure(scrollregion=grid_canvas.bbox("all")),
        )

        self._rebuild_grid()

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _mode_to_label(mode: ThermalMode) -> str:
        m = int(mode)
        if m & int(ThermalMode.USE_DERATE):
            return "Thermal + Derate"
        if m & int(ThermalMode.USE_THERMAL):
            return "Thermal Only"
        return "Basic"

    @staticmethod
    def _label_to_mode(label: str) -> ThermalMode:
        # Thermal/Derate = 결합 2차 LPTN(논문 최종 모델) → USE_COUPLING 포함.
        if label == "Thermal + Derate":
            return (
                ThermalMode.USE_DERATE
                | ThermalMode.USE_THERMAL
                | ThermalMode.MODEL_ORDER_2
                | ThermalMode.USE_COUPLING
            )
        if label == "Thermal Only":
            return (
                ThermalMode.USE_THERMAL
                | ThermalMode.MODEL_ORDER_2
                | ThermalMode.USE_COUPLING
            )
        return ThermalMode.DISABLE

    # ------------------------------------------------------------- actions
    def _on_mode_change(self):
        mode = self._label_to_mode(self.mode_var.get())
        self.sim.set_mode(mode)
        self.status_var.set(f"mode -> {self.mode_var.get()}")

    def _on_reset_pose(self):
        self.sim.request_reset_pose()
        self._clear_history()
        self.status_var.set("reset pose queued (graphs t=0)")

    def _clear_history(self):
        """Wipe all rolling buffers so graphs restart from index 0."""
        with self._history_lock:
            self._elapsed = 0.0
            self._t_history.clear()
            for name in self.motor_ordering:
                self._h_history[name].clear()
                self._w_history[name].clear()
                self._i_history[name].clear()
                self._tau_history[name].clear()
        # Force the next refresh tick to repaint τmax labels immediately so
        # the user sees a fresh value right after Reset Pose.
        self._last_tmax_refresh = 0.0

    def _on_reset_temps(self):
        seed_str = self.seed_var.get().strip()
        seed = int(seed_str) if seed_str.isdigit() else None
        self.sim.request_reset_temps(seed=seed)
        # Re-sampling thermal state means previous history no longer aligns
        # with the new initial temps. Wipe buffers so the graphs restart at
        # t=0 with the freshly sampled state.
        self._clear_history()
        self.status_var.set(
            f"reset temps queued (seed={seed if seed is not None else 'auto'}, graphs t=0)"
        )

    def _toggle_group(self, group: str):
        members = [n for n, g in self.motor_groups.items() if g == group]
        if not members:
            return
        currently_all_on = all(self.selected[m].get() for m in members)
        new_val = not currently_all_on
        for m in members:
            self.selected[m].set(new_val)
        self._on_selection_change()

    def _select_all(self):
        for v in self.selected.values():
            v.set(True)
        self._on_selection_change()

    def _select_none(self):
        for v in self.selected.values():
            v.set(False)
        self._on_selection_change()

    def _on_selection_change(self):
        self._rebuild_grid()
        if self._graph_window is not None and self._graph_window.winfo_exists():
            # Layout depends on the selection; rebuild axes to match.
            self._rebuild_graph_axes()

    # --------------------------------------------------------------- grid
    def _selected_names(self) -> List[str]:
        return [n for n in self.motor_ordering if self.selected[n].get()]

    def _make_cell(self, parent: tk.Widget, name: str) -> Dict[str, tk.Widget]:
        cell = ttk.LabelFrame(parent, text=name, padding=4)
        amb_var = tk.StringVar(value="amb       -- °C")
        h_var = tk.StringVar(value="hous     -- °C")
        w_var = tk.StringVar(value="wind     -- °C")
        # τmax_ref / τmax_ref_hot are the cold (25 °C) and hot (100 °C)
        # reference stall torques. They sit right above the live τmax so the
        # derate effect is easy to compare against both extremes.
        tmax_ref_var = tk.StringVar(value="τmax@25°  -- Nm")
        tmax_ref_hot_var = tk.StringVar(value="τmax@100° -- Nm")
        tmax_var = tk.StringVar(value="τmax now  -- Nm")
        tcmd_var = tk.StringVar(value="τcmd      -- Nm")
        tder_var = tk.StringVar(value="τder      -- Nm")

        font_mono = ("TkFixedFont", 9)
        amb_lbl = ttk.Label(cell, textvariable=amb_var, font=font_mono)
        h_lbl = ttk.Label(cell, textvariable=h_var, font=font_mono)
        w_lbl = ttk.Label(cell, textvariable=w_var, font=font_mono)
        tmax_ref_lbl = ttk.Label(
            cell, textvariable=tmax_ref_var, font=font_mono, foreground="#666666"
        )
        tmax_ref_hot_lbl = ttk.Label(
            cell, textvariable=tmax_ref_hot_var, font=font_mono, foreground="#888888"
        )
        tmax_lbl = ttk.Label(cell, textvariable=tmax_var, font=font_mono)
        tcmd_lbl = ttk.Label(cell, textvariable=tcmd_var, font=font_mono)
        tder_lbl = ttk.Label(cell, textvariable=tder_var, font=font_mono)

        amb_lbl.grid(row=0, column=0, sticky="w")
        h_lbl.grid(row=1, column=0, sticky="w")
        w_lbl.grid(row=2, column=0, sticky="w")
        tmax_ref_lbl.grid(row=3, column=0, sticky="w")
        tmax_ref_hot_lbl.grid(row=4, column=0, sticky="w")
        tmax_lbl.grid(row=5, column=0, sticky="w")
        tcmd_lbl.grid(row=6, column=0, sticky="w")
        tder_lbl.grid(row=7, column=0, sticky="w")

        return {
            "name": name,
            "frame": cell,
            "amb_var": amb_var,
            "h_var": h_var,
            "w_var": w_var,
            "tmax_ref_var": tmax_ref_var,
            "tmax_ref_hot_var": tmax_ref_hot_var,
            "tmax_var": tmax_var,
            "tcmd_var": tcmd_var,
            "tder_var": tder_var,
            "h_lbl": h_lbl,
            "tmax_ref_lbl": tmax_ref_lbl,
            "tmax_ref_hot_lbl": tmax_ref_hot_lbl,
            "tmax_lbl": tmax_lbl,
            "tcmd_lbl": tcmd_lbl,
            "tder_lbl": tder_lbl,
        }

    def _rebuild_grid(self):
        for child in self.grid_inner.winfo_children():
            child.destroy()
        self.grid_cells.clear()

        names = self._selected_names()
        if not names:
            ttk.Label(
                self.grid_inner, text="(no motors selected)", foreground="#888"
            ).grid(row=0, column=0, padx=8, pady=8)
            return

        for idx, name in enumerate(names):
            r, c = divmod(idx, GRID_COLS)
            cell = self._make_cell(self.grid_inner, name)
            cell["frame"].grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
            self.grid_inner.columnconfigure(c, weight=1)
            self.grid_cells[idx] = cell

    # -------------------------------------------------------- graph popup
    def _open_graph_window(self):
        if self._graph_window is not None and self._graph_window.winfo_exists():
            self._graph_window.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("Motor Temperature / Torque / Current — Selected Motors")
        win.geometry("1400x900")
        win.protocol("WM_DELETE_WINDOW", self._close_graph_window)

        self._graph_fig = Figure(figsize=(14, 9), dpi=90)
        self._graph_canvas = FigureCanvasTkAgg(self._graph_fig, master=win)
        self._graph_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._graph_window = win
        self._rebuild_graph_axes()
        self._graph_refresh_loop()

    def _close_graph_window(self):
        try:
            if self._graph_canvas is not None:
                self._graph_canvas.get_tk_widget().destroy()
        except Exception:
            pass
        if self._graph_fig is not None:
            plt.close(self._graph_fig)
        self._graph_canvas = None
        self._graph_fig = None
        self._graph_axes.clear()
        if self._graph_window is not None:
            try:
                self._graph_window.destroy()
            except Exception:
                pass
        self._graph_window = None

    def _rebuild_graph_axes(self):
        if self._graph_fig is None:
            return
        self._graph_fig.clf()
        self._graph_axes.clear()

        names = self._selected_names()
        self._graph_selection = list(names)

        if not names:
            ax = self._graph_fig.add_subplot(1, 1, 1)
            ax.text(
                0.5, 0.5, "(no motors selected)",
                ha="center", va="center", color="#888",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            self._graph_canvas.draw_idle()
            return

        n = len(names)
        cols = min(GRID_COLS, n)
        rows = (n + cols - 1) // cols

        for idx, name in enumerate(names):
            r = idx // cols
            c = idx % cols
            ax_T = self._graph_fig.add_subplot(rows, cols, idx + 1)
            ax_I = ax_T.twinx()
            ax_tau = ax_T.twinx()
            ax_tau.spines["right"].set_position(("axes", 1.18))
            ax_tau.set_frame_on(True)
            ax_tau.patch.set_visible(False)

            ax_T.set_title(name, fontsize=8)
            ax_T.tick_params(axis="both", labelsize=6)
            ax_I.tick_params(axis="both", labelsize=6, colors="#d62728")
            ax_tau.tick_params(axis="both", labelsize=6, colors="#2ca02c")

            if r == rows - 1:
                ax_T.set_xlabel("t (s)", fontsize=7)
            if c == 0:
                ax_T.set_ylabel("T (°C)", color="#1f77b4", fontsize=7)
            if c == cols - 1:
                ax_I.set_ylabel("I (mA)", color="#d62728", fontsize=7)
                ax_tau.set_ylabel("τ (Nm)", color="#2ca02c", fontsize=7)

            (line_h,) = ax_T.plot([], [], color="#1f77b4", lw=1.1, label="housing")
            (line_w,) = ax_T.plot(
                [], [], color="#1f77b4", lw=0.9, ls="--", label="winding"
            )
            (line_i,) = ax_I.plot([], [], color="#d62728", lw=0.9, label="current")
            (line_tau,) = ax_tau.plot(
                [], [], color="#2ca02c", lw=0.9, label="torque"
            )

            self._graph_axes[name] = (
                ax_T, ax_I, ax_tau, line_h, line_w, line_i, line_tau,
            )

        self._graph_fig.subplots_adjust(
            left=0.05, right=0.93, top=0.95, bottom=0.06,
            wspace=0.55, hspace=0.55,
        )
        self._graph_canvas.draw_idle()

    @staticmethod
    def _downsample(
        xs: List[float],
        ys: List[float],
        budget: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reduce a series to at most `budget` evenly spaced points."""
        n = len(xs)
        if n == 0:
            return np.empty(0), np.empty(0)
        if n <= budget:
            return np.asarray(xs), np.asarray(ys)
        idx = np.linspace(0, n - 1, budget).astype(np.int64)
        return np.asarray(xs)[idx], np.asarray(ys)[idx]

    def _graph_refresh_loop(self):
        if self._graph_window is None or not self._graph_window.winfo_exists():
            return
        if self._graph_fig is None or self._graph_canvas is None:
            return

        # Rebuild if the user toggled the selection while the popup was open
        # but the rebuild path didn't fire (defensive).
        if self._selected_names() != self._graph_selection:
            self._rebuild_graph_axes()

        with self._history_lock:
            t_arr = list(self._t_history)
            histories = {
                name: (
                    list(self._h_history[name]),
                    list(self._w_history[name]),
                    list(self._i_history[name]),
                    list(self._tau_history[name]),
                )
                for name in self._graph_axes
            }

        if t_arr:
            t_lo_x = t_arr[0]
            t_hi_x = t_arr[-1] if t_arr[-1] > t_arr[0] else t_arr[0] + 1.0

            for name, axes in self._graph_axes.items():
                ax_T, ax_I, ax_tau, line_h, line_w, line_i, line_tau = axes
                h_arr, w_arr, i_arr, tau_arr = histories.get(name, ([], [], [], []))
                m = min(len(t_arr), len(h_arr), len(w_arr), len(i_arr), len(tau_arr))
                if m == 0:
                    line_h.set_data([], [])
                    line_w.set_data([], [])
                    line_i.set_data([], [])
                    line_tau.set_data([], [])
                    continue

                xs_full = t_arr[:m]
                h_full = h_arr[:m]
                w_full = w_arr[:m]
                i_full = i_arr[:m]
                tau_full = tau_arr[:m]

                xs_h, ys_h = self._downsample(xs_full, h_full, GRAPH_RENDER_BUDGET)
                _,    ys_w = self._downsample(xs_full, w_full, GRAPH_RENDER_BUDGET)
                _,    ys_i = self._downsample(xs_full, i_full, GRAPH_RENDER_BUDGET)
                _,    ys_tau = self._downsample(xs_full, tau_full, GRAPH_RENDER_BUDGET)

                line_h.set_data(xs_h, ys_h)
                line_w.set_data(xs_h, ys_w)
                line_i.set_data(xs_h, ys_i)
                line_tau.set_data(xs_h, ys_tau)

                ax_T.set_xlim(t_lo_x, t_hi_x)

                t_lo = min(min(h_full), min(w_full))
                t_hi = max(max(h_full), max(w_full))
                t_margin = max(2.0, (t_hi - t_lo) * 0.1)
                ax_T.set_ylim(t_lo - t_margin, t_hi + t_margin)

                i_lo = min(i_full)
                i_hi = max(i_full)
                i_margin = max(50.0, (i_hi - i_lo) * 0.1)
                ax_I.set_ylim(i_lo - i_margin, i_hi + i_margin)

                tau_lo = min(tau_full)
                tau_hi = max(tau_full)
                tau_margin = max(0.05, (tau_hi - tau_lo) * 0.1)
                ax_tau.set_ylim(tau_lo - tau_margin, tau_hi + tau_margin)

            try:
                self._graph_canvas.draw_idle()
            except Exception:
                pass

        self.root.after(GRAPH_REFRESH_MS, self._graph_refresh_loop)

    # --------------------------------------------------------- export PNG
    def _export_graphs(self):
        """One-shot dump: 5x6 grid of all 30 motors using the full history."""
        with self._history_lock:
            t_arr = list(self._t_history)
            histories = {
                name: (
                    list(self._h_history[name]),
                    list(self._w_history[name]),
                    list(self._i_history[name]),
                    list(self._tau_history[name]),
                )
                for name in self.motor_ordering
            }

        if not t_arr:
            self.status_var.set("export skipped: no samples yet")
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        default_name = f"thermal_export_{ts}.png"
        path = filedialog.asksaveasfilename(
            title="Export thermal graphs (PNG)",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not path:
            self.status_var.set("export cancelled")
            return

        rows = EXPORT_PANEL_ROWS
        cols = EXPORT_PANEL_COLS
        names = list(self.motor_ordering)[: rows * cols]

        fig = Figure(figsize=(20, 12), dpi=110)
        for idx, name in enumerate(names):
            r = idx // cols
            c = idx % cols
            ax_T = fig.add_subplot(rows, cols, idx + 1)
            ax_I = ax_T.twinx()
            ax_tau = ax_T.twinx()
            ax_tau.spines["right"].set_position(("axes", 1.18))
            ax_tau.set_frame_on(True)
            ax_tau.patch.set_visible(False)

            h_full, w_full, i_full, tau_full = histories.get(name, ([], [], [], []))
            m = min(len(t_arr), len(h_full), len(w_full), len(i_full), len(tau_full))
            xs = t_arr[:m]

            ax_T.plot(xs, h_full[:m], color="#1f77b4", lw=1.0, label="housing")
            ax_T.plot(xs, w_full[:m], color="#1f77b4", lw=0.8, ls="--", label="winding")
            ax_I.plot(xs, i_full[:m], color="#d62728", lw=0.8, label="current")
            ax_tau.plot(xs, tau_full[:m], color="#2ca02c", lw=0.8, label="torque")

            ax_T.set_title(name, fontsize=9)
            ax_T.tick_params(axis="both", labelsize=7)
            ax_I.tick_params(axis="both", labelsize=7, colors="#d62728")
            ax_tau.tick_params(axis="both", labelsize=7, colors="#2ca02c")
            if r == rows - 1:
                ax_T.set_xlabel("t (s)", fontsize=8)
            if c == 0:
                ax_T.set_ylabel("T (°C)", color="#1f77b4", fontsize=8)
            if c == cols - 1:
                ax_I.set_ylabel("I (mA)", color="#d62728", fontsize=8)
                ax_tau.set_ylabel("τ (Nm)", color="#2ca02c", fontsize=8)

        fig.subplots_adjust(
            left=0.04, right=0.94, top=0.96, bottom=0.05,
            wspace=0.55, hspace=0.55,
        )
        try:
            fig.savefig(path, bbox_inches="tight")
            self.status_var.set(f"exported -> {os.path.basename(path)}")
        except Exception as e:
            self.status_var.set(f"export failed: {e}")
        finally:
            plt.close(fig)

    # ------------------------------------------------------------ refresh
    def _refresh_loop(self):
        try:
            snap = self.sim.get_thermal_snapshot()
        except Exception:
            snap = {}

        # update rolling history
        if snap:
            with self._history_lock:
                # First sample after a Reset Pose lands at t=0.
                t = 0.0 if not self._t_history else self._elapsed + GRAPH_DT_SEC
                self._elapsed = t
                self._t_history.append(t)
                for name, data in snap.items():
                    self._h_history[name].append(data["h_t"])
                    self._w_history[name].append(data["w_t"])
                    self._i_history[name].append(data["current_mA"])
                    self._tau_history[name].append(data["tau_der"])

        # τmax / τmax_ref refresh once per second so the value is readable.
        now = time.monotonic()
        update_tmax = (now - self._last_tmax_refresh) >= TMAX_REFRESH_SEC
        if update_tmax:
            self._last_tmax_refresh = now

        # update grid cells
        for idx, cell in self.grid_cells.items():
            name = cell["name"]
            data = snap.get(name)
            if data is None:
                cell["amb_var"].set("amb       -- °C")
                cell["h_var"].set("hous     -- °C")
                cell["w_var"].set("wind     -- °C")
                if update_tmax:
                    cell["tmax_ref_var"].set("τmax@25°  -- Nm")
                    cell["tmax_ref_hot_var"].set("τmax@100° -- Nm")
                    cell["tmax_var"].set("τmax now  -- Nm")
                cell["tcmd_var"].set("τcmd      -- Nm")
                cell["tder_var"].set("τder      -- Nm")
                continue

            h_t = data["h_t"]
            w_t = data["w_t"]
            a_t = data["a_t"]
            spec = data["spec_t_max"]
            tmax = data["tau_max"]
            tmax_ref = data["tau_max_ref"]
            tmax_ref_hot = data["tau_max_ref_hot"]
            tcmd = data["tau_cmd"]
            tder = data["tau_der"]

            cell["amb_var"].set(f"amb     {a_t:6.2f} °C")
            cell["h_var"].set(f"hous    {h_t:6.2f} °C")
            cell["w_var"].set(f"wind    {w_t:6.2f} °C")
            if update_tmax:
                cell["tmax_ref_var"].set(f"τmax@25°{tmax_ref:7.3f} Nm")
                cell["tmax_ref_hot_var"].set(f"τmax@100°{tmax_ref_hot:6.3f} Nm")
                cell["tmax_var"].set(f"τmax now{tmax:7.3f} Nm")
                # Visual cue: dim red if current envelope shrank vs. reference.
                if tmax < tmax_ref - 1e-3:
                    cell["tmax_lbl"].configure(foreground="#cc7700")
                else:
                    cell["tmax_lbl"].configure(foreground="#000000")
            cell["tcmd_var"].set(f"τcmd    {tcmd:+7.3f} Nm")
            cell["tder_var"].set(f"τder    {tder:+7.3f} Nm")

            # housing color: red on overheat, orange near spec_t_max, else default
            if data["overheat"]:
                cell["h_lbl"].configure(foreground="#cc0000")
            elif h_t >= spec - 5.0:
                cell["h_lbl"].configure(foreground="#cc7700")
            else:
                cell["h_lbl"].configure(foreground="#000000")

            # tau_cmd / tau_der: red iff currently clipped, default otherwise
            if data["clipped"]:
                cell["tcmd_lbl"].configure(foreground="#cc0000")
                cell["tder_lbl"].configure(foreground="#cc0000")
            else:
                cell["tcmd_lbl"].configure(foreground="#000000")
                cell["tder_lbl"].configure(foreground="#000000")

        self.root.after(REFRESH_MS, self._refresh_loop)

    # ------------------------------------------------------------- close
    def _on_close(self):
        if self.on_close is not None:
            try:
                self.on_close()
            except Exception:
                pass
        self._close_graph_window()
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()


def launch_gui_blocking(sim, on_close: Optional[Callable[[], None]] = None):
    """Convenience wrapper for `run_thermal_gui.py`."""
    gui = ThermalGUI(
        sim,
        motor_ordering=sim.robot.motor_ordering,
        motor_groups=sim.robot.joint_groups,
        on_close=on_close,
    )
    gui.mainloop()
