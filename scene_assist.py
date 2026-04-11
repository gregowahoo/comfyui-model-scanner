#!/usr/bin/env python3
"""
scene_assist.py  –  LTX-Video 2.3 Scene Planner
Designed for the WhatDreamsCost FFLF ComfyUI workflow.

No external dependencies — pure standard-library (tkinter only).
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json, math, os, re
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import List

# ─────────────────────────────────────────────────────────────────────────────
# LTX-Video 2.3 constants
# ─────────────────────────────────────────────────────────────────────────────

# Valid frame counts for LTX-Video: 8n+1  (temporal compression factor)
VALID_FRAMES = [8 * n + 1 for n in range(3, 42)]   # 25 … 329

RESOLUTIONS = [
    "1280x720", "1024x576", "1024x640", "960x544",
    "848x480",  "832x480",  "768x512",  "704x480",
    "720x1280", "576x1024", "544x960",  "480x832",  "512x704",
    "640x640",
]

FPS_OPTS = [24, 25, 30]

PRESETS = {          # name: (steps, cfg)
    "Draft":    (20, 2.5),
    "Fast":     (25, 3.0),
    "Balanced": (30, 3.5),
    "Quality":  (40, 4.0),
    "Max":      (50, 5.0),
}

NEG_DEFAULT = "worst quality, inconsistent motion, blurry, jittery, distorted"
SPEECH_WPM  = 130   # typical narration / dialogue pace

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  (dark, ComfyUI-adjacent)
# ─────────────────────────────────────────────────────────────────────────────
BG   = "#16172a"
BG2  = "#1f2040"
BG3  = "#272852"
BG4  = "#2e3060"
ACC  = "#7b5ea7"
AC2  = "#a084e8"
AC3  = "#c4b0f5"
DIM  = "#5a5a88"
TXT  = "#dde0f5"
TXT2 = "#a0a4d0"
GRN  = "#6ecb73"
YEL  = "#fcd56a"
RED  = "#f26a6a"
BDR  = "#383870"
TL_BAR = "#3a3a70"
TL_FF  = "#6ecb73"   # first-frame marker
TL_LF  = "#f26a6a"   # last-frame marker
TL_MF  = "#fcd56a"   # middle-frame marker

# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MFrame:
    """One middle keyframe entry."""
    path:  str   = ""
    pos:   float = 0.0    # absolute: frame number OR seconds, per Shot.ins_mode
    label: str   = ""

    def to_dict(self):        return asdict(self)
    @classmethod
    def from_dict(cls, d):    return cls(**d)


@dataclass
class Shot:
    name:       str   = "Shot 1"
    first:      str   = ""
    middles:    List  = field(default_factory=list)   # List[MFrame]
    last:       str   = ""
    positive:   str   = ""
    negative:   str   = NEG_DEFAULT
    resolution: str   = "1024x576"
    frames:     int   = 97
    fps:        int   = 24
    steps:      int   = 30
    cfg:        float = 3.5
    seed:       int   = -1
    ins_mode:   str   = "frames"    # "frames" | "seconds"
    notes:      str   = ""

    @property
    def dur(self) -> float:
        return round(self.frames / self.fps, 3)

    def to_dict(self):
        d = asdict(self)
        d["middles"] = [m.to_dict() for m in self.middles]
        return d

    @classmethod
    def from_dict(cls, d: dict):
        raw_mid = d.pop("middles", [])
        s = cls(**d)
        s.middles = [MFrame.from_dict(m) for m in raw_mid]
        return s


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def nearest_valid_frames(target_frames: float) -> int:
    """Return closest LTX-valid frame count (8n+1) to target."""
    return min(VALID_FRAMES, key=lambda v: abs(v - target_frames))

def seconds_to_frames(secs: float, fps: int) -> int:
    return nearest_valid_frames(secs * fps)

def frames_to_seconds(frames: int, fps: int) -> float:
    return round(frames / fps, 2)

def estimate_speech_seconds(text: str) -> float:
    """Estimate duration of spoken dialogue found inside quotation marks."""
    quoted = re.findall(r'"([^"]+)"', text) + re.findall(r"'([^']+)'", text)
    total_words = sum(len(q.split()) for q in quoted)
    return round(total_words / SPEECH_WPM * 60, 2) if total_words else 0.0

def motion_complexity(prompt: str) -> str:
    """Rough motion complexity from prompt keywords → 'low'|'medium'|'high'."""
    low_kws  = ["still", "static", "slow", "gently", "calm", "fade", "breathe"]
    high_kws = ["run", "dash", "explode", "rapid", "burst", "action", "fight",
                "chase", "spin", "whirl", "fast", "quick", "slam", "jump"]
    p = prompt.lower()
    if any(k in p for k in high_kws):  return "high"
    if any(k in p for k in low_kws):   return "low"
    return "medium"

def recommend_shot(positive: str, dialogue: str = "") -> dict:
    """
    Given a positive prompt and optional extra dialogue text, return a dict of
    recommended LTX settings and a list of human-readable advice strings.
    """
    speech_s = estimate_speech_seconds(positive + " " + dialogue)
    complexity = motion_complexity(positive)

    # Estimate scene duration
    base_s = max(speech_s * 1.15, 2.0)
    if complexity == "high":   base_s = max(base_s, 3.0)
    if complexity == "low":    base_s = max(base_s, 1.5)

    fps = 24
    target_f = base_s * fps
    rec_frames = nearest_valid_frames(target_f)
    actual_dur = round(rec_frames / fps, 2)

    # Steps / CFG
    if complexity == "high":
        steps, cfg = PRESETS["Quality"]
    elif complexity == "low":
        steps, cfg = PRESETS["Fast"]
    else:
        steps, cfg = PRESETS["Balanced"]

    # Middle-frame timing suggestions
    mid_suggestions = []
    if rec_frames >= 49:
        mid_suggestions.append(round(actual_dur * 0.5, 2))   # midpoint
    if rec_frames >= 97:
        mid_suggestions.append(round(actual_dur * 0.33, 2))
        mid_suggestions.append(round(actual_dur * 0.67, 2))

    advice = []
    advice.append(f"Complexity: {complexity}")
    if speech_s:
        advice.append(f"Detected ~{speech_s}s of dialogue in prompt")
    advice.append(f"Recommended duration: ~{actual_dur}s  ({rec_frames} frames at {fps} fps)")
    if mid_suggestions:
        pts = ", ".join(f"{s}s" for s in sorted(mid_suggestions))
        advice.append(f"Suggested middle-frame positions: {pts}")
    advice.append(f"Steps: {steps}   CFG: {cfg}")

    return {
        "frames": rec_frames,
        "fps": fps,
        "steps": steps,
        "cfg": cfg,
        "dur": actual_dur,
        "advice": advice,
        "mid_seconds": sorted(mid_suggestions),
    }


def fmt_settings(shot: Shot) -> str:
    """Format a shot's settings as a clean reference block for ComfyUI entry."""
    lines = [
        f"=== {shot.name} ===",
        f"Resolution  : {shot.resolution}",
        f"Frames      : {shot.frames}",
        f"FPS         : {shot.fps}",
        f"Duration    : {shot.dur}s",
        f"Steps       : {shot.steps}",
        f"CFG         : {shot.cfg}",
        f"Seed        : {shot.seed if shot.seed != -1 else 'random'}",
        f"Insert mode : {shot.ins_mode}",
        "",
        "POSITIVE PROMPT:",
        shot.positive or "(none)",
        "",
        "NEGATIVE PROMPT:",
        shot.negative or "(none)",
    ]
    if shot.middles:
        lines += ["", "MIDDLE FRAMES:"]
        for i, mf in enumerate(shot.middles, 1):
            unit = shot.ins_mode
            lines.append(f"  MF{i}  pos={mf.pos} {unit}"
                         + (f"  label={mf.label}" if mf.label else "")
                         + (f"  path={mf.path}" if mf.path else ""))
    if shot.notes:
        lines += ["", "NOTES:", shot.notes]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Reusable styled widget helpers
# ─────────────────────────────────────────────────────────────────────────────

def _label(parent, text, size=9, bold=False, fg=TXT, bg=BG, **kw):
    font = ("Segoe UI", size, "bold" if bold else "normal")
    return tk.Label(parent, text=text, font=font, fg=fg, bg=bg, **kw)

def _section(parent, title, bg=BG):
    """Return a labelled section frame with a divider line."""
    outer = tk.Frame(parent, bg=bg)
    outer.pack(fill="x", pady=(10, 4))

    header = tk.Frame(outer, bg=bg)
    header.pack(fill="x")
    _label(header, title, size=8, bold=True, fg=DIM, bg=bg).pack(side="left")
    tk.Frame(header, bg=BDR, height=1).pack(side="left", fill="x", expand=True, padx=(8, 0), pady=7)

    inner = tk.Frame(outer, bg=bg)
    inner.pack(fill="x")
    return inner

def _entry(parent, textvariable=None, width=40, bg=BG3, **kw):
    e = tk.Entry(parent, textvariable=textvariable, width=width, bg=bg,
                 fg=TXT, insertbackground=TXT, relief="flat",
                 highlightbackground=BDR, highlightcolor=ACC, highlightthickness=1,
                 font=("Segoe UI", 10), **kw)
    return e

def _btn(parent, text, cmd, bg=BG4, fg=TXT, padx=8, pady=4, **kw):
    return tk.Button(parent, text=text, command=cmd,
                     bg=bg, fg=fg, activebackground=ACC, activeforeground=TXT,
                     relief="flat", font=("Segoe UI", 9), padx=padx, pady=pady,
                     cursor="hand2", **kw)

def _text(parent, height=3, width=60, bg=BG3, **kw):
    t = tk.Text(parent, height=height, width=width, bg=bg,
                fg=TXT, insertbackground=TXT, relief="flat",
                highlightbackground=BDR, highlightcolor=ACC, highlightthickness=1,
                font=("Segoe UI", 10), wrap="word", **kw)
    return t

def _combobox(parent, values, textvariable=None, width=18):
    style = ttk.Style()
    style.theme_use("default")
    style.configure("Dark.TCombobox",
                    fieldbackground=BG3, background=BG3, foreground=TXT,
                    arrowcolor=TXT2, selectbackground=ACC, selectforeground=TXT,
                    bordercolor=BDR, lightcolor=BDR, darkcolor=BDR)
    cb = ttk.Combobox(parent, values=values, textvariable=textvariable,
                      width=width, style="Dark.TCombobox", state="readonly",
                      font=("Segoe UI", 10))
    cb.option_add("*TCombobox*Listbox.background", BG3)
    cb.option_add("*TCombobox*Listbox.foreground", TXT)
    cb.option_add("*TCombobox*Listbox.selectBackground", ACC)
    return cb


# ─────────────────────────────────────────────────────────────────────────────
# Timeline canvas
# ─────────────────────────────────────────────────────────────────────────────

class TimelineCanvas(tk.Canvas):
    """Visualises first / middle / last keyframe positions."""

    H      = 62
    PAD_H  = 12    # horizontal padding each side
    BAR_Y  = 26
    BAR_H  = 12
    LBL_Y  = 50

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG2, height=self.H,
                         highlightthickness=0, **kw)
        self._shot = None

    def draw(self, shot: Shot):
        self._shot = shot
        self.delete("all")
        w = self.winfo_width() or 500
        bx0 = self.PAD_H
        bx1 = w - self.PAD_H
        bw  = bx1 - bx0
        by0 = self.BAR_Y
        by1 = by0 + self.BAR_H

        # Background bar
        self.create_rectangle(bx0, by0, bx1, by1, fill=TL_BAR, outline="", tags="bar")

        # Active region (gradient illusion via two rects)
        self.create_rectangle(bx0, by0, bx1, by1,
                              fill=BG4, outline=BDR, width=1)

        dur = shot.dur or 1.0

        def x_for(secs):
            return bx0 + (secs / dur) * bw

        def draw_marker(x, colour, symbol="■", lbl=""):
            r = 7
            self.create_oval(x-r, by0-r//2, x+r, by1+r//2,
                             fill=colour, outline=BG2, width=2)
            if symbol == "■":
                self.create_rectangle(x-4, by0+1, x+4, by1-1,
                                      fill=colour, outline="")
            if lbl:
                self.create_text(x, self.LBL_Y, text=lbl,
                                 fill=TXT2, font=("Segoe UI", 7), anchor="center")

        # First frame
        draw_marker(bx0, TL_FF, lbl="0s")

        # Last frame
        draw_marker(bx1, TL_LF, lbl=f"{dur}s")

        # Middle frames
        for i, mf in enumerate(shot.middles, 1):
            if shot.ins_mode == "seconds":
                secs = mf.pos
            else:
                secs = mf.pos / shot.fps if shot.fps else 0.0
            secs = max(0.0, min(secs, dur))
            x = x_for(secs)
            lbl_text = f"{round(secs,1)}s" if shot.ins_mode == "seconds" \
                       else f"f{int(mf.pos)}"
            draw_marker(x, TL_MF, lbl=lbl_text)

        # Legend
        for col, txt, px in [(TL_FF,"First",bx0+22), (TL_MF,"Middle",w//2),
                              (TL_LF,"Last",bx1-22)]:
            self.create_oval(px-4, by0+2, px+4, by1-2, fill=col, outline="")
        self.create_text(bx0+30, by0+(self.BAR_H//2),
                         text="First", fill=TXT2, font=("Segoe UI",7), anchor="w")
        self.create_text(bx1-30, by0+(self.BAR_H//2),
                         text="Last",  fill=TXT2, font=("Segoe UI",7), anchor="e")


# ─────────────────────────────────────────────────────────────────────────────
# Middle-frame row widget
# ─────────────────────────────────────────────────────────────────────────────

class MFrameRow(tk.Frame):
    def __init__(self, parent, index: int, mframe: MFrame, ins_mode: str,
                 on_remove, on_change, bg=BG, **kw):
        super().__init__(parent, bg=bg, **kw)
        self._mf       = mframe
        self._on_remove = on_remove
        self._on_change = on_change

        # Index label
        _label(self, f"MF{index}", size=8, fg=YEL, bg=bg).pack(side="left", padx=(0,6))

        # Label entry
        self._lbl_var = tk.StringVar(value=mframe.label)
        e_lbl = _entry(self, textvariable=self._lbl_var, width=12)
        e_lbl.pack(side="left", padx=(0,4))
        e_lbl.bind("<FocusOut>", self._changed)
        _label(self, "label", size=8, fg=DIM, bg=bg).pack(side="left", padx=(0,10))

        # Position entry
        unit = "sec" if ins_mode == "seconds" else "frame"
        _label(self, unit, size=8, fg=DIM, bg=bg).pack(side="left")
        self._pos_var = tk.StringVar(value=str(mframe.pos))
        e_pos = _entry(self, textvariable=self._pos_var, width=8)
        e_pos.pack(side="left", padx=(2,8))
        e_pos.bind("<FocusOut>", self._changed)

        # Image path
        self._path_var = tk.StringVar(value=mframe.path)
        e_path = _entry(self, textvariable=self._path_var, width=24)
        e_path.pack(side="left", padx=(0,4))
        e_path.bind("<FocusOut>", self._changed)
        _btn(self, "…", self._browse, padx=4, pady=2).pack(side="left", padx=(0,8))

        # Remove
        _btn(self, "✕", on_remove, bg=BG3, fg=RED, padx=6, pady=2).pack(side="left")

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select image", filetypes=[("Images","*.png *.jpg *.jpeg *.webp"),
                                              ("All","*.*")])
        if path:
            self._path_var.set(path)
            self._changed()

    def _changed(self, _=None):
        self._mf.label = self._lbl_var.get()
        self._mf.path  = self._path_var.get()
        try:
            self._mf.pos = float(self._pos_var.get())
        except ValueError:
            pass
        self._on_change()

    def read_into(self, mf: MFrame):
        mf.label = self._lbl_var.get()
        mf.path  = self._path_var.get()
        try:   mf.pos = float(self._pos_var.get())
        except ValueError: pass


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Scene Assist  —  LTX-Video 2.3")
        self.geometry("1340x860")
        self.minsize(900, 600)
        self.configure(bg=BG)

        self.shots:        List[Shot] = [Shot()]
        self.current:      int        = 0
        self.project_path: str        = None
        self._mframe_rows: list       = []   # MFrameRow instances

        self._build_menu()
        self._build_layout()
        self._load_shot(0)

    # ── menu ──────────────────────────────────────────────────────────────────
    def _build_menu(self):
        mb = tk.Menu(self, bg=BG2, fg=TXT, activebackground=ACC,
                     activeforeground=TXT, relief="flat", tearoff=False)
        fm = tk.Menu(mb, bg=BG2, fg=TXT, activebackground=ACC,
                     activeforeground=TXT, tearoff=False)
        fm.add_command(label="New Project",       command=self._new_project)
        fm.add_command(label="Open Project…",     command=self._open_project)
        fm.add_command(label="Save Project",      command=self._save_project)
        fm.add_command(label="Save Project As…",  command=self._save_project_as)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.quit)
        mb.add_cascade(label="File", menu=fm)
        self.config(menu=mb)

    # ── two-panel layout ──────────────────────────────────────────────────────
    def _build_layout(self):
        # ── Left: shot list ──────────────────────────────────────────────────
        left = tk.Frame(self, bg=BG2, width=210)
        left.pack(side="left", fill="y", padx=(8, 0), pady=8)
        left.pack_propagate(False)
        self._build_shot_list(left)

        # ── Right: scrollable editor ─────────────────────────────────────────
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self._scroll_cv = tk.Canvas(right, bg=BG, highlightthickness=0)
        vsb = tk.Scrollbar(right, orient="vertical", command=self._scroll_cv.yview)
        self._scroll_cv.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._scroll_cv.pack(side="left", fill="both", expand=True)

        self._editor = tk.Frame(self._scroll_cv, bg=BG)
        self._editor_win = self._scroll_cv.create_window(
            (0, 0), window=self._editor, anchor="nw")

        self._editor.bind("<Configure>",
            lambda e: self._scroll_cv.configure(
                scrollregion=self._scroll_cv.bbox("all")))
        self._scroll_cv.bind("<Configure>",
            lambda e: self._scroll_cv.itemconfig(self._editor_win, width=e.width))
        self._scroll_cv.bind_all("<MouseWheel>",
            lambda e: self._scroll_cv.yview_scroll(-1*(e.delta//120), "units"))

        self._build_editor(self._editor)

    # ── shot list panel ───────────────────────────────────────────────────────
    def _build_shot_list(self, p):
        _label(p, "SHOTS", size=8, bold=True, fg=DIM, bg=BG2
               ).pack(anchor="w", padx=10, pady=(10, 4))

        lb_f = tk.Frame(p, bg=BG2)
        lb_f.pack(fill="both", expand=True, padx=6)

        self._listbox = tk.Listbox(
            lb_f, bg=BG3, fg=TXT, selectbackground=ACC, selectforeground=TXT,
            activestyle="none", font=("Segoe UI", 10), bd=0, relief="flat",
            highlightthickness=1, highlightbackground=BDR, highlightcolor=ACC)
        self._listbox.pack(fill="both", expand=True)
        self._listbox.bind("<<ListboxSelect>>", self._on_shot_select)

        btn_f = tk.Frame(p, bg=BG2)
        btn_f.pack(fill="x", padx=6, pady=(4, 8))
        for txt, cmd in [("+ Add", self._shot_add), ("Dup", self._shot_dup),
                          ("↑", self._shot_up),    ("↓",  self._shot_dn),
                          ("✕", self._shot_del)]:
            _btn(btn_f, txt, cmd, padx=5, pady=3).pack(side="left", padx=2)

        self._refresh_listbox()

    def _refresh_listbox(self):
        self._listbox.delete(0, "end")
        for i, s in enumerate(self.shots):
            self._listbox.insert("end", f"  {i+1}. {s.name}")
        if 0 <= self.current < len(self.shots):
            self._listbox.selection_set(self.current)
            self._listbox.see(self.current)

    def _on_shot_select(self, _=None):
        sel = self._listbox.curselection()
        if sel and sel[0] != self.current:
            self._save_current()
            self.current = sel[0]
            self._load_shot(self.current)

    def _shot_add(self):
        self._save_current()
        n = len(self.shots) + 1
        self.shots.append(Shot(name=f"Shot {n}"))
        self.current = len(self.shots) - 1
        self._refresh_listbox()
        self._load_shot(self.current)

    def _shot_dup(self):
        self._save_current()
        if not self.shots: return
        d = deepcopy(self.shots[self.current].to_dict())
        dup = Shot.from_dict(d)
        dup.name += " (copy)"
        self.shots.insert(self.current + 1, dup)
        self.current += 1
        self._refresh_listbox()
        self._load_shot(self.current)

    def _shot_del(self):
        if len(self.shots) <= 1:
            messagebox.showinfo("Scene Assist", "Cannot delete the last shot.")
            return
        if messagebox.askyesno("Delete Shot",
                               f"Delete '{self.shots[self.current].name}'?"):
            del self.shots[self.current]
            self.current = min(self.current, len(self.shots) - 1)
            self._refresh_listbox()
            self._load_shot(self.current)

    def _shot_up(self):
        if self.current > 0:
            self._save_current()
            i = self.current
            self.shots[i], self.shots[i-1] = self.shots[i-1], self.shots[i]
            self.current -= 1
            self._refresh_listbox()
            self._load_shot(self.current)

    def _shot_dn(self):
        if self.current < len(self.shots) - 1:
            self._save_current()
            i = self.current
            self.shots[i], self.shots[i+1] = self.shots[i+1], self.shots[i]
            self.current += 1
            self._refresh_listbox()
            self._load_shot(self.current)

    # ── editor panel ──────────────────────────────────────────────────────────
    def _build_editor(self, p):
        pad = dict(padx=14)

        # ── Shot name + notes ────────────────────────────────────────────────
        row0 = tk.Frame(p, bg=BG)
        row0.pack(fill="x", **pad, pady=(8, 0))

        _label(row0, "Shot name:", bg=BG, fg=TXT2).pack(side="left")
        self._v_name = tk.StringVar()
        e_name = _entry(row0, textvariable=self._v_name, width=28)
        e_name.pack(side="left", padx=(6, 20))
        e_name.bind("<FocusOut>", lambda _: self._name_changed())

        _label(row0, "Notes:", bg=BG, fg=TXT2).pack(side="left")
        self._v_notes = tk.StringVar()
        _entry(row0, textvariable=self._v_notes, width=40).pack(side="left", padx=6)

        # ── KEYFRAMES ────────────────────────────────────────────────────────
        kf = _section(p, "KEYFRAMES", bg=BG)

        # First frame
        ff_row = tk.Frame(kf, bg=BG)
        ff_row.pack(fill="x", padx=14, pady=2)
        _label(ff_row, "First frame:", fg=TL_FF, bg=BG, width=12, anchor="w"
               ).pack(side="left")
        self._v_first = tk.StringVar()
        _entry(ff_row, textvariable=self._v_first, width=44).pack(side="left", padx=(0,4))
        _btn(ff_row, "Browse…", self._browse_first, padx=6, pady=3).pack(side="left")

        # Middle frames container
        self._mid_container = tk.Frame(kf, bg=BG)
        self._mid_container.pack(fill="x", padx=14, pady=2)

        _btn(kf, "+ Add Middle Frame", self._add_middle, bg=BG3, padx=8, pady=4
             ).pack(anchor="w", padx=14, pady=(2, 4))

        # Last frame
        lf_row = tk.Frame(kf, bg=BG)
        lf_row.pack(fill="x", padx=14, pady=2)
        _label(lf_row, "Last frame:", fg=TL_LF, bg=BG, width=12, anchor="w"
               ).pack(side="left")
        self._v_last = tk.StringVar()
        _entry(lf_row, textvariable=self._v_last, width=44).pack(side="left", padx=(0,4))
        _btn(lf_row, "Browse…", self._browse_last, padx=6, pady=3).pack(side="left")

        # ── TIMELINE ─────────────────────────────────────────────────────────
        tl_sec = _section(p, "TIMELINE", bg=BG)
        self._timeline = TimelineCanvas(tl_sec)
        self._timeline.pack(fill="x", padx=14, pady=(2, 6))
        self.after(50, self._redraw_timeline)

        # ── PROMPTS ───────────────────────────────────────────────────────────
        pr = _section(p, "PROMPTS", bg=BG)
        _label(pr, "Positive", bold=True, fg=GRN, bg=BG).pack(anchor="w", padx=14)
        self._txt_pos = _text(pr, height=4)
        self._txt_pos.pack(fill="x", padx=14, pady=(2, 8))
        _label(pr, "Negative", bold=True, fg=RED, bg=BG).pack(anchor="w", padx=14)
        self._txt_neg = _text(pr, height=3)
        self._txt_neg.pack(fill="x", padx=14, pady=(2, 4))

        # ── LTX SETTINGS ──────────────────────────────────────────────────────
        ls = _section(p, "LTX SETTINGS", bg=BG)

        row1 = tk.Frame(ls, bg=BG)
        row1.pack(fill="x", padx=14, pady=4)

        # Resolution
        _label(row1, "Resolution:", bg=BG, fg=TXT2).pack(side="left")
        self._v_res = tk.StringVar()
        cb_res = _combobox(row1, RESOLUTIONS, textvariable=self._v_res, width=14)
        cb_res.pack(side="left", padx=(4, 20))
        cb_res.bind("<<ComboboxSelected>>", lambda _: self._settings_changed())

        # Frames
        _label(row1, "Frames:", bg=BG, fg=TXT2).pack(side="left")
        self._v_frames = tk.StringVar()
        cb_fr = _combobox(row1, [str(f) for f in VALID_FRAMES],
                          textvariable=self._v_frames, width=6)
        cb_fr.pack(side="left", padx=(4, 6))
        cb_fr.bind("<<ComboboxSelected>>", lambda _: self._settings_changed())

        # FPS
        _label(row1, "FPS:", bg=BG, fg=TXT2).pack(side="left")
        self._v_fps = tk.StringVar()
        cb_fps = _combobox(row1, [str(f) for f in FPS_OPTS],
                           textvariable=self._v_fps, width=5)
        cb_fps.pack(side="left", padx=(4, 6))
        cb_fps.bind("<<ComboboxSelected>>", lambda _: self._settings_changed())

        # Duration display (read-only)
        self._dur_lbl = _label(row1, "= ?", fg=YEL, bg=BG)
        self._dur_lbl.pack(side="left", padx=(4, 20))

        # Insert mode
        _label(row1, "Insert mode:", bg=BG, fg=TXT2).pack(side="left")
        self._v_ins = tk.StringVar(value="frames")
        for val in ("frames", "seconds"):
            tk.Radiobutton(row1, text=val, variable=self._v_ins, value=val,
                           bg=BG, fg=TXT, selectcolor=ACC, activebackground=BG,
                           activeforeground=AC2, font=("Segoe UI", 9),
                           command=self._ins_mode_changed
                           ).pack(side="left", padx=3)

        row2 = tk.Frame(ls, bg=BG)
        row2.pack(fill="x", padx=14, pady=(0, 4))

        # Steps
        _label(row2, "Steps:", bg=BG, fg=TXT2).pack(side="left")
        self._v_steps = tk.StringVar()
        _entry(row2, textvariable=self._v_steps, width=5).pack(side="left", padx=(4,16))

        # CFG
        _label(row2, "CFG:", bg=BG, fg=TXT2).pack(side="left")
        self._v_cfg = tk.StringVar()
        _entry(row2, textvariable=self._v_cfg, width=6).pack(side="left", padx=(4,16))

        # Seed
        _label(row2, "Seed:", bg=BG, fg=TXT2).pack(side="left")
        self._v_seed = tk.StringVar()
        _entry(row2, textvariable=self._v_seed, width=14).pack(side="left", padx=(4,20))
        _btn(row2, "Randomize", lambda: self._v_seed.set("-1"),
             padx=6, pady=3).pack(side="left")

        # Quality presets
        row3 = tk.Frame(ls, bg=BG)
        row3.pack(fill="x", padx=14, pady=(0, 8))
        _label(row3, "Preset:", bg=BG, fg=TXT2).pack(side="left", padx=(0, 6))
        for name, (st, cg) in PRESETS.items():
            _btn(row3, name, lambda s=st, c=cg: self._apply_preset(s, c),
                 bg=BG3, padx=7, pady=3).pack(side="left", padx=3)

        # ── TIMING ADVISOR ────────────────────────────────────────────────────
        adv = _section(p, "TIMING ADVISOR", bg=BG)

        dial_row = tk.Frame(adv, bg=BG)
        dial_row.pack(fill="x", padx=14, pady=4)
        _label(dial_row, "Extra dialogue / notes:", bg=BG, fg=TXT2).pack(side="left")
        self._v_dialogue = tk.StringVar()
        _entry(dial_row, textvariable=self._v_dialogue, width=36
               ).pack(side="left", padx=(6, 8))
        _btn(dial_row, "Recommend Settings", self._do_recommend,
             bg=ACC, fg=TXT, padx=8, pady=4).pack(side="left")

        self._adv_text = _text(adv, height=6, bg=BG2)
        self._adv_text.pack(fill="x", padx=14, pady=(4, 4))
        self._adv_text.config(state="disabled")

        # ── Copy settings button ──────────────────────────────────────────────
        copy_row = tk.Frame(p, bg=BG)
        copy_row.pack(fill="x", padx=14, pady=(8, 16))
        _btn(copy_row, "📋  Copy Settings to Clipboard", self._copy_settings,
             bg=AC2, fg=BG, padx=12, pady=6).pack(side="left")

    # ── browse helpers ────────────────────────────────────────────────────────
    def _browse_first(self):
        p = filedialog.askopenfilename(title="First frame image",
            filetypes=[("Images","*.png *.jpg *.jpeg *.webp"),("All","*.*")])
        if p: self._v_first.set(p)

    def _browse_last(self):
        p = filedialog.askopenfilename(title="Last frame image",
            filetypes=[("Images","*.png *.jpg *.jpeg *.webp"),("All","*.*")])
        if p: self._v_last.set(p)

    # ── middle frame management ───────────────────────────────────────────────
    def _rebuild_mid_rows(self, shot: Shot):
        for w in self._mframe_rows:
            w.destroy()
        self._mframe_rows.clear()

        for i, mf in enumerate(shot.middles, 1):
            row = MFrameRow(
                self._mid_container, index=i, mframe=mf,
                ins_mode=shot.ins_mode,
                on_remove=lambda idx=i-1: self._remove_middle(idx),
                on_change=self._redraw_timeline,
                bg=BG,
            )
            row.pack(fill="x", pady=2)
            self._mframe_rows.append(row)

    def _add_middle(self):
        self._save_current()
        shot = self.shots[self.current]
        mf = MFrame()
        # default position: midpoint of total duration
        if shot.ins_mode == "seconds":
            mf.pos = round(shot.dur / 2, 2)
        else:
            mf.pos = shot.frames // 2
        shot.middles.append(mf)
        self._rebuild_mid_rows(shot)
        self._redraw_timeline()

    def _remove_middle(self, idx: int):
        shot = self.shots[self.current]
        if 0 <= idx < len(shot.middles):
            del shot.middles[idx]
            self._rebuild_mid_rows(shot)
            self._redraw_timeline()

    # ── settings change handlers ──────────────────────────────────────────────
    def _settings_changed(self):
        try:
            fr  = int(self._v_frames.get())
            fps = int(self._v_fps.get())
            self._dur_lbl.config(text=f"= {round(fr/fps, 2)}s")
        except (ValueError, ZeroDivisionError):
            pass
        self._redraw_timeline()

    def _ins_mode_changed(self):
        """Rebuild middle frame rows when insert-mode changes (unit labels change)."""
        self._save_current()
        shot = self.shots[self.current]
        self._rebuild_mid_rows(shot)
        self._redraw_timeline()

    def _name_changed(self):
        if 0 <= self.current < len(self.shots):
            self.shots[self.current].name = self._v_name.get()
            self._refresh_listbox()

    def _apply_preset(self, steps: int, cfg: float):
        self._v_steps.set(str(steps))
        self._v_cfg.set(str(cfg))

    def _redraw_timeline(self):
        self._save_current()
        shot = self.shots[self.current] if self.shots else Shot()
        self._timeline.draw(shot)

    # ── advisor ───────────────────────────────────────────────────────────────
    def _do_recommend(self):
        self._save_current()
        shot = self.shots[self.current]
        rec  = recommend_shot(shot.positive, self._v_dialogue.get())

        # Apply recommended values
        self._v_frames.set(str(rec["frames"]))
        self._v_fps.set(str(rec["fps"]))
        self._v_steps.set(str(rec["steps"]))
        self._v_cfg.set(str(rec["cfg"]))
        self._dur_lbl.config(text=f"= {rec['dur']}s")

        # Suggest middle frames
        if rec["mid_seconds"]:
            shot.middles.clear()
            for s in rec["mid_seconds"]:
                if shot.ins_mode == "seconds":
                    pos = s
                else:
                    pos = round(s * rec["fps"])
                shot.middles.append(MFrame(pos=pos,
                                           label=f"~{s}s"))
            self._rebuild_mid_rows(shot)

        # Show advice
        txt = "\n".join(f"  {a}" for a in rec["advice"])
        self._adv_text.config(state="normal")
        self._adv_text.delete("1.0", "end")
        self._adv_text.insert("end", txt)
        self._adv_text.config(state="disabled")

        self._save_current()
        self._redraw_timeline()

    # ── copy to clipboard ─────────────────────────────────────────────────────
    def _copy_settings(self):
        self._save_current()
        shot = self.shots[self.current]
        text = fmt_settings(shot)
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Scene Assist",
                            "Settings copied to clipboard.\n\nPaste into a text editor or "
                            "keep handy while entering values in ComfyUI.")

    # ── load / save current shot ──────────────────────────────────────────────
    def _load_shot(self, idx: int):
        shot = self.shots[idx]

        self._v_name.set(shot.name)
        self._v_notes.set(shot.notes)
        self._v_first.set(shot.first)
        self._v_last.set(shot.last)

        self._txt_pos.delete("1.0", "end")
        self._txt_pos.insert("end", shot.positive)
        self._txt_neg.delete("1.0", "end")
        self._txt_neg.insert("end", shot.negative)

        self._v_res.set(shot.resolution)
        self._v_frames.set(str(shot.frames))
        self._v_fps.set(str(shot.fps))
        self._v_steps.set(str(shot.steps))
        self._v_cfg.set(str(shot.cfg))
        self._v_seed.set(str(shot.seed))
        self._v_ins.set(shot.ins_mode)
        self._dur_lbl.config(text=f"= {shot.dur}s")

        self._rebuild_mid_rows(shot)
        self._adv_text.config(state="normal")
        self._adv_text.delete("1.0", "end")
        self._adv_text.config(state="disabled")

        self.after(80, self._redraw_timeline)

    def _save_current(self):
        if not (0 <= self.current < len(self.shots)):
            return
        shot = self.shots[self.current]

        shot.name       = self._v_name.get()
        shot.notes      = self._v_notes.get()
        shot.first      = self._v_first.get()
        shot.last       = self._v_last.get()
        shot.positive   = self._txt_pos.get("1.0", "end-1c")
        shot.negative   = self._txt_neg.get("1.0", "end-1c")
        shot.resolution = self._v_res.get()
        shot.ins_mode   = self._v_ins.get()

        try: shot.frames = int(self._v_frames.get())
        except ValueError: pass
        try: shot.fps = int(self._v_fps.get())
        except ValueError: pass
        try: shot.steps = int(self._v_steps.get())
        except ValueError: pass
        try: shot.cfg = float(self._v_cfg.get())
        except ValueError: pass
        try: shot.seed = int(self._v_seed.get())
        except ValueError: shot.seed = -1

        # Read middle-frame rows back into shot
        for row, mf in zip(self._mframe_rows, shot.middles):
            row.read_into(mf)

    # ── project file I/O ──────────────────────────────────────────────────────
    def _new_project(self):
        if messagebox.askyesno("New Project", "Discard current project?"):
            self.shots        = [Shot()]
            self.current      = 0
            self.project_path = None
            self.title("Scene Assist  —  LTX-Video 2.3")
            self._refresh_listbox()
            self._load_shot(0)

    def _open_project(self):
        path = filedialog.askopenfilename(
            title="Open Scene Assist project",
            filetypes=[("JSON project","*.json"),("All","*.*")])
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.shots = [Shot.from_dict(d) for d in data.get("shots", [])]
            if not self.shots: self.shots = [Shot()]
            self.current      = 0
            self.project_path = path
            self.title(f"Scene Assist  —  {os.path.basename(path)}")
            self._refresh_listbox()
            self._load_shot(0)
        except Exception as e:
            messagebox.showerror("Open Project", f"Failed to open:\n{e}")

    def _save_project(self):
        if self.project_path:
            self._write_project(self.project_path)
        else:
            self._save_project_as()

    def _save_project_as(self):
        path = filedialog.asksaveasfilename(
            title="Save Scene Assist project",
            defaultextension=".json",
            filetypes=[("JSON project","*.json"),("All","*.*")])
        if not path: return
        self.project_path = path
        self.title(f"Scene Assist  —  {os.path.basename(path)}")
        self._write_project(path)

    def _write_project(self, path: str):
        self._save_current()
        data = {"shots": [s.to_dict() for s in self.shots]}
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            messagebox.showerror("Save Project", f"Failed to save:\n{e}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
