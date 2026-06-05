#!/usr/bin/env python3
r"""
ComfyUI Model Maintenance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Scan, deduplicate, audit, and track usage of ComfyUI model files.

Save to:  C:\py\model_maintenance.py
Run:      python C:\py\model_maintenance.py

Requirements:
  pip install pyyaml
"""

import datetime
import hashlib
import json
import os
import re
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

try:
    import yaml
    YAML_OK = True
except ImportError:
    YAML_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MB = 1_048_576
GB = 1_073_741_824

MODEL_EXTS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".ggml", ".pkl"}

# (min_bytes, max_bytes)  —  None = no limit
SIZE_RULES: dict[str, tuple] = {
    "checkpoints":    (300 * MB, None),
    "loras":          (1 * MB,   2_000 * MB),
    "vae":            (50 * MB,  2_500 * MB),
    "unet":           (300 * MB, None),
    "clip":           (100 * MB, None),
    "text_encoders":  (100 * MB, None),
    "controlnet":     (30 * MB,  3_000 * MB),
    "upscale_models": (1 * MB,   300 * MB),
    "embeddings":     (0,        15 * MB),
    "hypernetworks":  (1 * MB,   500 * MB),
}

# (name_fragment, suggested_folder_type)
NAME_HINTS: list[tuple[str, str]] = [
    ("_lora",       "loras"),  ("lora_",       "loras"),
    ("-lora",       "loras"),  ("_locon",       "loras"),
    ("_lycoris",    "loras"),  ("_lokr",        "loras"),
    ("_vae",        "vae"),    ("vae_",         "vae"),
    ("-vae",        "vae"),
    ("controlnet",  "controlnet"), ("control_v", "controlnet"),
    ("esrgan",      "upscale_models"), ("realesrgan", "upscale_models"),
    ("upscaler",    "upscale_models"), ("4x_",    "upscale_models"),
    ("_4x",         "upscale_models"), ("2x_",    "upscale_models"),
    ("8x_",         "upscale_models"),
    ("embedding",   "embeddings"), ("textual_inv", "embeddings"),
    ("ip-adapter",  "ipadapter"),  ("ipadapter",   "ipadapter"),
    ("ip_adapter",  "ipadapter"),
]

# Loader node → input field names that hold model filenames
LOADER_FIELDS: dict[str, list[str]] = {
    "CheckpointLoaderSimple":   ["ckpt_name"],
    "CheckpointLoader":         ["ckpt_name"],
    "unCLIPCheckpointLoader":   ["ckpt_name"],
    "UNETLoader":               ["unet_name"],
    "CLIPLoader":               ["clip_name"],
    "DualCLIPLoader":           ["clip_name1", "clip_name2"],
    "TripleCLIPLoader":         ["clip_name1", "clip_name2", "clip_name3"],
    "VAELoader":                ["vae_name"],
    "LoraLoader":               ["lora_name"],
    "LoraLoaderModelOnly":      ["lora_name"],
    "ControlNetLoader":         ["control_net_name"],
    "UpscaleModelLoader":       ["model_name"],
    "CLIPVisionLoader":         ["clip_name"],
    "StyleModelLoader":         ["style_model_name"],
    "HypernetworkLoader":       ["hypernetwork_name"],
    "SAMModelLoader":           ["model_name"],
    "GroundingDinoModelLoader": ["model_name"],
    "IPAdapterModelLoader":     ["ipadapter_file"],
    "InstantIDModelLoader":     ["instantid_file"],
    "PulidModelLoader":         ["pulid_file"],
    "PhotoMakerLoader":         ["photomaker_model_path"],
    "LTXVLoader":               ["ckpt_name"],
    "WanVideoLoader":           ["model"],
    "HunyuanVideoLoader":       ["model"],
    "FaceRestoreWithModel":     ["model_name"],
    "ReActorFaceSwap":          ["model"],
    "aistudynow_QwenVLModel":   ["model_name"],
    "QwenImageEditLoader":      ["model_name"],
}

CONFIG_FILE  = r"C:\py\model_maintenance_config.json"
MODEL_EXT_RE = re.compile(r"\.(safetensors|ckpt|pt|pth|bin|gguf|ggml|pkl)$", re.IGNORECASE)

DEFAULT_YAMLS = [
    r"C:\Comfy\ComfyUI\extra_model_paths.yaml",
    r"H:\Comfy\ComfyUI\extra_model_paths.yaml"
]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelFile:
    path:         str
    filename:     str
    folder_type:  str
    install_label: str
    size:         int
    mtime:        float
    sha256:       Optional[str] = None

    @property
    def size_str(self) -> str:
        if self.size >= GB:  return f"{self.size/GB:.2f} GB"
        if self.size >= MB:  return f"{self.size/MB:.1f} MB"
        return f"{self.size/1024:.1f} KB"

    @property
    def mtime_str(self) -> str:
        if not self.mtime: return ""
        return datetime.datetime.fromtimestamp(self.mtime).strftime("%m/%d/%y %H:%M")


@dataclass
class MisplacedModel:
    model:          ModelFile
    reason:         str
    suggested_type: Optional[str]
    confidence:     str  # "High" | "Medium"


# ─────────────────────────────────────────────────────────────────────────────
# YAML parsing
# ─────────────────────────────────────────────────────────────────────────────

def _resolve(raw: str, base: str = "") -> str:
    raw = raw.strip()
    if not raw: return ""
    p = Path(raw)
    if p.is_absolute(): return str(p)
    if base: return str(Path(base) / p)
    return raw


def parse_yaml_paths(yaml_path: str) -> dict[str, list[str]]:
    """Return {folder_type: [absolute_existing_paths]} from an extra_model_paths.yaml."""
    if not os.path.isfile(yaml_path):
        return {}
    try:
        with open(yaml_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    except Exception:
        return {}

    result: dict[str, list[str]] = {}

    if YAML_OK:
        try:
            data = yaml.safe_load(raw) or {}
        except Exception:
            data = {}
        for section_name, section in data.items():
            if not isinstance(section, dict): continue
            base = str(section.get("base_path", "")).strip()
            for key, val in section.items():
                if key == "base_path": continue
                for line in str(val).splitlines():
                    p = _resolve(line.strip(), base)
                    if p and os.path.isdir(p):
                        result.setdefault(key, []).append(p)
    else:
        # Minimal fallback parser
        base = ""
        cur_key = None
        in_block = False
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"): continue
            if stripped.startswith("base_path:"):
                base = stripped.split(":", 1)[1].strip().strip("\"'")
            elif ":" in stripped and not stripped.endswith("|"):
                key, _, val = stripped.partition(":")
                val = val.strip().strip("\"'")
                if val:
                    p = _resolve(val, base)
                    if p and os.path.isdir(p):
                        result.setdefault(key.strip(), []).append(p)
                cur_key = key.strip(); in_block = False
            elif stripped.endswith("|"):
                cur_key = stripped.rstrip("|").strip().rstrip(":")
                in_block = True
            elif in_block and cur_key and stripped:
                p = _resolve(stripped, base)
                if p and os.path.isdir(p):
                    result.setdefault(cur_key, []).append(p)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Scanning
# ─────────────────────────────────────────────────────────────────────────────

def scan_models(yaml_entries: list[dict], cb=None) -> list[ModelFile]:
    """Scan all enabled YAML entries. cb(msg) for progress updates."""
    all_models: list[ModelFile] = []
    seen: set[str] = set()

    for entry in yaml_entries:
        if not entry.get("enabled", True): continue
        yp = entry["path"]
        label = Path(yp).parent.name

        type_paths = parse_yaml_paths(yp)
        if cb: cb(f"[{label}]  {len(type_paths)} model types found in YAML")

        for folder_type, dirs in type_paths.items():
            for root in dirs:
                if not os.path.isdir(root): continue
                for dirpath, dirnames, files in os.walk(root):
                    depth = os.path.relpath(dirpath, root).count(os.sep)
                    if depth > 3: dirnames.clear(); continue
                    for fn in files:
                        if Path(fn).suffix.lower() not in MODEL_EXTS: continue
                        full = os.path.normpath(os.path.join(dirpath, fn))
                        if full in seen: continue
                        seen.add(full)
                        try:
                            st = os.stat(full)
                        except Exception:
                            continue
                        all_models.append(ModelFile(
                            path=full, filename=fn,
                            folder_type=folder_type, install_label=label,
                            size=st.st_size, mtime=st.st_mtime,
                        ))
                        if cb and len(all_models) % 100 == 0:
                            cb(f"Found {len(all_models):,} model files…")
    return all_models


# ─────────────────────────────────────────────────────────────────────────────
# Analysis: Duplicates
# ─────────────────────────────────────────────────────────────────────────────

def find_name_duplicates(models: list[ModelFile]) -> list[list[ModelFile]]:
    groups: dict[str, list[ModelFile]] = {}
    for m in models:
        groups.setdefault(m.filename.lower(), []).append(m)
    return sorted([g for g in groups.values() if len(g) > 1],
                  key=lambda g: sum(x.size for x in g), reverse=True)


def hash_file(path: str) -> Optional[str]:
    """SHA-256, partial for large files (first+last 4 MB + size)."""
    try:
        size = os.path.getsize(path)
        h = hashlib.sha256()
        chunk = 4 * MB
        with open(path, "rb") as f:
            if size > chunk * 2:
                h.update(f.read(chunk))
                f.seek(-chunk, 2)
                h.update(f.read())
                h.update(str(size).encode())
            else:
                while data := f.read(8 * MB):
                    h.update(data)
        return h.hexdigest()
    except Exception:
        return None


def find_hash_duplicates(models: list[ModelFile], cb=None) -> list[list[ModelFile]]:
    groups: dict[str, list[ModelFile]] = {}
    for i, m in enumerate(models):
        if cb: cb(f"Hashing {i+1}/{len(models)}: {m.filename}")
        h = hash_file(m.path)
        if h:
            m.sha256 = h
            groups.setdefault(h, []).append(m)
    return sorted([g for g in groups.values() if len(g) > 1],
                  key=lambda g: sum(x.size for x in g), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis: Misplaced
# ─────────────────────────────────────────────────────────────────────────────

def find_misplaced(models: list[ModelFile]) -> list[MisplacedModel]:
    results: list[MisplacedModel] = []
    for m in models:
        ft  = m.folder_type.lower()
        fnl = m.filename.lower()
        issue = None

        # Name-hint check (high confidence)
        for frag, suggested in NAME_HINTS:
            if frag in fnl and suggested != ft:
                issue = MisplacedModel(m,
                    f"Filename contains '{frag}' — typical of {suggested}/",
                    suggested, "High")
                break

        # Extension-based check (high confidence)
        # GGUF/GGML are quantised formats loaded by UNETLoader — never CheckpointLoaderSimple
        if not issue:
            ext = Path(m.filename).suffix.lower()
            if ext in (".gguf", ".ggml") and ft == "checkpoints":
                issue = MisplacedModel(m,
                    f"{ext} is a quantised format loaded by UNETLoader, not checkpoints/",
                    "unet", "High")

        # Size check (medium confidence, only if no name hit)
        if not issue and ft in SIZE_RULES:
            lo, hi = SIZE_RULES[ft]
            if lo and m.size < lo:
                sug = "embeddings" if m.size < 15*MB else ("loras" if ft=="checkpoints" else None)
                issue = MisplacedModel(m,
                    f"{m.size_str} is unusually small for {ft}/",
                    sug, "Medium")
            elif hi and m.size > hi:
                sug = "checkpoints" if ft in ("loras","embeddings") else None
                issue = MisplacedModel(m,
                    f"{m.size_str} is unusually large for {ft}/",
                    sug, "Medium")

        if issue:
            results.append(issue)
    # High confidence first, then by size descending
    results.sort(key=lambda x: (0 if x.confidence=="High" else 1, -x.model.size))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Analysis: Unused
# ─────────────────────────────────────────────────────────────────────────────

def extract_refs(json_path: str) -> set[str]:
    try:
        with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
    except Exception:
        return set()
    refs: set[str] = set()

    def add(v):
        if isinstance(v, str) and MODEL_EXT_RE.search(v):
            refs.add(Path(v).name.lower())

    if isinstance(data, dict) and "nodes" in data:
        for n in data.get("nodes", []):
            if isinstance(n, dict):
                for wv in n.get("widgets_values", []):
                    add(wv)

    if isinstance(data, dict):
        for node_id, node in data.items():
            if not isinstance(node, dict): continue
            ntype  = node.get("class_type", "")
            inputs = node.get("inputs", {})
            if isinstance(inputs, dict):
                for field in LOADER_FIELDS.get(ntype, []):
                    add(inputs.get(field, ""))
                for v in inputs.values():
                    add(v)
    return refs


def find_unused(models: list[ModelFile],
                yaml_entries: list[dict],
                cb=None) -> tuple[list[ModelFile], list[str]]:
    wf_dirs: list[str] = []
    for entry in yaml_entries:
        if not entry.get("enabled", True): continue
        root = str(Path(entry["path"]).parent)
        d = os.path.join(root, "user", "default", "workflows")
        if os.path.isdir(d):
            wf_dirs.append(d)

    if not wf_dirs:
        return [], []

    all_refs: set[str] = set()
    count = 0
    for wf_dir in wf_dirs:
        for dp, _, files in os.walk(wf_dir):
            for fn in files:
                if not fn.endswith(".json"): continue
                all_refs |= extract_refs(os.path.join(dp, fn))
                count += 1
                if cb and count % 25 == 0:
                    cb(f"Scanned {count} workflows…")

    if cb: cb(f"Finished: {count} workflows, {len(all_refs)} unique model refs")
    unused = sorted([m for m in models if m.filename.lower() not in all_refs],
                    key=lambda m: m.size, reverse=True)
    return unused, wf_dirs


# ─────────────────────────────────────────────────────────────────────────────
# Palette & style helpers
# ─────────────────────────────────────────────────────────────────────────────

BG, PNL, PNL2 = "#111122", "#0c0c1e", "#080818"
ACC, DIM       = "#6c72ff", "#3a3a66"
FG,  FG2       = "#d4d4f0", "#8888bb"
RED, YEL, GRN  = "#ff7070", "#ffcc55", "#66cc88"
MONO, MONO_B   = ("Consolas", 10), ("Consolas", 10, "bold")


def _styles(s: ttk.Style):
    s.theme_use("clam")
    s.configure("TFrame",            background=BG)
    s.configure("TLabel",            background=BG,  foreground=FG,  font=MONO)
    s.configure("Dim.TLabel",        background=BG,  foreground=FG2, font=("Consolas",9))
    s.configure("Head.TLabel",       background=BG,  foreground=ACC, font=("Consolas",14,"bold"))
    s.configure("Panel.TLabel",      background=PNL, foreground=FG,  font=MONO)
    s.configure("TButton",           background=DIM, foreground=FG,  font=MONO, borderwidth=0)
    s.map("TButton",                 background=[("active","#5050aa")])
    s.configure("Accent.TButton",    background=ACC, foreground="#0a0a1a", font=MONO_B, borderwidth=0)
    s.map("Accent.TButton",          background=[("active","#8a90ff")])
    s.configure("Hash.TButton",      background="#3d2e00", foreground=FG, font=MONO, borderwidth=0)
    s.map("Hash.TButton",            background=[("active","#6a5000")])
    s.configure("Treeview",          background=PNL2, foreground=FG,
                                     fieldbackground=PNL2, font=MONO, rowheight=24)
    s.configure("Treeview.Heading",  background="#1a1a3e", foreground=ACC, font=MONO_B)
    s.map("Treeview",                background=[("selected","#282860")])
    s.configure("TNotebook",         background=BG, tabmargins=[0,0,0,0])
    s.configure("TNotebook.Tab",     background=PNL, foreground=FG2, font=MONO, padding=[12,4])
    s.map("TNotebook.Tab",           background=[("selected","#1a1a3e")],
                                     foreground=[("selected",ACC)])
    s.configure("TSeparator",        background=DIM)
    s.configure("Horizontal.TProgressbar",
                                     background=ACC, troughcolor=PNL2, borderwidth=0)


def _tree(parent, cols: list[tuple], height: int = 200) -> tuple[ttk.Treeview, tk.Frame]:
    """Create a treeview with scrollbars inside a frame. cols = [(id, heading, width, anchor)]."""
    frame = tk.Frame(parent, bg=PNL2)
    ids = [c[0] for c in cols]
    tv = ttk.Treeview(frame, columns=ids, show="headings", selectmode="browse")
    for cid, head, w, anch in cols:
        tv.heading(cid, text=head,
                   command=lambda _c=cid, _tv=tv: _sort_tree(_tv, _c))
        tv.column(cid, width=w, anchor=anch, minwidth=40)
    vsb = ttk.Scrollbar(frame, orient="vertical",   command=tv.yview)
    hsb = ttk.Scrollbar(frame, orient="horizontal", command=tv.xview)
    tv.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    tv.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1); frame.columnconfigure(0, weight=1)
    return tv, frame


def _sort_tree(tv: ttk.Treeview, col: str):
    """Toggle-sort a treeview column."""
    data = [(tv.set(k, col), k) for k in tv.get_children("")]
    reverse = getattr(tv, f"_sort_rev_{col}", False)
    try:
        data.sort(key=lambda x: float(x[0].replace(",","").replace(" GB","").replace(" MB","").replace(" KB","")), reverse=reverse)
    except ValueError:
        data.sort(reverse=reverse)
    for i, (_, k) in enumerate(data):
        tv.move(k, "", i)
    setattr(tv, f"_sort_rev_{col}", not reverse)


def _summary_bar(parent, text_var: tk.StringVar) -> tk.Label:
    lbl = tk.Label(parent, textvariable=text_var, bg=PNL, fg=FG2,
                   font=("Consolas",9), anchor="w", padx=8, pady=4)
    lbl.pack(fill="x", side="bottom")
    return lbl


def _ctx_menu(root: tk.Tk, tv: ttk.Treeview,
              path_col_idx: int, status_var: tk.StringVar,
              delete_cb=None):
    """Right-click context menu for any result treeview."""
    menu = tk.Menu(root, tearoff=0, bg="#1a1a3e", fg=FG,
                   activebackground="#3a3a7e", font=("Consolas",9))

    def _get_path():
        sel = tv.selection()
        if not sel: return None
        vals = tv.item(sel[0])["values"]
        return str(vals[path_col_idx]) if vals else None

    def copy_path():
        p = _get_path()
        if p: root.clipboard_clear(); root.clipboard_append(p); status_var.set(f"Copied: {p}")

    def copy_name():
        p = _get_path()
        if p: n = Path(p).name; root.clipboard_clear(); root.clipboard_append(n); status_var.set(f"Copied: {n}")

    def open_folder():
        import subprocess
        p = _get_path()
        if p and os.path.isfile(p):
            subprocess.Popen(f'explorer /select,"{p}"')

    menu.add_command(label="📋  Copy full path",  command=copy_path)
    menu.add_command(label="📋  Copy filename",   command=copy_name)
    menu.add_separator()
    menu.add_command(label="📁  Open in Explorer",command=open_folder)

    if delete_cb:
        menu.add_separator()
        def do_delete():
            p = _get_path()
            if p: delete_cb(p, tv)
        menu.add_command(label="🗑   Move to Recycle Bin", command=do_delete)

    tv.bind("<Button-3>", lambda e: menu.tk_popup(e.x_root, e.y_root))
    tv.bind("<Double-1>", lambda _: open_folder())


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────

class ModelMaintenance(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("ComfyUI Model Maintenance")
        self.geometry("1280x860")
        self.configure(bg=BG)
        self.minsize(1000, 680)

        self._yaml_entries: list[dict] = self._load_config()
        self._models:   list[ModelFile]     = []
        self._dupes_n:  list[list[ModelFile]] = []
        self._dupes_h:  list[list[ModelFile]] = []
        self._misplaced: list[MisplacedModel] = []
        self._misplaced_paths: set[str]     = set()
        self._unused:   list[ModelFile]     = []
        self._wf_dirs:  list[str]           = []
        self._hash_done = False

        _styles(ttk.Style(self))
        self._build_ui()

        enabled = [e for e in self._yaml_entries if e.get("enabled")]
        if enabled:
            self.after(300, self._start_scan)

    # ── Config ───────────────────────────────────────────────────────────

    def _load_config(self) -> list[dict]:
        try:
            with open(CONFIG_FILE) as f:
                d = json.load(f)
            if d.get("yamls"):
                return d["yamls"]
        except Exception:
            pass
        return [{"path": p, "enabled": os.path.isfile(p)} for p in DEFAULT_YAMLS]

    def _save_config(self):
        try:
            os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
            with open(CONFIG_FILE, "w") as f:
                json.dump({"yamls": self._yaml_entries}, f, indent=2)
        except Exception:
            pass

    # ── UI ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Title
        hdr = ttk.Frame(self); hdr.pack(fill="x", padx=16, pady=(14,4))
        ttk.Label(hdr, text="⚙  COMFYUI MODEL MAINTENANCE", style="Head.TLabel").pack(side="left")
        ttk.Label(hdr, text="Duplicates · Misplaced · Unused",
                  style="Dim.TLabel").pack(side="right", pady=(4,0))
        ttk.Separator(self).pack(fill="x", padx=16, pady=(0,6))

        # Installations panel
        ip = tk.Frame(self, bg=PNL, padx=10, pady=8); ip.pack(fill="x", padx=16)

        hrow = tk.Frame(ip, bg=PNL); hrow.pack(fill="x")
        tk.Label(hrow, text="COMFYUI INSTALLATIONS  (extra_model_paths.yaml)",
                 bg=PNL, fg=ACC, font=("Consolas",9,"bold")).pack(side="left")
        tk.Label(hrow, text="check to include", bg=PNL, fg="#4040aa",
                 font=("Consolas",8)).pack(side="left", padx=(8,0))

        # Scrollable YAML row list
        rout = tk.Frame(ip, bg=PNL2); rout.pack(fill="x", pady=(4,0))
        self._rc = tk.Canvas(rout, bg=PNL2, height=88, highlightthickness=0, bd=0)
        rsb = ttk.Scrollbar(rout, orient="vertical", command=self._rc.yview)
        self._rc.configure(yscrollcommand=rsb.set)
        self._rc.pack(side="left", fill="both", expand=True)
        rsb.pack(side="right", fill="y")
        self._rf = tk.Frame(self._rc, bg=PNL2)
        self._rw = self._rc.create_window((0,0), window=self._rf, anchor="nw")
        self._rf.bind("<Configure>", lambda _: self._rc.configure(scrollregion=self._rc.bbox("all")))
        self._rc.bind("<Configure>", lambda e: self._rc.itemconfig(self._rw, width=e.width))

        br = tk.Frame(ip, bg=PNL); br.pack(fill="x", pady=(6,0))
        ttk.Button(br, text="\uff0b Add YAML", command=self._add_yaml).pack(side="left", padx=(0,6))
        ttk.Button(br, text="\u2611 Enable All",  command=self._enable_all).pack(side="left", padx=(0,4))
        ttk.Button(br, text="\u2610 Disable All", command=self._disable_all).pack(side="left")
        ttk.Button(br, text="\U0001f50d Scan All", style="Accent.TButton",
                   command=self._start_scan).pack(side="right")
        self._scan_sv = tk.StringVar(value="Not scanned yet")
        tk.Label(br, textvariable=self._scan_sv, bg=PNL, fg="#5050aa",
                 font=("Consolas",9)).pack(side="right", padx=(0,12))

        self._rebuild_rows()

        # Progress bar (hidden until scan)
        self._pb = ttk.Progressbar(self, mode="indeterminate",
                                   style="Horizontal.TProgressbar", length=100)
        self._pb.pack(fill="x", padx=16, pady=(4,0))
        self._pb.pack_forget()

        # Status var must exist before tabs call _ctx_menu
        self._status = tk.StringVar(value="Ready — add YAML files and click Scan.")

        # Notebook
        self._nb = ttk.Notebook(self)
        self._nb.pack(fill="both", expand=True, padx=16, pady=8)

        self._build_tab_inventory()
        self._build_tab_duplicates()
        self._build_tab_misplaced()
        self._build_tab_unused()
        self._update_tab_labels()

        # Status bar
        tk.Label(self, textvariable=self._status, bg="#080818", fg="#404080",
                 font=("Consolas",9), anchor="w").pack(fill="x", side="bottom", padx=16, pady=(0,4))

    # ── Installation rows ────────────────────────────────────────────────

    def _rebuild_rows(self):
        for w in self._rf.winfo_children(): w.destroy()
        for i, entry in enumerate(self._yaml_entries):
            self._add_row(i, entry)

    def _add_row(self, idx: int, entry: dict):
        row = tk.Frame(self._rf, bg=PNL2); row.pack(fill="x", pady=1)
        var = tk.BooleanVar(value=entry.get("enabled", True))

        def toggle(i=idx, v=var):
            v.set(not v.get())
            self._yaml_entries[i]["enabled"] = v.get()
            self._save_config()
            cb.configure(text="☑" if v.get() else "☐")
            lbl.configure(fg=_row_color(self._yaml_entries[i]))

        cb = tk.Button(row, text="☑" if var.get() else "☐",
                       command=toggle, bg=PNL2, fg=ACC,
                       font=("Consolas", 11), relief="flat", bd=0, padx=2,
                       activebackground=PNL2, activeforeground="#8a90ff",
                       cursor="hand2")
        cb.pack(side="left", padx=(4,0))

        exists = os.path.isfile(entry["path"])
        lbl = tk.Label(row, text=entry["path"], bg=PNL2,
                       fg=_row_color(entry), font=("Consolas",9), anchor="w")
        lbl.pack(side="left", fill="x", expand=True, padx=(4,8))

        if not exists:
            tk.Label(row, text="[not found]", bg=PNL2, fg="#5a2a2a",
                     font=("Consolas",8)).pack(side="left", padx=(0,4))

        def remove(i=idx):
            self._yaml_entries.pop(i)
            self._save_config(); self._rebuild_rows()

        tk.Button(row, text="\u00d7", command=remove, bg="#1a0808", fg="#884444",
                  font=("Consolas",9,"bold"), relief="flat", bd=0, padx=6, pady=0,
                  activebackground="#2a1010", activeforeground="#ff8888",
                  cursor="hand2").pack(side="right", padx=(0,4))
        row._var = var

    def _add_yaml(self):
        p = filedialog.askopenfilename(
            title="Select extra_model_paths.yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")])
        if not p: return
        p = os.path.normpath(p)
        if any(e["path"] == p for e in self._yaml_entries):
            messagebox.showinfo("Already added", f"Already in list:\n{p}"); return
        self._yaml_entries.append({"path": p, "enabled": True})
        self._save_config(); self._rebuild_rows()
        self._rc.update_idletasks(); self._rc.yview_moveto(1.0)

    def _enable_all(self):
        for e in self._yaml_entries: e["enabled"] = True
        self._save_config(); self._rebuild_rows()

    def _disable_all(self):
        for e in self._yaml_entries: e["enabled"] = False
        self._save_config(); self._rebuild_rows()

    # ── Tab: Inventory ───────────────────────────────────────────────────

    def _build_tab_inventory(self):
        tab = tk.Frame(self._nb, bg=BG); self._nb.add(tab, text="  📦 Inventory  ")

        # Search bar
        sb = tk.Frame(tab, bg=BG); sb.pack(fill="x", padx=8, pady=(6,2))
        tk.Label(sb, text="🔍 Search name:", bg=BG, fg=FG2, font=("Consolas",9)).pack(side="left")
        self._inv_search_var = tk.StringVar(value="")
        self._inv_search_entry = tk.Entry(sb, textvariable=self._inv_search_var,
                                          bg=PNL2, fg=FG, insertbackground=ACC,
                                          font=("Consolas",10), relief="flat", bd=2,
                                          highlightthickness=1, highlightbackground=DIM,
                                          highlightcolor=ACC)
        self._inv_search_entry.pack(side="left", padx=(6,4), fill="x", expand=True)
        self._inv_search_var.trace_add("write", lambda *_: self._on_search_change())
        tk.Button(sb, text="✕", command=lambda: self._inv_search_var.set(""),
                  bg=PNL2, fg=FG2, font=("Consolas",9), relief="flat", bd=0,
                  activebackground=PNL2, activeforeground=RED, cursor="hand2",
                  padx=8).pack(side="left")
        self._inv_search_after_id = None

        # Filter bar
        fb = tk.Frame(tab, bg=BG); fb.pack(fill="x", padx=8, pady=(2,4))
        tk.Label(fb, text="Filter type:", bg=BG, fg=FG2, font=("Consolas",9)).pack(side="left")
        self._inv_type_var = tk.StringVar(value="All")
        self._inv_type_cb  = ttk.Combobox(fb, textvariable=self._inv_type_var,
                                           state="readonly", width=20, font=("Consolas",9))
        self._inv_type_cb.pack(side="left", padx=(6,16))
        self._inv_type_cb.bind("<<ComboboxSelected>>", lambda _: self._filter_inventory())

        tk.Label(fb, text="Filter install:", bg=BG, fg=FG2, font=("Consolas",9)).pack(side="left")
        self._inv_inst_var = tk.StringVar(value="All")
        self._inv_inst_cb  = ttk.Combobox(fb, textvariable=self._inv_inst_var,
                                           state="readonly", width=22, font=("Consolas",9))
        self._inv_inst_cb.pack(side="left", padx=(6,0))
        self._inv_inst_cb.bind("<<ComboboxSelected>>", lambda _: self._filter_inventory())

        # Size + drive filter row
        fb2 = tk.Frame(tab, bg=BG); fb2.pack(fill="x", padx=8, pady=(0,4))
        tk.Label(fb2, text="Min size:", bg=BG, fg=FG2, font=("Consolas",9)).pack(side="left")
        self._inv_size_var = tk.DoubleVar(value=0)
        self._inv_size_lbl = tk.Label(fb2, text="any", bg=BG, fg=ACC,
                                      font=("Consolas",9), width=10, anchor="w")
        self._inv_size_lbl.pack(side="left", padx=(6,0))
        tk.Scale(fb2, variable=self._inv_size_var, from_=0, to=20, resolution=0.5,
                 orient="horizontal", length=280, bg=BG, fg=FG, troughcolor=DIM,
                 highlightthickness=0, bd=0, sliderrelief="raised",
                 activebackground=ACC, showvalue=False,
                 command=lambda _: self._on_size_slider()).pack(side="left", padx=(8,0))
        tk.Label(fb2, text="Drive:", bg=BG, fg=FG2, font=("Consolas",9)).pack(side="left", padx=(20,0))
        self._inv_drive_var = tk.StringVar(value="All")
        self._inv_drive_cb  = ttk.Combobox(fb2, textvariable=self._inv_drive_var,
                                            state="readonly", width=6, font=("Consolas",9))
        self._inv_drive_cb.pack(side="left", padx=(6,0))
        self._inv_drive_cb.bind("<<ComboboxSelected>>", lambda _: self._filter_inventory())

        th = tk.Frame(tab, bg=BG); th.pack(fill="x", padx=8)
        ttk.Button(th, text="⚙ Columns",
                   command=lambda: self._open_col_picker("inv")).pack(side="right", pady=(0,2))

        self._inv_all_cols = [
            ("filename", "Filename",    320, "w"),
            ("type",     "Type",        130, "w"),
            ("size",     "Size",         88, "e"),
            ("modified", "Modified",    130, "center"),
            ("install",  "Install",     130, "w"),
            ("path",     "Full Path",   500, "w"),
        ]
        self._inv_tv, inv_f = _tree(tab, self._inv_all_cols)
        inv_f.pack(fill="both", expand=True, padx=8, pady=(0,0))
        _ctx_menu(self, self._inv_tv, 5, self._status, delete_cb=self._delete_model_from)
        self._inv_vis = self._col_vis_load("inv", [c[0] for c in self._inv_all_cols])
        self._inv_tv["displaycolumns"] = [c[0] for c in self._inv_all_cols if c[0] in self._inv_vis]
        self._inv_tv.bind("<ButtonRelease-1>", lambda _: self._col_widths_save(self._inv_tv, "inv"))
        self.after(120, lambda: self._col_widths_load(self._inv_tv, "inv"))

        self._inv_sum = tk.StringVar(value="")
        _summary_bar(tab, self._inv_sum)

    def _on_size_slider(self):
        v = self._inv_size_var.get()
        self._inv_size_lbl.configure(text="any" if v == 0 else f"≥ {v:.1f} GB")
        self._filter_inventory()

    def _on_search_change(self):
        """Debounce inventory search by 150ms so typing stays smooth."""
        if self._inv_search_after_id is not None:
            try: self.after_cancel(self._inv_search_after_id)
            except Exception: pass
        self._inv_search_after_id = self.after(150, self._filter_inventory)

    def _filter_inventory(self):
        ft = self._inv_type_var.get()
        fi = self._inv_inst_var.get()
        fd = self._inv_drive_var.get()
        min_bytes = int(self._inv_size_var.get() * GB)
        query = self._inv_search_var.get().strip().lower()
        # Space-separated terms must ALL match (AND search)
        terms = [t for t in query.split() if t]
        self._inv_tv.delete(*self._inv_tv.get_children())
        shown, total_size = 0, 0
        for m in self._models:
            if ft != "All" and m.folder_type != ft: continue
            if fi != "All" and m.install_label != fi: continue
            if fd != "All" and Path(m.path).drive.upper() != fd.upper(): continue
            if min_bytes and m.size < min_bytes: continue
            if terms:
                fnl = m.filename.lower()
                if not all(t in fnl for t in terms): continue
            self._inv_tv.insert("", "end", values=(
                m.filename, m.folder_type, m.size_str,
                m.mtime_str, m.install_label, m.path))
            shown += 1; total_size += m.size
        suffix = f"    (search: {query})" if query else ""
        self._inv_sum.set(f"  {shown:,} files    {_fmtsz(total_size)} total{suffix}")

    # ── Tab: Duplicates ──────────────────────────────────────────────────

    def _build_tab_duplicates(self):
        tab = tk.Frame(self._nb, bg=BG); self._nb.add(tab, text="  🔁 Duplicates  ")

        # Top bar
        tb = tk.Frame(tab, bg=BG); tb.pack(fill="x", padx=8, pady=(6,4))
        tk.Label(tb, text="By Name:", bg=BG, fg=FG2, font=("Consolas",9,"bold")).pack(side="left")
        tk.Label(tb, text="same filename in multiple locations",
                 bg=BG, fg=FG2, font=("Consolas",9)).pack(side="left", padx=(6,20))
        self._hash_btn = ttk.Button(tb, text="\U0001f511 Hash Scan (exact content)",
                                    style="Hash.TButton", command=self._start_hash_scan)
        self._hash_btn.pack(side="right")
        tk.Label(tb, text="Scan for true duplicates regardless of name \u2192",
                 bg=BG, fg=FG2, font=("Consolas",9)).pack(side="right", padx=(0,8))

        # Mode toggle
        self._dupe_mode = tk.StringVar(value="name")
        for lbl, val in [("By Name","name"),("By Hash (after scan)","hash")]:
            tk.Radiobutton(tab, text=lbl, variable=self._dupe_mode, value=val,
                           bg=BG, fg="#a0a0cc", selectcolor=BG, activebackground=BG,
                           font=("Consolas",9), command=self._show_dupes).pack(side="top", anchor="w", padx=8)

        th = tk.Frame(tab, bg=BG); th.pack(fill="x", padx=8)
        ttk.Button(th, text="⚙ Columns",
                   command=lambda: self._open_col_picker("dup")).pack(side="right", pady=(0,2))

        self._dup_all_cols = [
            ("group",    "Group",       260, "w"),
            ("copies",   "Copies",       60, "center"),
            ("wasted",   "Wasted",       88, "e"),
            ("install",  "Install",     120, "w"),
            ("type",     "Type",        110, "w"),
            ("size",     "Size",         88, "e"),
            ("modified", "Modified",    130, "center"),
            ("path",     "Full Path",   500, "w"),
        ]
        self._dup_tv, dup_f = _tree(tab, self._dup_all_cols)
        self._dup_tv.tag_configure("group",     foreground=YEL, font=("Consolas",10,"bold"))
        self._dup_tv.tag_configure("hash_group",foreground=RED, font=("Consolas",10,"bold"))
        self._dup_tv.tag_configure("copy",      foreground=FG2)
        self._dup_tv.tag_configure("wrong_loc", foreground="#ffaa44")
        dup_f.pack(fill="both", expand=True, padx=8)
        _ctx_menu(self, self._dup_tv, 7, self._status, delete_cb=self._delete_model_from)
        self._dup_vis = self._col_vis_load("dup", [c[0] for c in self._dup_all_cols])
        self._dup_tv["displaycolumns"] = [c[0] for c in self._dup_all_cols if c[0] in self._dup_vis]
        self._dup_tv.bind("<ButtonRelease-1>", lambda _: self._col_widths_save(self._dup_tv, "dup"))
        self.after(120, lambda: self._col_widths_load(self._dup_tv, "dup"))

        self._dup_sum = tk.StringVar(value="")
        _summary_bar(tab, self._dup_sum)

    def _show_dupes(self):
        mode   = self._dupe_mode.get()
        groups = self._dupes_h if mode == "hash" else self._dupes_n
        tv     = self._dup_tv
        tv.delete(*tv.get_children())

        total_wasted = 0
        for grp in groups:
            wasted = sum(m.size for m in grp) - max(m.size for m in grp)
            total_wasted += wasted
            tag = "hash_group" if mode == "hash" else "group"
            indicator = "≡ HASH MATCH" if mode == "hash" else "⊕ SAME NAME"

            # Detect mixed folder types within the group
            folder_counts: dict[str, int] = {}
            for m in grp:
                folder_counts[m.folder_type] = folder_counts.get(m.folder_type, 0) + 1
            mixed = len(folder_counts) > 1
            # "majority" folder type — copies in other folder types are outliers
            majority_ft = max(folder_counts, key=folder_counts.__getitem__)

            mixed_badge = "  ⚠ mixed folders" if mixed else ""
            parent = tv.insert("", "end", values=(
                f"{indicator}  {grp[0].filename}{mixed_badge}",
                len(grp), _fmtsz(wasted),
                "", "", "", "", ""),
                tags=(tag,), open=True)
            for m in grp:
                wrong    = m.path in self._misplaced_paths
                outlier  = mixed and m.folder_type != majority_ft
                ftype    = f"⚠  {m.folder_type}" if (wrong or outlier) else m.folder_type
                row_tags = ("copy", "wrong_loc") if (wrong or outlier) else ("copy",)
                tv.insert(parent, "end", values=(
                    "", "", "", m.install_label,
                    ftype, m.size_str, m.mtime_str, m.path),
                    tags=row_tags)

        label = "hash duplicates" if mode == "hash" else "filename duplicates"
        self._dup_sum.set(
            f"  {len(groups):,} {label}    "
            f"{_fmtsz(total_wasted)} could be freed"
            + ("" if mode == "name" else "  ← exact content matches")
        )

    # ── Tab: Misplaced ───────────────────────────────────────────────────

    def _build_tab_misplaced(self):
        tab = tk.Frame(self._nb, bg=BG); self._nb.add(tab, text="  📂 Misplaced  ")

        tk.Label(tab, text="Models that appear to be in the wrong folder type "
                           "based on filename patterns and file sizes.",
                 bg=BG, fg=FG2, font=("Consolas",9)).pack(anchor="w", padx=8, pady=(6,4))

        th = tk.Frame(tab, bg=BG); th.pack(fill="x", padx=8)
        ttk.Button(th, text="⚙ Columns",
                   command=lambda: self._open_col_picker("mis")).pack(side="right", pady=(0,2))

        self._mis_all_cols = [
            ("confidence", "Confidence",  90, "center"),
            ("filename",   "Filename",   300, "w"),
            ("cur_type",   "In Folder",  120, "w"),
            ("sug_type",   "Likely Type",120, "w"),
            ("size",       "Size",        88, "e"),
            ("reason",     "Reason",     340, "w"),
            ("install",    "Install",    120, "w"),
            ("path",       "Full Path",  500, "w"),
        ]
        self._mis_tv, mis_f = _tree(tab, self._mis_all_cols)
        self._mis_tv.tag_configure("high",   foreground=RED)
        self._mis_tv.tag_configure("medium", foreground=YEL)
        mis_f.pack(fill="both", expand=True, padx=8)
        _ctx_menu(self, self._mis_tv, 7, self._status, delete_cb=self._delete_model_from)
        self._mis_vis = self._col_vis_load("mis", [c[0] for c in self._mis_all_cols])
        self._mis_tv["displaycolumns"] = [c[0] for c in self._mis_all_cols if c[0] in self._mis_vis]
        self._mis_tv.bind("<ButtonRelease-1>", lambda _: self._col_widths_save(self._mis_tv, "mis"))
        self.after(120, lambda: self._col_widths_load(self._mis_tv, "mis"))

        self._mis_sum = tk.StringVar(value="")
        _summary_bar(tab, self._mis_sum)

    # ── Tab: Unused ──────────────────────────────────────────────────────

    def _build_tab_unused(self):
        tab = tk.Frame(self._nb, bg=BG); self._nb.add(tab, text="  💤 Unused  ")

        self._unused_note = tk.StringVar(value="")
        tk.Label(tab, textvariable=self._unused_note,
                 bg=BG, fg=FG2, font=("Consolas",9), wraplength=900, justify="left"
                 ).pack(anchor="w", padx=8, pady=(6,4))

        th = tk.Frame(tab, bg=BG); th.pack(fill="x", padx=8)
        ttk.Button(th, text="⚙ Columns",
                   command=lambda: self._open_col_picker("unu")).pack(side="right", pady=(0,2))

        self._unu_all_cols = [
            ("filename", "Filename",    320, "w"),
            ("type",     "Type",        120, "w"),
            ("size",     "Size",         88, "e"),
            ("modified", "Modified",    130, "center"),
            ("install",  "Install",     120, "w"),
            ("path",     "Full Path",   500, "w"),
        ]
        self._unu_tv, unu_f = _tree(tab, self._unu_all_cols)
        unu_f.pack(fill="both", expand=True, padx=8)
        _ctx_menu(self, self._unu_tv, 5, self._status, delete_cb=self._delete_model_from)
        self._unu_vis = self._col_vis_load("unu", [c[0] for c in self._unu_all_cols])
        self._unu_tv["displaycolumns"] = [c[0] for c in self._unu_all_cols if c[0] in self._unu_vis]
        self._unu_tv.bind("<ButtonRelease-1>", lambda _: self._col_widths_save(self._unu_tv, "unu"))
        self.after(120, lambda: self._col_widths_load(self._unu_tv, "unu"))

        self._unu_sum = tk.StringVar(value="")
        _summary_bar(tab, self._unu_sum)

    # ── Delete / Recycle Bin ─────────────────────────────────────────────

    @staticmethod
    def _recycle_file(path: str) -> bool:
        """Move a file to the Windows Recycle Bin via SHFileOperationW."""
        try:
            import ctypes
            import ctypes.wintypes as W
            class _SFOS(ctypes.Structure):
                _fields_ = [
                    ("hwnd",        W.HWND),  ("wFunc",       W.UINT),
                    ("pFrom",       ctypes.c_void_p), ("pTo", ctypes.c_void_p),
                    ("fFlags",      W.WORD),  ("fAnyAborted", W.BOOL),
                    ("hNameMappings", ctypes.c_void_p),
                    ("lpTitle",     ctypes.c_void_p),
                ]
            buf = ctypes.create_unicode_buffer(path + "\0")   # double-null terminated
            op  = _SFOS(wFunc=3,                              # FO_DELETE
                        pFrom=ctypes.cast(buf, ctypes.c_void_p).value,
                        fFlags=0x0040 | 0x0010 | 0x0004)      # ALLOWUNDO|NOCONFIRMATION|SILENT
            ret = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
            return ret == 0 and not op.fAnyAborted
        except Exception:
            return False

    def _delete_model_from(self, path: str, tv: ttk.Treeview):
        """Confirm then recycle the file; remove it from the treeview and model list."""
        m = next((x for x in self._models if x.path == path), None)
        size_str = m.size_str if m else ""
        name     = Path(path).name

        if not messagebox.askyesno(
            "Move to Recycle Bin",
            f"Move to Recycle Bin?\n\n{name}   {size_str}\n\n{path}",
            icon="warning", default="no"
        ):
            return

        if not self._recycle_file(path):
            messagebox.showerror("Error", f"Could not move to Recycle Bin:\n{path}")
            return

        # Remove from internal model list and misplaced paths
        self._models          = [x for x in self._models if x.path != path]
        self._misplaced_paths.discard(path)

        # Remove the selected row; in the duplicates tree also clean up parent if only 1 child left
        for iid in tv.selection():
            parent = tv.parent(iid)
            tv.delete(iid)
            if parent and len(tv.get_children(parent)) <= 1:
                tv.delete(parent)   # no longer a duplicate group

        self._status.set(f"🗑  Recycled: {name}")
        self._update_tab_labels()

    # ── Column visibility & width persistence ────────────────────────────

    def _col_vis_load(self, key: str, all_ids: list) -> set:
        try:
            with open(CONFIG_FILE) as f:
                d = json.load(f)
            saved = d.get("col_visibility", {}).get(key)
            if saved:
                return set(saved) & set(all_ids)
        except Exception:
            pass
        return set(all_ids)

    def _col_vis_save(self, key: str, visible: set):
        try:
            try:
                with open(CONFIG_FILE) as f:
                    d = json.load(f)
            except Exception:
                d = {}
            d.setdefault("col_visibility", {})[key] = list(visible)
            with open(CONFIG_FILE, "w") as f:
                json.dump(d, f, indent=2)
        except Exception:
            pass

    def _col_widths_load(self, tv: ttk.Treeview, key: str):
        try:
            with open(CONFIG_FILE) as f:
                d = json.load(f)
            for col, w in d.get("col_widths", {}).get(key, {}).items():
                try:
                    tv.column(col, width=int(w))
                except Exception:
                    pass
        except Exception:
            pass

    def _col_widths_save(self, tv: ttk.Treeview, key: str):
        try:
            widths = {col: tv.column(col, "width") for col in tv["columns"]}
            try:
                with open(CONFIG_FILE) as f:
                    d = json.load(f)
            except Exception:
                d = {}
            d.setdefault("col_widths", {})[key] = widths
            with open(CONFIG_FILE, "w") as f:
                json.dump(d, f, indent=2)
        except Exception:
            pass

    def _open_col_picker(self, key: str):
        tv_map = {
            "inv": (self._inv_tv, self._inv_all_cols, "_inv_vis"),
            "dup": (self._dup_tv, self._dup_all_cols, "_dup_vis"),
            "mis": (self._mis_tv, self._mis_all_cols, "_mis_vis"),
            "unu": (self._unu_tv, self._unu_all_cols, "_unu_vis"),
        }
        tv, all_cols, vis_attr = tv_map[key]
        visible = getattr(self, vis_attr)

        dlg = tk.Toplevel(self)
        dlg.title("Choose Columns")
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.update_idletasks()
        h = 76 + len(all_cols) * 30 + 44
        x = (dlg.winfo_screenwidth()  - 240) // 2
        y = (dlg.winfo_screenheight() - h)   // 2
        dlg.geometry(f"240x{h}+{x}+{y}")

        tk.Label(dlg, text="⚙  COLUMNS", bg=BG, fg=ACC,
                 font=("Consolas",11,"bold")).pack(pady=(12,6))

        vars_: dict = {}
        for col, head, _, _ in all_cols:
            var = tk.BooleanVar(value=col in visible)
            vars_[col] = var
            row = tk.Frame(dlg, bg=BG); row.pack(anchor="w", padx=16, pady=1)
            btn = tk.Button(row, text="☑" if var.get() else "☐",
                            bg=BG, fg=ACC, font=("Consolas",11), relief="flat", bd=0,
                            activebackground=BG, cursor="hand2")
            lbl = tk.Label(row, text=head, bg=BG, fg=FG, font=("Consolas",10), cursor="hand2")
            def _toggle(v=var, b=btn):
                v.set(not v.get())
                b.configure(text="☑" if v.get() else "☐")
            btn.configure(command=_toggle)
            lbl.bind("<Button-1>", lambda _, t=_toggle: t())
            btn.pack(side="left")
            lbl.pack(side="left", padx=(4,0))

        def apply():
            checked = {c for c, v in vars_.items() if v.get()}
            if not checked: checked = {all_cols[0][0]}
            setattr(self, vis_attr, checked)
            tv["displaycolumns"] = [c[0] for c in all_cols if c[0] in checked]
            self._col_vis_save(key, checked)
            dlg.destroy()

        ttk.Button(dlg, text="Apply", style="Accent.TButton",
                   command=apply).pack(pady=10)
        dlg.bind("<Return>", lambda _: apply())

    # ── Scan workflow ────────────────────────────────────────────────────

    def _start_scan(self):
        enabled = [e for e in self._yaml_entries if e.get("enabled")]
        if not enabled:
            messagebox.showwarning("Nothing enabled",
                                   "Check at least one installation."); return
        if not YAML_OK:
            self._status.set("PyYAML not found — using built-in fallback parser (works fine for ComfyUI YAML)")
        self._models = []; self._dupes_n = []; self._dupes_h = []
        self._misplaced = []; self._unused = []; self._wf_dirs = []
        self._hash_done = False

        self._pb.pack(fill="x", padx=16, pady=(4,0))
        self._pb.start(12)
        self._scan_sv.set("Scanning…")
        self._status.set("Scanning model libraries…")
        threading.Thread(target=self._scan_worker, args=(enabled,), daemon=True).start()

    def _scan_worker(self, entries):
        def cb(msg):
            self.after(0, self._status.set, msg)

        models = scan_models(entries, cb=cb)
        cb(f"Analysing {len(models):,} models…")

        dupes_n  = find_name_duplicates(models)
        misplaced = find_misplaced(models)
        unused, wf_dirs = find_unused(models, entries, cb=cb)

        self.after(0, self._scan_done,
                   models, dupes_n, misplaced, unused, wf_dirs)

    def _scan_done(self, models, dupes_n, misplaced, unused, wf_dirs):
        self._pb.stop(); self._pb.pack_forget()
        self._models          = models
        self._dupes_n         = dupes_n
        self._misplaced       = misplaced
        self._misplaced_paths = {mp.model.path for mp in misplaced}
        self._unused    = unused
        self._wf_dirs   = wf_dirs

        self._populate_inventory()
        self._populate_duplicates()
        self._populate_misplaced()
        self._populate_unused()
        self._update_tab_labels()

        total_sz = sum(m.size for m in models)
        self._scan_sv.set(
            f"\u2713 {len(models):,} models  \u2022  {_fmtsz(total_sz)}")
        self._status.set(
            f"Scan complete — {len(models):,} models | "
            f"{len(dupes_n)} name dupes | "
            f"{len(misplaced)} misplaced | "
            f"{len(unused)} unused")

    # ── Populate tabs ────────────────────────────────────────────────────

    def _populate_inventory(self):
        types    = sorted({m.folder_type   for m in self._models})
        installs = sorted({m.install_label for m in self._models})
        drives   = sorted({Path(m.path).drive.upper() for m in self._models if m.path})
        self._inv_type_cb["values"]  = ["All"] + types
        self._inv_inst_cb["values"]  = ["All"] + installs
        self._inv_drive_cb["values"] = ["All"] + drives
        self._inv_type_var.set("All"); self._inv_inst_var.set("All"); self._inv_drive_var.set("All")
        self._filter_inventory()

    def _populate_duplicates(self):
        self._show_dupes()

    def _populate_misplaced(self):
        tv = self._mis_tv
        tv.delete(*tv.get_children())
        for mp in self._misplaced:
            m = mp.model
            tag = "high" if mp.confidence == "High" else "medium"
            tv.insert("", "end", tags=(tag,), values=(
                mp.confidence,
                m.filename, m.folder_type,
                mp.suggested_type or "?",
                m.size_str, mp.reason,
                m.install_label, m.path))
        hi = sum(1 for x in self._misplaced if x.confidence=="High")
        md = len(self._misplaced) - hi
        self._mis_sum.set(
            f"  {len(self._misplaced):,} suspicious files  "
            f"({hi} high confidence, {md} medium confidence)")

    def _populate_unused(self):
        tv = self._unu_tv
        tv.delete(*tv.get_children())
        for m in self._unused:
            tv.insert("", "end", values=(
                m.filename, m.folder_type, m.size_str,
                m.mtime_str, m.install_label, m.path))
        total_sz = sum(m.size for m in self._unused)
        dirs_str = "  |  ".join(self._wf_dirs) if self._wf_dirs else "none found"
        self._unused_note.set(
            f"Models not referenced in any workflow scanned.  "
            f"Workflow locations scanned:  {dirs_str}")
        self._unu_sum.set(
            f"  {len(self._unused):,} unused models    "
            f"{_fmtsz(total_sz)} could be freed / moved to warehouse")

    def _update_tab_labels(self):
        d_cnt = len(self._dupes_n)
        m_cnt = len(self._misplaced)
        u_cnt = len(self._unused)
        i_cnt = len(self._models)
        self._nb.tab(0, text=f"  \U0001f4e6 Inventory ({i_cnt:,})  ")
        self._nb.tab(1, text=f"  \U0001f501 Duplicates ({d_cnt})  "
                    if d_cnt else "  \U0001f501 Duplicates  ")
        self._nb.tab(2, text=f"  \U0001f4c2 Misplaced ({m_cnt})  "
                    if m_cnt else "  \U0001f4c2 Misplaced  ")
        self._nb.tab(3, text=f"  \U0001f4a4 Unused ({u_cnt:,})  "
                    if u_cnt else "  \U0001f4a4 Unused  ")

    # ── Hash scan ────────────────────────────────────────────────────────

    def _start_hash_scan(self):
        if not self._models:
            messagebox.showinfo("No models", "Run a scan first."); return
        self._hash_btn.configure(state="disabled", text="\u23f3 Hashing…")
        self._pb.pack(fill="x", padx=16, pady=(4,0)); self._pb.start(12)
        threading.Thread(target=self._hash_worker, daemon=True).start()

    def _hash_worker(self):
        def cb(msg): self.after(0, self._status.set, msg)
        groups = find_hash_duplicates(self._models, cb=cb)
        self.after(0, self._hash_done_cb, groups)

    def _hash_done_cb(self, groups):
        self._pb.stop(); self._pb.pack_forget()
        self._dupes_h = groups
        self._hash_done = True
        self._hash_btn.configure(state="normal",
                                 text=f"\u2713 Hash done ({len(groups)} exact dupes)")
        self._dupe_mode.set("hash")
        self._show_dupes()
        self._update_tab_labels()
        self._status.set(
            f"Hash scan complete — {len(groups)} exact duplicate group(s) found")


# ─────────────────────────────────────────────────────────────────────────────
# Tiny helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmtsz(n: int) -> str:
    if n >= GB: return f"{n/GB:.2f} GB"
    if n >= MB: return f"{n/MB:.1f} MB"
    return f"{n/1024:.1f} KB"


def _row_color(entry: dict) -> str:
    if not os.path.isfile(entry["path"]): return "#6a3a3a"
    return FG2 if entry.get("enabled") else "#404060"


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ModelMaintenance()
    app.mainloop()
