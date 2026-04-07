"""
ComfyUI Workflow Model Scanner
- Open a workflow JSON with the file dialog
- Locates models across C:\\ComfyUI.Data\\models and F:\\ComfyUI.stuff\\models
- Shows file size, detects wrong ComfyUI subfolder, flags duplicates

Log file: scanner.log (written next to this script)
"""

import json
import logging
import os
import sys
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from collections import Counter

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_PATH = Path(__file__).parent / "scanner.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("scanner")

def log_exc(context: str):
    log.error(f"CRASH in {context}:\n{traceback.format_exc()}")

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_ROOTS = [
    r"C:\ComfyUI.Data\models",
    r"F:\ComfyUI.stuff\models",
]

MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft"}

MODEL_FIELDS = {
    "ckpt_name":        "Checkpoint",
    "unet_name":        "UNet",
    "vae_name":         "VAE",
    "lora_name":        "LoRA",
    "lora_01":          "LoRA",
    "lora_02":          "LoRA",
    "lora_03":          "LoRA",
    "clip_name":        "CLIP",
    "clip_name1":       "CLIP",
    "clip_name2":       "CLIP",
    "control_net_name": "ControlNet",
    "model_name":       "Model",
    "upscale_model":    "Upscaler",
    "encoder_name":     "Encoder",
    "decoder_name":     "Decoder",
    "embed_name":       "Embedding",
    "style_model_path": "Style Model",
    "ip_adapter_file":  "IP-Adapter",
    "ipadapter_file":   "IP-Adapter",
    "weight_dtype":     None,
}

# Expected ComfyUI subfolder name(s) per category.
# A model is "in the right folder" if its path contains at least one of these
# as a path component (case-insensitive).
EXPECTED_FOLDERS: dict[str, set[str]] = {
    "Checkpoint":  {"checkpoints", "checkpoint"},
    "UNet":        {"unet", "unet_models", "diffusion_models", "diffusion-models"},
    "VAE":         {"vae"},
    "LoRA":        {"loras", "lora"},
    "CLIP":        {"clip", "text_encoders", "text_encoder"},
    "ControlNet":  {"controlnet", "control_net", "control-net"},
    "Upscaler":    {"upscale_models", "upscale", "upscaler"},
    "Embedding":   {"embeddings", "embedding", "textual_inversion"},
    "IP-Adapter":  {"ipadapter", "ip_adapter", "ip-adapter"},
    "Style Model": {"style_models", "style_model"},
    "Encoder":     {"clip", "text_encoders"},
    "Decoder":     {"vae"},
    "Model":       set(),  # generic — skip check
}

CATCH_BY_EXTENSION = True

# ── Model search (HuggingFace + CivitAI) ──────────────────────────────────────

def _hf_api(url: str, timeout: int = 6):
    """Call a HuggingFace API endpoint, return parsed JSON or None."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-Model-Scanner/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        log.debug(f"HF API error: {e}")
        return None

def _civitai_api(url: str, timeout: int = 6):
    """Call a CivitAI API endpoint, return parsed JSON or None."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-Model-Scanner/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        log.debug(f"CivitAI API error: {e}")
        return None

def search_model_online(filename: str) -> list[dict]:
    """
    Search HuggingFace and CivitAI for a model file.
    Returns a list of result dicts: {source, name, url, download_url, info}
    """
    stem    = Path(filename).stem        # e.g. "ltx-2.3-22b-dev-fp8"
    results = []

    # ── HuggingFace: search models API ────────────────────────────────────────
    q     = urllib.parse.quote(stem)
    data  = _hf_api(f"https://huggingface.co/api/models?search={q}&limit=5&sort=downloads")
    if data:
        for model in data[:5]:
            model_id  = model.get("modelId") or model.get("id", "")
            # Check if any sibling file matches our filename
            siblings  = model.get("siblings", [])
            matched   = [s for s in siblings if s.get("rfilename","").lower() == filename.lower()]
            if matched:
                dl_url = f"https://huggingface.co/{model_id}/resolve/main/{matched[0]['rfilename']}"
                page   = f"https://huggingface.co/{model_id}"
                results.append({
                    "source":       "HuggingFace",
                    "name":         model_id,
                    "url":          page,
                    "download_url": dl_url,
                    "info":         f"exact file match in {model_id}",
                })
            elif not matched and model_id:
                # Loose match — link to the repo page so user can browse
                results.append({
                    "source":       "HuggingFace",
                    "name":         model_id,
                    "url":          f"https://huggingface.co/{model_id}",
                    "download_url": None,
                    "info":         "possible match (browse repo)",
                })

    # ── CivitAI: search by filename stem ──────────────────────────────────────
    q2   = urllib.parse.quote(stem.replace("-", " ").replace("_", " "))
    cdata = _civitai_api(f"https://civitai.com/api/v1/models?query={q2}&limit=5")
    if cdata:
        for model in cdata.get("items", [])[:5]:
            name     = model.get("name", "")
            model_id = model.get("id")
            page     = f"https://civitai.com/models/{model_id}" if model_id else None
            # Try to find a version file that matches
            dl_url = None
            for ver in model.get("modelVersions", [])[:3]:
                for f in ver.get("files", []):
                    if f.get("name","").lower() == filename.lower():
                        dl_url = f.get("downloadUrl")
                        break
                if dl_url:
                    break
            if page:
                results.append({
                    "source":       "CivitAI",
                    "name":         name,
                    "url":          page,
                    "download_url": dl_url,
                    "info":         "exact file match" if dl_url else "possible match",
                })

    log.info(f"Search '{filename}': {len(results)} result(s)")
    return results

# ── Helpers ────────────────────────────────────────────────────────────────────

def fmt_size(nbytes: int) -> str:
    if nbytes <= 0:
        return "?"
    if nbytes >= 1_073_741_824:
        return f"{nbytes / 1_073_741_824:.1f} GB"
    return f"{nbytes / 1_048_576:.0f} MB"

def path_parts_lower(p: str) -> set[str]:
    """Return the set of all folder name components of a path, lowercased."""
    return {part.lower() for part in Path(p).parts}

def folder_ok(full_path: str, category: str) -> bool:
    """Return True if the model sits inside the expected ComfyUI subfolder."""
    expected = EXPECTED_FOLDERS.get(category, set())
    if not expected:
        return True  # no rule → don't flag it
    parts = path_parts_lower(full_path)
    return bool(parts & expected)

# ── Model index ────────────────────────────────────────────────────────────────
# index: lower(filename) → list of {"path": str, "size": int}

def build_model_index(roots: list[str]) -> dict:
    index: dict[str, list[dict]] = {}
    total = 0
    for root in roots:
        if not os.path.isdir(root):
            log.warning(f"Root not found, skipping: {root}")
            continue
        log.info(f"Indexing: {root}")
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if Path(fname).suffix.lower() in MODEL_EXTENSIONS:
                    full = os.path.join(dirpath, fname)
                    try:
                        size = os.path.getsize(full)
                    except OSError:
                        size = 0
                    index.setdefault(fname.lower(), []).append({"path": full, "size": size})
                    total += 1
    log.info(f"Index complete: {total} files")
    return index

def preferred_location(entries: list[dict], index: dict) -> dict:
    """
    Prefer the C: drive copy — it's the fast local drive ComfyUI loads from.
    F: (or any other drive) is treated as slow warehouse storage.
    Falls back to alphabetical drive order if no C: copy exists.
    """
    if len(entries) == 1:
        return entries[0]
    def drive_rank(e):
        d = e["path"][:2].upper()
        if d == "C:": return 0
        if d == "D:": return 1   # common secondary SSD
        return ord(d[0])         # E, F, G... in order
    return min(entries, key=lambda e: (drive_rank(e), e["path"].lower()))

# ── Workflow parsing ───────────────────────────────────────────────────────────

def extract_models_from_workflow(workflow: dict) -> list[dict]:
    found: list[dict] = []
    seen:  set        = set()

    def record(field, value, node_id, node_type, source):
        if not isinstance(value, str) or not value.strip():
            return
        if MODEL_FIELDS.get(field) is None and field in MODEL_FIELDS:
            return
        ext      = Path(value).suffix.lower()
        is_known = field in MODEL_FIELDS and MODEL_FIELDS[field] is not None
        is_ext   = ext in MODEL_EXTENSIONS
        if not (is_known or (CATCH_BY_EXTENSION and is_ext)):
            return
        key = (field, value)
        if key in seen:
            return
        seen.add(key)
        entry = {
            "field":     field,
            "value":     value,
            "filename":  Path(value).name,
            "node_id":   str(node_id),
            "node_type": node_type,
            "category":  MODEL_FIELDS.get(field, "Model"),
            "source":    source,
        }
        log.debug(f"  ref: {entry['filename']}  field={field}  src={source}")
        found.append(entry)

    def scan_nodes(nodes, source):
        for node in nodes:
            if not isinstance(node, dict):
                continue
            nid  = node.get("id", "?")
            ntyp = node.get("type", "Unknown")
            for inp in node.get("inputs", []):
                val = inp.get("widget", {})
                if isinstance(val, dict):
                    record(inp.get("name", ""), val.get("value", ""), nid, ntyp, source)
            for wv in node.get("widgets_values", []):
                if isinstance(wv, str) and Path(wv).suffix.lower() in MODEL_EXTENSIONS:
                    record("widgets_values", wv, nid, ntyp, source)
            for m in node.get("properties", {}).get("models", []):
                if isinstance(m, dict):
                    name = m.get("name", "")
                    if name and Path(name).suffix.lower() in MODEL_EXTENSIONS:
                        record("properties.models", name, nid, ntyp, source)

    log.debug(f"Workflow keys: {list(workflow.keys())[:10]}")

    if all(isinstance(k, str) and k.isdigit() for k in list(workflow.keys())[:5]):
        log.info("Format: API")
        for nid, node in workflow.items():
            if isinstance(node, dict):
                for field, value in node.get("inputs", {}).items():
                    record(field, value, nid, node.get("class_type", "?"), "top-level")
    elif "nodes" in workflow:
        log.info("Format: UI")
        scan_nodes(workflow["nodes"], "top-level")
        subgraphs = workflow.get("definitions", {}).get("subgraphs", [])
        log.info(f"{len(subgraphs)} subgraph(s)")
        for sg in subgraphs:
            if isinstance(sg, dict):
                sg_name = sg.get("name", sg.get("id", "?"))
                scan_nodes(sg.get("nodes", []), f"subgraph:{sg_name}")
    else:
        log.warning("Unknown format — deep scan")
        def deep(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    record(k, v, "?", "?", "deep") if isinstance(v, str) else deep(v)
            elif isinstance(obj, list):
                [deep(i) for i in obj]
        deep(workflow)

    log.info(f"Extraction: {len(found)} model ref(s)")
    return found

# ── Resolution ─────────────────────────────────────────────────────────────────

def resolve_models(refs: list[dict], index: dict) -> list[dict]:
    results = []
    for ref in refs:
        fname   = ref["filename"].lower()
        vname   = Path(ref["value"]).name.lower()
        entries = index.get(fname) or index.get(vname) or []

        if not entries:
            log.debug(f"  MISSING  {ref['filename']}")
            results.append({
                **ref,
                "found":       False,
                "locations":   [],
                "size_bytes":  0,
                "size_fmt":    "—",
                "folder_ok":   True,   # can't check if not found
                "preferred":   None,
                "duplicate":   False,
            })
            continue

        pref      = preferred_location(entries, index)
        duplicate = len(entries) > 1
        f_ok      = folder_ok(pref["path"], ref.get("category", "Model"))

        log.debug(
            f"  FOUND {'DUP ' if duplicate else '    '}{'BAD_DIR ' if not f_ok else '       '}"
            f"{ref['filename']}  →  {pref['path']}"
        )
        results.append({
            **ref,
            "found":      True,
            "locations":  entries,          # all copies
            "size_bytes": pref["size"],
            "size_fmt":   fmt_size(pref["size"]),
            "folder_ok":  f_ok,
            "preferred":  pref["path"],
            "duplicate":  duplicate,
        })
    return results

# ── GUI ────────────────────────────────────────────────────────────────────────

BG     = "#1a1a2e"
PANEL  = "#16213e"
ACCENT = "#0f3460"
LIME   = "#00ff87"
AMBER  = "#ffb300"
RED    = "#ff5252"
TEXT   = "#e0e0e0"
DIM    = "#888888"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ComfyUI Model Scanner")
        self.geometry("1280x800")
        self.minsize(900, 560)
        self.configure(bg=BG)
        self._setup_styles()
        self._build_ui()
        self.model_index: dict = {}
        self.results:     list = []
        log.info("App started")

    # ── Styles ─────────────────────────────────────────────────────────────────
    def _setup_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame",        background=BG)
        s.configure("Panel.TFrame",  background=PANEL)
        s.configure("TLabel",        background=BG,    foreground=TEXT,  font=("Consolas", 10))
        s.configure("Header.TLabel", background=BG,    foreground=LIME,  font=("Consolas", 13, "bold"))
        s.configure("Dim.TLabel",    background=BG,    foreground=DIM,   font=("Consolas", 9))
        s.configure("Panel.TLabel",  background=PANEL, foreground=TEXT,  font=("Consolas", 10))
        s.configure("TButton",
            background=ACCENT, foreground=LIME,
            font=("Consolas", 10, "bold"), relief="flat", padding=(14, 7)
        )
        s.map("TButton",
            background=[("active", "#1a4a8a"), ("pressed", "#0a2a5a")],
            foreground=[("active", "#ffffff")]
        )
        s.configure("Treeview",
            background=PANEL, foreground=TEXT, fieldbackground=PANEL,
            rowheight=50, font=("Consolas", 10)
        )
        s.configure("Treeview.Heading",
            background=ACCENT, foreground=LIME,
            font=("Consolas", 10, "bold"), relief="flat"
        )
        s.map("Treeview",
            background=[("selected", "#1a3a6a")],
            foreground=[("selected", "#ffffff")]
        )
        s.configure("TProgressbar", troughcolor=PANEL, background=LIME, thickness=4)
        s.configure("TNotebook",              background=PANEL, borderwidth=0)
        s.configure("TNotebook.Tab",
            background=ACCENT, foreground=DIM,
            font=("Consolas", 9), padding=(10, 4)
        )
        s.map("TNotebook.Tab",
            background=[("selected", PANEL)],
            foreground=[("selected", LIME)]
        )

        # Row tags
        self._tags = {
            "found":       {"background": "#0d2d1f", "foreground": "#00e676"},
            "missing":     {"background": "#2d0d0d", "foreground": "#ff5252"},
            "duplicate":   {"background": "#1a2000", "foreground": "#c6ff00"},
            "wrong_folder":{"background": "#2a1800", "foreground": "#ffb300"},
            "dup_wrong":   {"background": "#2a1a00", "foreground": "#ff9800"},
        }

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        top = ttk.Frame(self, padding=(16, 12, 16, 6))
        top.pack(fill="x")
        ttk.Label(top, text="⬡  ComfyUI Model Scanner", style="Header.TLabel").pack(side="left")
        ttk.Label(top, text="size · folder check · duplicate detection", style="Dim.TLabel").pack(side="left", padx=(12, 0))

        # Controls
        ctrl = ttk.Frame(self, padding=(16, 2, 16, 6))
        ctrl.pack(fill="x")
        self.workflow_var = tk.StringVar(value="No workflow loaded")
        ttk.Button(ctrl, text="📂  Open Workflow JSON", command=self._open_workflow).pack(side="left")
        ttk.Button(ctrl, text="🔍  Re-scan Drives",     command=self._scan_drives  ).pack(side="left", padx=(8, 0))
        ttk.Button(ctrl, text="📋  Copy Path",          command=self._copy_path    ).pack(side="left", padx=(8, 0))
        ttk.Button(ctrl, text="📄  Open Log",           command=self._open_log     ).pack(side="left", padx=(8, 0))
        ttk.Label( ctrl, textvariable=self.workflow_var, style="Dim.TLabel").pack(side="left", padx=(16, 0))

        # Legend
        legend = ttk.Frame(self, padding=(16, 0, 16, 4))
        legend.pack(fill="x")
        for color, label in [
            (LIME,  "✔ Found"),
            (RED,   "✘ Missing"),
            ("#c6ff00", "⊗ Duplicate"),
            (AMBER, "⚠ Wrong folder"),
        ]:
            tk.Label(legend, text=label, bg=BG, fg=color,
                     font=("Consolas", 9)).pack(side="left", padx=(0, 16))

        # Filter row
        filt = ttk.Frame(self, padding=(16, 0, 16, 4))
        filt.pack(fill="x")
        ttk.Label(filt, text="Filter:", style="Dim.TLabel").pack(side="left")
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", lambda *_: self._apply_filter())
        tk.Entry(filt, textvariable=self.filter_var,
                 bg="#0f3460", fg="#e0e0e0", insertbackground=LIME,
                 relief="flat", font=("Consolas", 10), width=28
                 ).pack(side="left", padx=(6, 16))

        self.show_var = tk.StringVar(value="All")
        for label in ("All", "Found", "Missing", "Wrong Folder", "Duplicates"):
            tk.Radiobutton(filt, text=label, variable=self.show_var, value=label,
                           command=self._apply_filter,
                           bg=BG, fg=LIME, selectcolor=BG,
                           activebackground=BG, activeforeground="#fff",
                           font=("Consolas", 10), relief="flat"
                           ).pack(side="left", padx=(0, 6))

        # Main pane
        pane = tk.PanedWindow(self, orient="vertical", bg=BG, sashwidth=5, sashrelief="flat")
        pane.pack(fill="both", expand=True, padx=16, pady=(4, 0))

        # Results table
        tree_frame = ttk.Frame(pane, style="Panel.TFrame")
        pane.add(tree_frame, minsize=220)

        # Columns: Type | Model File | Size | Status | Folder | Path
        cols = ("category", "filename", "size", "status", "folder", "location")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse")
        self.tree.heading("category", text="Type",        command=lambda: self._sort("category"))
        self.tree.heading("filename", text="Model File",  command=lambda: self._sort("filename"))
        self.tree.heading("size",     text="Size",        command=lambda: self._sort("size"))
        self.tree.heading("status",   text="Status",      command=lambda: self._sort("status"))
        self.tree.heading("folder",   text="Folder",      command=lambda: self._sort("folder"))
        self.tree.heading("location", text="Full Path")
        self.tree.column("category", width=110, stretch=False)
        self.tree.column("filename", width=290, stretch=False)
        self.tree.column("size",     width=80,  stretch=False, anchor="e")
        self.tree.column("status",   width=115, stretch=False)
        self.tree.column("folder",   width=105, stretch=False)
        self.tree.column("location", width=500)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical",  command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        for tag, cfg in self._tags.items():
            self.tree.tag_configure(tag, **cfg)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        self._sort_col = None
        self._sort_rev = False

        # Bottom tabs: Detail + Log
        bottom = ttk.Frame(pane, style="Panel.TFrame", padding=6)
        pane.add(bottom, minsize=120)

        nb = ttk.Notebook(bottom)
        nb.pack(fill="both", expand=True)

        detail_tab = ttk.Frame(nb, style="Panel.TFrame")
        nb.add(detail_tab, text="  Model Detail  ")
        self.detail_text = tk.Text(
            detail_tab, bg="#0f1a2e", fg="#e0e0e0",
            relief="flat", font=("Consolas", 10), state="disabled"
        )
        self.detail_text.pack(fill="both", expand=True)

        log_tab = ttk.Frame(nb, style="Panel.TFrame")
        nb.add(log_tab, text="  Log  ")
        self.log_text = tk.Text(
            log_tab, bg="#0a0f1a", fg="#aaaaaa",
            relief="flat", font=("Consolas", 9), state="disabled"
        )
        log_vsb = ttk.Scrollbar(log_tab, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_vsb.set)
        log_vsb.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)
        self._wire_log_to_widget()

        # Status bar
        bar = ttk.Frame(self, padding=(16, 4, 16, 8))
        bar.pack(fill="x")
        self.status_var = tk.StringVar(value="Ready — click Open Workflow JSON to begin.")
        ttk.Label(bar, textvariable=self.status_var, style="Dim.TLabel").pack(side="left")
        self.stats_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.stats_var, style="Dim.TLabel").pack(side="right")

        self.progress = ttk.Progressbar(self, mode="indeterminate", style="TProgressbar")

    # ── Log widget handler ─────────────────────────────────────────────────────
    def _wire_log_to_widget(self):
        widget = self.log_text
        COLORS = {"DEBUG": "#555577", "INFO": "#aaaaaa",
                  "WARNING": "#ffb300", "ERROR": "#ff5252", "CRITICAL": "#ff1744"}

        class WidgetHandler(logging.Handler):
            def emit(self_h, record):
                msg   = self_h.format(record) + "\n"
                color = COLORS.get(record.levelname, "#aaaaaa")
                def append():
                    widget.configure(state="normal")
                    widget.insert("end", msg, record.levelname)
                    widget.tag_configure(record.levelname, foreground=color)
                    widget.see("end")
                    widget.configure(state="disabled")
                try:
                    widget.after(0, append)
                except Exception:
                    pass

        h = WidgetHandler()
        h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
        log.addHandler(h)

    # ── Workflow loading ───────────────────────────────────────────────────────
    def _open_workflow(self):
        path = filedialog.askopenfilename(
            title="Open ComfyUI Workflow",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            log.info(f"Loading: {path}")
            with open(path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
        except Exception as e:
            log_exc("json.load")
            self._show_error("Could not read workflow", e)
            return

        try:
            self.workflow_var.set(os.path.basename(path))
            self.status_var.set(f"Parsing: {os.path.basename(path)}")
            self.update_idletasks()
            refs = extract_models_from_workflow(workflow)
            if not refs:
                messagebox.showinfo("No Models", "No model references found.")
                self.status_var.set("No model references found.")
                return
            sub = sum(1 for r in refs if r["source"].startswith("subgraph:"))
            self.status_var.set(
                f"{len(refs)} model ref(s) — {len(refs)-sub} top-level, {sub} in subgraphs — scanning drives…"
            )
            self._start_scan(refs)
        except Exception as e:
            log_exc("parse")
            self._show_error("Parse error", e)

    # ── Drive scan ─────────────────────────────────────────────────────────────
    def _scan_drives(self):
        if not self.results:
            messagebox.showinfo("No Workflow", "Load a workflow first.")
            return
        refs = [{k: v for k, v in r.items()
                 if k not in ("found","locations","size_bytes","size_fmt",
                              "folder_ok","preferred","duplicate")}
                for r in self.results]
        self._start_scan(refs)

    def _start_scan(self, refs):
        self.progress.pack(fill="x", padx=16, pady=(0, 4))
        self.progress.start(12)
        self.status_var.set("Scanning drives…")
        def worker():
            try:
                index    = build_model_index(MODEL_ROOTS)
                resolved = resolve_models(refs, index)
                self.after(0, lambda: self._scan_done(index, resolved))
            except Exception as e:
                log_exc("scan worker")
                self.after(0, lambda: (self._stop_progress(), self._show_error("Scan error", e)))
        threading.Thread(target=worker, daemon=True).start()

    def _stop_progress(self):
        self.progress.stop()
        self.progress.pack_forget()

    def _scan_done(self, index, results):
        self._stop_progress()
        self.model_index = index
        self.results     = results
        self._apply_filter()

        found      = sum(1 for r in results if r["found"])
        missing    = sum(1 for r in results if not r["found"])
        duplicates = sum(1 for r in results if r.get("duplicate"))
        wrong      = sum(1 for r in results if r["found"] and not r["folder_ok"])

        parts = [f"✔ {found} found", f"✘ {missing} missing"]
        if duplicates: parts.append(f"⊗ {duplicates} duplicated")
        if wrong:      parts.append(f"⚠ {wrong} wrong folder")
        self.stats_var.set("   ".join(parts))
        self.status_var.set(f"Scan complete — {len(index)} model files indexed.")
        log.info(f"Done: {found} found, {missing} missing, {duplicates} dup, {wrong} wrong folder")

    # ── Tree population ────────────────────────────────────────────────────────
    def _row_tag(self, r: dict) -> str:
        if not r["found"]:
            return "missing"
        dup   = r.get("duplicate", False)
        f_ok  = r.get("folder_ok", True)
        if dup and not f_ok: return "dup_wrong"
        if not f_ok:         return "wrong_folder"
        if dup:              return "duplicate"
        return "found"

    def _status_label(self, r: dict) -> str:
        if not r["found"]:
            return "✘ MISSING"
        parts = []
        if r.get("duplicate"):
            n = len(r.get("locations", []))
            parts.append(f"⊗ ×{n} copies")
        if not r.get("folder_ok", True):
            parts.append("⚠ wrong folder")
        return "  ".join(parts) if parts else "✔ Found"

    def _populate_tree(self, results):
        self.tree.delete(*self.tree.get_children())
        for r in results:
            if not r["found"]:
                loc_cell = "— not found —"
            else:
                # Sort copies: C: first, then F:, then anything else
                entries = sorted(
                    r.get("locations", []),
                    key=lambda e: (0 if e["path"].upper().startswith("C:") else 1, e["path"].lower())
                )
                loc_cell = "\n".join(e["path"] for e in entries)

            self.tree.insert("", "end",
                values=(
                    r.get("category", "Model"),
                    r["filename"],
                    r.get("size_fmt", "—"),
                    self._status_label(r),
                    "✔ OK" if r.get("folder_ok", True) else "⚠ wrong",
                    loc_cell,
                ),
                tags=(self._row_tag(r),),
                iid=r["filename"] + r["field"]
            )

    def _apply_filter(self):
        if not self.results:
            return
        text = self.filter_var.get().lower()
        show = self.show_var.get()

        def match(r):
            if text and text not in r["filename"].lower() and text not in r.get("category","").lower():
                return False
            if show == "Found":        return r["found"]
            if show == "Missing":      return not r["found"]
            if show == "Wrong Folder": return r["found"] and not r.get("folder_ok", True)
            if show == "Duplicates":   return r.get("duplicate", False)
            return True

        self._populate_tree([r for r in self.results if match(r)])

    # ── Sort ───────────────────────────────────────────────────────────────────
    def _sort(self, col):
        col_idx = {"category": 0, "filename": 1, "size": 2, "status": 3, "folder": 4}
        idx = col_idx.get(col, 1)
        if self._sort_col == col:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_col = col
            self._sort_rev = False

        def sort_key(r):
            if col == "size":
                return r.get("size_bytes", 0)
            return str(self.tree.item(r["filename"] + r["field"])["values"][idx]).lower()

        # Re-sort self.results in the current filtered view
        filtered = [r for r in self.results
                    if self.tree.exists(r["filename"] + r["field"])]
        filtered.sort(key=sort_key, reverse=self._sort_rev)
        self._populate_tree(filtered)

    # ── Detail panel ───────────────────────────────────────────────────────────
    def _open_folder(self, folder: str):
        """Open Windows Explorer at the given folder path."""
        try:
            import subprocess
            subprocess.Popen(f'explorer /select,"{folder}"', shell=True)
        except Exception as e:
            log_exc("_open_folder")

    def _detail_insert_link(self, label: str, folder: str, tag_id: str):
        """Insert an underlined clickable link inline into self.detail_text."""
        self.detail_text.insert("end", label, tag_id)
        self.detail_text.tag_configure(tag_id,
            foreground="#4fc3f7", underline=True, font=("Consolas", 9))
        self.detail_text.tag_bind(tag_id, "<Button-1>",
            lambda e, f=folder: self._open_folder(f))
        self.detail_text.tag_bind(tag_id, "<Enter>",
            lambda e: self.detail_text.configure(cursor="hand2"))
        self.detail_text.tag_bind(tag_id, "<Leave>",
            lambda e: self.detail_text.configure(cursor=""))

    def _on_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        fname = self.tree.item(sel[0])["values"][1]
        r = next((x for x in self.results if x["filename"] == fname), None)
        if not r:
            return

        locs = r.get("locations", [])

        # ── Sort copies: C: first ──────────────────────────────────────────────
        sorted_locs = sorted(
            locs,
            key=lambda e: (0 if e["path"].upper().startswith("C:") else 1, e["path"].lower())
        )

        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")

        def line(text):
            self.detail_text.insert("end", text + "\n")

        line(f"  Model File : {r['filename']}")
        line(f"  Category   : {r.get('category','?')}")
        line(f"  Size       : {r.get('size_fmt','—')}")
        line(f"  Field      : {r['field']}")
        line(f"  Node Type  : {r['node_type']}  (id {r['node_id']})")
        line(f"  Source     : {r.get('source','?')}")
        line(f"  Status     : {'FOUND' if r['found'] else 'MISSING'}")

        if not r["found"]:
            line("")
            line("  ── Download Search ──────────────────────────────────────")
            # Search button
            btn_tag = f"searchbtn_{id(r)}"
            self.detail_text.insert("end", "  ")
            self.detail_text.insert("end", "  🔎  Search HuggingFace & CivitAI  ", btn_tag)
            self.detail_text.insert("end", "\n")
            self.detail_text.tag_configure(btn_tag,
                foreground=LIME, background="#0f3460",
                font=("Consolas", 10, "bold"), relief="raised")
            self.detail_text.tag_bind(btn_tag, "<Button-1>",
                lambda e, rec=r: self._search_missing(rec))
            self.detail_text.tag_bind(btn_tag, "<Enter>",
                lambda e: self.detail_text.configure(cursor="hand2"))
            self.detail_text.tag_bind(btn_tag, "<Leave>",
                lambda e: self.detail_text.configure(cursor=""))
            self._last_missing_result = r   # track which result the search is for

        if r["found"]:
            # Folder check
            expected = EXPECTED_FOLDERS.get(r.get("category",""), set())
            if expected:
                exp_str = " / ".join(sorted(expected))
                line(f"  Folder OK  : {'✔ Yes' if r['folder_ok'] else f'⚠ No  (expected one of: {exp_str})'}")

            if sorted_locs:
                pref_path = r.get("preferred", "")
                line(f"  Copies ({len(sorted_locs)}):")

                for idx, e in enumerate(sorted_locs):
                    marker = " ← preferred" if e["path"] == pref_path else ""
                    drive  = e["path"][:2].upper()
                    folder = str(Path(e["path"]).parent)
                    tag_id = f"link_{idx}_{id(r)}"
                    # Path + inline link on same line
                    self.detail_text.insert("end", f"    [{drive}] {e['path']}  ({fmt_size(e['size'])}){marker}   ")
                    self._detail_insert_link("↗ folder", folder, tag_id)
                    self.detail_text.insert("end", "\n")

                if len(sorted_locs) > 1:
                    line("")
                    pref_dir = str(Path(pref_path).parent) if pref_path else ""
                    line(f"  Preferred  : {pref_dir}")
                    line(f"  (C: drive preferred — fastest for ComfyUI to load from)")
                    other_dirs = [str(Path(e["path"]).parent)
                                  for e in sorted_locs if e["path"] != pref_path]
                    line(f"  Can delete : {chr(10).join('    ' + d for d in other_dirs)}")

        self.detail_text.configure(state="disabled")

    # ── Online search for missing models ──────────────────────────────────────
    def _open_url(self, url: str):
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception as e:
            log_exc("_open_url")

    def _search_missing(self, r: dict):
        """Run online search in background, then render results into detail panel."""
        filename = r["filename"]
        log.info(f"Searching online for: {filename}")
        self.status_var.set(f"Searching online for {filename}…")

        # Show a "searching…" spinner line
        self.detail_text.configure(state="normal")
        self.detail_text.insert("end", "  Searching…\n")
        self.detail_text.configure(state="disabled")

        def worker():
            try:
                hits = search_model_online(filename)
                self.after(0, lambda: self._show_search_results(r, hits))
            except Exception as e:
                log_exc("_search_missing worker")
                self.after(0, lambda: self.status_var.set(f"Search error: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _show_search_results(self, r: dict, hits: list):
        """Append search results with clickable links into the detail panel."""
        self.status_var.set(
            f"Found {len(hits)} result(s) for {r['filename']}" if hits
            else f"No results found for {r['filename']}"
        )

        self.detail_text.configure(state="normal")

        # Remove the "Searching…" line we added
        idx = self.detail_text.search("  Searching…", "1.0", tk.END)
        if idx:
            self.detail_text.delete(idx, f"{idx} lineend+1c")

        def line(text=""):
            self.detail_text.insert("end", text + "\n")

        if not hits:
            line("  No results found on HuggingFace or CivitAI.")
            line("  Try searching manually:")
            stem = Path(r["filename"]).stem.replace("-", " ").replace("_", " ")
            hf_url  = f"https://huggingface.co/models?search={urllib.parse.quote(stem)}"
            civ_url = f"https://civitai.com/models?query={urllib.parse.quote(stem)}"
            self._append_result_link("  🤗 HuggingFace search", hf_url,  f"hf_manual_{id(r)}")
            self._append_result_link("  🟠 CivitAI search",     civ_url, f"civ_manual_{id(r)}")
        else:
            # Group by source
            hf_hits  = [h for h in hits if h["source"] == "HuggingFace"]
            civ_hits = [h for h in hits if h["source"] == "CivitAI"]

            if hf_hits:
                line("  HuggingFace:")
                for i, h in enumerate(hf_hits):
                    exact = h["download_url"] is not None
                    label = f"    {'✔' if exact else '~'} {h['name']}  ({h['info']})"
                    line(label)
                    if h["download_url"]:
                        self._append_result_link(
                            "      ⬇ Download direct",
                            h["download_url"],
                            f"hf_dl_{i}_{id(r)}"
                        )
                    self._append_result_link(
                        "      🔗 Open repo page",
                        h["url"],
                        f"hf_pg_{i}_{id(r)}"
                    )

            if civ_hits:
                line("  CivitAI:")
                for i, h in enumerate(civ_hits):
                    exact = h["download_url"] is not None
                    label = f"    {'✔' if exact else '~'} {h['name']}  ({h['info']})"
                    line(label)
                    if h["download_url"]:
                        self._append_result_link(
                            "      ⬇ Download direct",
                            h["download_url"],
                            f"civ_dl_{i}_{id(r)}"
                        )
                    self._append_result_link(
                        "      🔗 Open model page",
                        h["url"],
                        f"civ_pg_{i}_{id(r)}"
                    )

        self.detail_text.configure(state="disabled")

    def _append_result_link(self, label: str, url: str, tag_id: str):
        """Insert a clickable link line that opens a URL in the browser."""
        self.detail_text.insert("end", label, tag_id)
        self.detail_text.insert("end", "\n")
        self.detail_text.tag_configure(tag_id,
            foreground="#4fc3f7", underline=True, font=("Consolas", 9))
        self.detail_text.tag_bind(tag_id, "<Button-1>",
            lambda e, u=url: self._open_url(u))
        self.detail_text.tag_bind(tag_id, "<Enter>",
            lambda e: self.detail_text.configure(cursor="hand2"))
        self.detail_text.tag_bind(tag_id, "<Leave>",
            lambda e: self.detail_text.configure(cursor=""))

    # ── Copy path ──────────────────────────────────────────────────────────────
    def _copy_path(self):
        sel = self.tree.selection()
        if not sel:
            self.status_var.set("Select a row first.")
            return
        loc_cell = str(self.tree.item(sel[0])["values"][5])
        # If stacked (multiple paths), copy only the first (C: preferred)
        first_path = loc_cell.split("\n")[0].strip()
        if first_path and first_path != "— not found —":
            self.clipboard_clear()
            self.clipboard_append(first_path)
            self.status_var.set(f"Copied: {first_path}")
        else:
            self.status_var.set("No path to copy (model not found).")

    # ── Open log ───────────────────────────────────────────────────────────────
    def _open_log(self):
        try:
            os.startfile(str(LOG_PATH))
        except Exception:
            messagebox.showinfo("Log", str(LOG_PATH))

    # ── Error dialog ───────────────────────────────────────────────────────────
    def _show_error(self, title: str, exc: Exception):
        try:
            messagebox.showerror(title,
                f"{type(exc).__name__}: {exc}\n\nSee Log tab or scanner.log")
            self.status_var.set(f"Error: {type(exc).__name__} — see Log tab")
        except Exception:
            pass


# ── Global exception hook ─────────────────────────────────────────────────────
def _exc_hook(t, v, tb):
    log.critical("Unhandled:\n" + "".join(traceback.format_exception(t, v, tb)))
    try:
        messagebox.showerror("Unexpected Error",
            f"{t.__name__}: {v}\n\nSee scanner.log\n({LOG_PATH})")
    except Exception:
        pass

sys.excepthook = _exc_hook

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info(f"Python: {sys.executable}")
    log.info(f"Log:    {LOG_PATH}")
    try:
        App().mainloop()
    except Exception as e:
        log_exc("mainloop")
        input("Press Enter to exit…")
