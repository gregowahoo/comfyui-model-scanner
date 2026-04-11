"""
ComfyUI Workflow Manager
Browse, search, and preview ComfyUI workflow JSON files.

Run:   python workflow_manager.py [optional_folder]
"""

import json
import sys
import hashlib
import threading
import queue
from pathlib import Path
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Pillow is required:  pip install Pillow")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from workflow_preview.renderer import render_workflow

# ── Geometry ───────────────────────────────────────────────────────────────────
THUMB_W  = 220
THUMB_H  = 155
CELL_PAD = 10
LABEL_H  = 38
SUBDIR_H = 14
CELL_W   = THUMB_W + CELL_PAD * 2
CELL_H   = THUMB_H + LABEL_H + SUBDIR_H + CELL_PAD * 2
MIN_COLS = 2
TREE_W   = 240

# ── Persistence ────────────────────────────────────────────────────────────────
_APP_DIR    = Path.home() / ".comfyui_workflows"
CACHE_DIR   = _APP_DIR / "thumbnails"
CONFIG_FILE = _APP_DIR / "config.json"

# ── Palette ────────────────────────────────────────────────────────────────────
BG         = "#1e1e28"
BG_PANEL   = "#16161f"
BG_CELL    = "#26263a"
BG_HOVER   = "#2e2e44"
BG_SEL     = "#383860"
FG         = "#d4d4e8"
FG_DIM     = "#6e6e8a"
ACCENT     = "#7878cc"
BORDER     = "#383850"
BORDER_SEL = "#8080d0"


# ══════════════════════════════════════════════════════════════════════════════
# Thumbnail cache
# ══════════════════════════════════════════════════════════════════════════════

class ThumbnailCache:
    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _key(self, path: Path) -> str:
        mtime = int(path.stat().st_mtime)
        return hashlib.md5(f"{path}|{mtime}|{THUMB_W}x{THUMB_H}".encode()).hexdigest()

    def get(self, path: Path) -> Image.Image | None:
        try:
            cp = CACHE_DIR / f"{self._key(path)}.png"
            if cp.exists():
                return Image.open(cp).convert("RGB")
        except Exception:
            pass
        return None

    def put(self, path: Path, img: Image.Image):
        import io
        try:
            buf = io.BytesIO()
            img.save(buf, "PNG", optimize=True)
            (CACHE_DIR / f"{self._key(path)}.png").write_bytes(buf.getvalue())
        except Exception:
            pass

    def render(self, path: Path) -> Image.Image | None:
        cached = self.get(path)
        if cached:
            return cached
        full = render_workflow(path, THUMB_W * 2, THUMB_H * 2)
        if full is None:
            return None
        thumb = full.resize((THUMB_W, THUMB_H), Image.Resampling.LANCZOS)
        self.put(path, thumb)
        return thumb


_cache = ThumbnailCache()


# ══════════════════════════════════════════════════════════════════════════════
# Folder tree panel
# ══════════════════════════════════════════════════════════════════════════════

class FolderTree(tk.Frame):
    """Collapsible/expandable folder tree."""

    def __init__(self, parent, on_select):
        super().__init__(parent, bg=BG_PANEL, width=TREE_W)
        self.pack_propagate(False)
        self._on_select = on_select   # callback(folder: Path)
        self._root: Path | None = None
        self._build()

    # ── Build ─────────────────────────────────────────────────────────────────
    def _build(self):
        hdr = tk.Frame(self, bg=BG_PANEL, pady=7)
        hdr.pack(fill="x")
        tk.Label(hdr, text="FOLDERS", bg=BG_PANEL, fg=FG_DIM,
                 font=("Segoe UI", 8, "bold"), padx=10).pack(side="left")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        container = tk.Frame(self, bg=BG_PANEL)
        container.pack(fill="both", expand=True)

        self._tree = ttk.Treeview(container, style="FolderTree.Treeview",
                                   show="tree", selectmode="browse")
        vsb = ttk.Scrollbar(container, orient="vertical",
                             command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._tree.pack(side="left", fill="both", expand=True)

        self._tree.column("#0", width=TREE_W - 18, minwidth=120)
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self._tree.bind("<Double-Button-1>", self._on_double)

    # ── Public ────────────────────────────────────────────────────────────────
    def load(self, root: Path):
        self._root = root
        self._tree.delete(*self._tree.get_children())

        total = sum(1 for _ in root.rglob("*.json"))
        root_node = self._tree.insert(
            "", "end",
            text=f"\u2302  {root.name}  ({total})",
            values=[str(root)],
            open=True,
        )
        self._populate(root_node, root)

        # Select root (shows all)
        self._tree.selection_set(root_node)
        self._tree.focus(root_node)

    # ── Internals ─────────────────────────────────────────────────────────────
    def _populate(self, parent_node: str, folder: Path):
        try:
            subdirs = sorted(
                (d for d in folder.iterdir() if d.is_dir() and not d.name.startswith(".")),
                key=lambda d: d.name.lower(),
            )
        except PermissionError:
            return

        for d in subdirs:
            total = sum(1 for _ in d.rglob("*.json"))
            if total == 0:
                continue
            direct = sum(1 for _ in d.glob("*.json"))
            has_sub = total > direct
            cnt = f"{direct}+{total - direct}" if has_sub else str(direct)
            node = self._tree.insert(
                parent_node, "end",
                text=f"\u25b8  {d.name}  ({cnt})",
                values=[str(d)],
            )
            self._populate(node, d)

    def _on_tree_select(self, _event):
        sel = self._tree.selection()
        if not sel:
            return
        vals = self._tree.item(sel[0], "values")
        if vals:
            self._on_select(Path(vals[0]))

    def _on_double(self, event):
        """Toggle expand/collapse on double-click."""
        node = self._tree.identify_row(event.y)
        if node:
            self._tree.item(node, open=not self._tree.item(node, "open"))


# ══════════════════════════════════════════════════════════════════════════════
# Scrollable thumbnail grid
# ══════════════════════════════════════════════════════════════════════════════

class WorkflowGrid(tk.Frame):

    def __init__(self, parent, on_select, on_open):
        super().__init__(parent, bg=BG)
        self._on_select = on_select
        self._on_open   = on_open

        self._items:    list[Path]                     = []
        self._root_dir: Path | None                    = None
        self._photos:   dict[Path, ImageTk.PhotoImage] = {}
        self._failed:   set[Path]                      = set()
        self._selected: int | None = None
        self._hover:    int | None = None
        self._cols = 4

        self._render_q:    queue.Queue     = queue.Queue()
        self._cancel_flag: threading.Event = threading.Event()

        self._build()
        threading.Thread(target=self._render_worker, daemon=True).start()

    def _build(self):
        self._canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self._canvas.bind("<Configure>",       self._on_configure)
        self._canvas.bind("<Button-1>",        self._on_click)
        self._canvas.bind("<Double-Button-1>", self._on_dbl)
        self._canvas.bind("<Motion>",          self._on_motion)
        self._canvas.bind("<Leave>",           self._on_leave)
        self._canvas.bind("<MouseWheel>",      self._on_wheel)
        self._canvas.bind("<Return>",          lambda _e: self._open_selected())
        self._canvas.bind("<Up>",    lambda _e: self._move(-self._cols))
        self._canvas.bind("<Down>",  lambda _e: self._move(+self._cols))
        self._canvas.bind("<Left>",  lambda _e: self._move(-1))
        self._canvas.bind("<Right>", lambda _e: self._move(+1))

    # ── Public ────────────────────────────────────────────────────────────────
    def set_items(self, paths: list[Path], root_dir: Path | None = None):
        self._cancel_flag.set()
        self._cancel_flag = threading.Event()
        self._items    = list(paths)
        self._root_dir = root_dir
        self._photos   = {}
        self._selected = None
        self._hover    = None
        self._redraw_all()
        for p in paths:
            self._render_q.put((p, self._cancel_flag))

    # ── Geometry ──────────────────────────────────────────────────────────────
    def _cols_for(self, w: int) -> int:
        return max(MIN_COLS, (w - CELL_PAD) // (CELL_W + CELL_PAD))

    def _cell_rect(self, idx: int) -> tuple[int, int, int, int]:
        col, row = idx % self._cols, idx // self._cols
        x = CELL_PAD + col * (CELL_W + CELL_PAD)
        y = CELL_PAD + row * (CELL_H + CELL_PAD)
        return x, y, x + CELL_W, y + CELL_H

    def _idx_at(self, cx: float, cy: float) -> int | None:
        ay  = cy + self._canvas.canvasy(0)
        col = int((cx - CELL_PAD) // (CELL_W + CELL_PAD))
        row = int((ay - CELL_PAD) // (CELL_H + CELL_PAD))
        idx = row * self._cols + col
        if 0 <= col < self._cols and 0 <= idx < len(self._items):
            x0, y0, x1, y1 = self._cell_rect(idx)
            if x0 <= cx <= x1 and y0 <= ay <= y1:
                return idx
        return None

    def _scroll_region(self):
        rows = max(1, (len(self._items) + self._cols - 1) // self._cols)
        return (0, 0, self._canvas.winfo_width(),
                rows * (CELL_H + CELL_PAD) + CELL_PAD)

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _redraw_all(self):
        self._canvas.delete("all")
        if not self._items:
            self._canvas.create_text(
                max(200, self._canvas.winfo_width() // 2), 100,
                text="No workflows found in this folder.",
                fill=FG_DIM, font=("Segoe UI", 13), justify="center",
            )
            self._canvas.configure(scrollregion=(0, 0, 1, 1))
            return
        self._canvas.configure(scrollregion=self._scroll_region())
        for idx, path in enumerate(self._items):
            self._draw_cell(idx, path)

    def _draw_cell(self, idx: int, path: Path):
        c  = self._canvas
        x0, y0, x1, y1 = self._cell_rect(idx)
        tag = f"c{idx}"

        if idx == self._selected:
            bg, bd, bw = BG_SEL,   BORDER_SEL, 2
        elif idx == self._hover:
            bg, bd, bw = BG_HOVER, BORDER,     1
        else:
            bg, bd, bw = BG_CELL,  BORDER,     1

        c.create_rectangle(x0, y0, x1, y1,
                           fill=bg, outline=bd, width=bw, tags=tag)

        tx, ty = x0 + CELL_PAD, y0 + CELL_PAD

        if path in self._photos:
            c.create_image(tx, ty, anchor="nw",
                           image=self._photos[path], tags=tag)
        else:
            c.create_rectangle(tx, ty, tx + THUMB_W, ty + THUMB_H,
                               fill="#1c1c2c", outline="", tags=tag)
            lbl = "not a workflow" if path in self._failed else "\u2026"
            c.create_text(tx + THUMB_W // 2, ty + THUMB_H // 2,
                          text=lbl, fill="#555566" if path in self._failed else FG_DIM,
                          font=("Segoe UI", 9), tags=tag)

        fg_lbl = "#c0c0ff" if idx == self._selected else FG
        c.create_text(
            x0 + CELL_W // 2, y0 + CELL_PAD + THUMB_H + 5,
            text=path.stem, fill=fg_lbl,
            font=("Segoe UI", 9), width=CELL_W - 8,
            anchor="n", justify="center", tags=tag,
        )

        # Subfolder badge
        if self._root_dir and path.parent != self._root_dir:
            try:
                badge = str(path.parent.relative_to(self._root_dir))
            except ValueError:
                badge = path.parent.name
            c.create_text(
                x0 + CELL_W // 2, y1 - 4,
                text=badge, fill="#5a5a80",
                font=("Segoe UI", 8), width=CELL_W - 8,
                anchor="s", justify="center", tags=tag,
            )

    def _refresh(self, idx: int):
        if 0 <= idx < len(self._items):
            self._canvas.delete(f"c{idx}")
            self._draw_cell(idx, self._items[idx])

    # ── Events ────────────────────────────────────────────────────────────────
    def _on_configure(self, event):
        self._cols = self._cols_for(event.width)
        self._redraw_all()

    def _on_click(self, event):
        self._canvas.focus_set()
        idx = self._idx_at(event.x, event.y)
        if idx is None:
            return
        old, self._selected = self._selected, idx
        if old is not None: self._refresh(old)
        self._refresh(idx)
        self._on_select(self._items[idx])

    def _on_dbl(self, event):
        idx = self._idx_at(event.x, event.y)
        if idx is not None:
            self._on_open(self._items[idx])

    def _on_motion(self, event):
        idx = self._idx_at(event.x, event.y)
        if idx == self._hover:
            return
        old, self._hover = self._hover, idx
        if old is not None: self._refresh(old)
        if idx is not None: self._refresh(idx)

    def _on_leave(self, _e):
        old, self._hover = self._hover, None
        if old is not None: self._refresh(old)

    def _on_wheel(self, event):
        self._canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def _move(self, delta: int):
        n = len(self._items)
        if not n:
            return
        cur = self._selected if self._selected is not None else -1
        nxt = max(0, min(n - 1, cur + delta))
        if nxt == self._selected:
            return
        old, self._selected = self._selected, nxt
        if old is not None: self._refresh(old)
        self._refresh(nxt)
        self._on_select(self._items[nxt])
        x0, y0, *_ = self._cell_rect(nxt)
        sr = self._scroll_region()
        self._canvas.yview_moveto(max(0, (y0 - CELL_PAD) / max(1, sr[3])))

    def _open_selected(self):
        if self._selected is not None:
            self._on_open(self._items[self._selected])

    # ── Background render ─────────────────────────────────────────────────────
    def _render_worker(self):
        while True:
            try:
                path, token = self._render_q.get(timeout=1)
            except queue.Empty:
                continue
            try:
                if token.is_set() or path not in self._items:
                    continue
                if path in self._photos or path in self._failed:
                    continue
                img = _cache.render(path)
                if token.is_set():
                    continue
                if img:
                    photo = ImageTk.PhotoImage(img)
                    self.after(0, lambda p=path, ph=photo, t=token:
                               self._thumb_done(p, ph, t))
                else:
                    self.after(0, lambda p=path, t=token:
                               self._thumb_fail(p, t))
            except Exception:
                pass
            finally:
                self._render_q.task_done()

    def _thumb_done(self, path, photo, token):
        if token.is_set() or path not in self._items:
            return
        self._photos[path] = photo
        try:
            self._refresh(self._items.index(path))
        except ValueError:
            pass

    def _thumb_fail(self, path, token):
        if token.is_set():
            return
        self._failed.add(path)
        try:
            self._refresh(self._items.index(path))
        except ValueError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Right-side preview pane
# ══════════════════════════════════════════════════════════════════════════════

class PreviewPane(tk.Frame):
    W = 370

    def __init__(self, parent, on_open):
        super().__init__(parent, bg=BG_PANEL, width=self.W)
        self.pack_propagate(False)
        self._on_open  = on_open
        self._path:    Path | None        = None
        self._pil_img: Image.Image | None = None
        self._photo    = None
        self._build()

    def _build(self):
        self._name_var = tk.StringVar(value="Select a workflow")
        tk.Label(self, textvariable=self._name_var,
                 bg=BG_PANEL, fg=FG,
                 font=("Segoe UI", 10, "bold"),
                 wraplength=self.W - 24, justify="left",
                 padx=14, pady=10, anchor="w").pack(fill="x")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        img_h = int(self.W * 0.72)
        self._img_cv = tk.Canvas(self, bg="#1a1a25",
                                  highlightthickness=0, height=img_h)
        self._img_cv.pack(fill="x")
        self._img_cv.bind("<Configure>", lambda _e: self.after(30, self._repaint))
        self._img_cv.create_text(self.W // 2, img_h // 2,
                                  text="No workflow selected",
                                  fill=FG_DIM, font=("Segoe UI", 11))

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        mf = tk.Frame(self, bg=BG_PANEL)
        mf.pack(fill="x", padx=14, pady=10)
        self._meta: dict[str, tk.StringVar] = {}
        for label in ("Folder", "Size", "Modified", "Nodes", "Links"):
            row = tk.Frame(mf, bg=BG_PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{label}:", bg=BG_PANEL, fg=FG_DIM,
                     font=("Segoe UI", 9), width=9, anchor="w").pack(side="left")
            v = tk.StringVar(value="\u2014")
            tk.Label(row, textvariable=v, bg=BG_PANEL, fg=FG,
                     font=("Segoe UI", 9), anchor="w").pack(side="left")
            self._meta[label] = v

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        self._open_btn = tk.Button(
            self, text="\u29c9  Open Full View",
            command=self._open,
            bg="#38385a", fg=FG, relief="flat",
            padx=16, pady=7, cursor="hand2",
            activebackground=ACCENT,
            font=("Segoe UI", 10), state="disabled",
        )
        self._open_btn.pack(fill="x", padx=14, pady=12)

    def show(self, path: Path):
        self._path    = path
        self._pil_img = None
        self._name_var.set(path.stem)
        self._open_btn.configure(state="normal")
        self._meta["Folder"].set(path.parent.name)
        st = path.stat()
        self._meta["Size"].set(f"{st.st_size // 1024:,} KB  ({st.st_size:,} bytes)")
        self._meta["Modified"].set(
            datetime.fromtimestamp(st.st_mtime).strftime("%b %d, %Y  %H:%M"))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._meta["Nodes"].set(str(len(data.get("nodes", []))))
            self._meta["Links"].set(str(len(data.get("links", []))))
        except Exception:
            self._meta["Nodes"].set("\u2014")
            self._meta["Links"].set("\u2014")
        cw = max(1, self._img_cv.winfo_width())
        ch = max(1, self._img_cv.winfo_height())
        self._img_cv.delete("all")
        self._img_cv.create_text(cw // 2, ch // 2,
                                  text="Rendering\u2026",
                                  fill=FG_DIM, font=("Segoe UI", 10))
        threading.Thread(target=self._bg_render, args=(path,), daemon=True).start()

    def _bg_render(self, path: Path):
        img = _cache.render(path)
        if self._path == path:
            self.after(0, lambda: self._show(img))

    def _show(self, img: Image.Image | None):
        self._img_cv.delete("all")
        if img is None:
            cw, ch = max(1, self._img_cv.winfo_width()), max(1, self._img_cv.winfo_height())
            self._img_cv.create_text(cw // 2, ch // 2,
                                      text="Not a ComfyUI workflow",
                                      fill=FG_DIM, font=("Segoe UI", 10))
            return
        self._pil_img = img
        self._repaint()

    def _repaint(self):
        if self._pil_img is None:
            return
        cw = max(1, self._img_cv.winfo_width())
        ch = max(1, self._img_cv.winfo_height())
        iw, ih = self._pil_img.size
        sc     = min(cw / iw, ch / ih)
        nw, nh = max(1, int(iw * sc)), max(1, int(ih * sc))
        scaled = self._pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(scaled)
        self._img_cv.delete("all")
        self._img_cv.create_image((cw - nw) // 2, (ch - nh) // 2,
                                   anchor="nw", image=self._photo)

    def _open(self):
        if self._path:
            self._on_open(self._path)


# ══════════════════════════════════════════════════════════════════════════════
# Main window
# ══════════════════════════════════════════════════════════════════════════════

class WorkflowManager(tk.Tk):

    def __init__(self, initial_folder: str | None = None):
        super().__init__()
        self.title("ComfyUI Workflow Manager")
        self.geometry("1600x900")
        self.minsize(1000, 600)
        self.configure(bg=BG)
        self.option_add("*tearOff", False)

        self._all:             list[Path]  = []
        self._selected_folder: Path | None = None
        self._cfg = self._load_cfg()

        self._setup_style()
        self._build()

        start = initial_folder or self._cfg.get("last_folder")
        if start and Path(start).is_dir():
            self.after(150, lambda: self._load_folder(start))

    # ── Style ─────────────────────────────────────────────────────────────────
    def _setup_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        # Scrollbars
        s.configure("TScrollbar",
                    background="#2a2a3a", troughcolor="#1a1a26",
                    arrowcolor=FG_DIM, borderwidth=0, relief="flat")
        # Combobox
        s.configure("TCombobox",
                    fieldbackground="#282840", background="#282840",
                    foreground=FG, selectbackground=ACCENT, selectforeground=FG)
        s.map("TCombobox",
              fieldbackground=[("readonly", "#282840")],
              foreground=[("readonly", FG)])
        # Folder tree
        s.configure("FolderTree.Treeview",
                    background=BG_PANEL, foreground=FG,
                    fieldbackground=BG_PANEL,
                    borderwidth=0, relief="flat",
                    rowheight=26, font=("Segoe UI", 9),
                    indent=14)
        s.map("FolderTree.Treeview",
              background=[("selected", BG_SEL)],
              foreground=[("selected", "#c0c0ff")])
        s.layout("FolderTree.Treeview", [
            ("FolderTree.Treeview.treearea", {"sticky": "nswe"})
        ])

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build(self):
        # Toolbar
        bar = tk.Frame(self, bg=BG_PANEL, pady=7)
        bar.pack(fill="x")

        tk.Button(bar, text="\U0001f4c1  Browse\u2026",
                  command=self._browse,
                  bg="#38385a", fg=FG, relief="flat",
                  padx=14, pady=5, cursor="hand2",
                  activebackground=ACCENT,
                  font=("Segoe UI", 10)).pack(side="left", padx=(10, 0))

        self._folder_var = tk.StringVar(value="No folder selected")
        tk.Label(bar, textvariable=self._folder_var,
                 bg=BG_PANEL, fg=FG_DIM,
                 font=("Segoe UI", 9)).pack(side="left", padx=12)

        tk.Label(bar, text="Sort:", bg=BG_PANEL, fg=FG_DIM,
                 font=("Segoe UI", 9)).pack(side="right", padx=(0, 4))
        self._sort_var = tk.StringVar(value="Date \u2193")
        cb = ttk.Combobox(bar, textvariable=self._sort_var, state="readonly",
                          values=["Date \u2193", "Date \u2191",
                                  "Name A\u2013Z", "Name Z\u2013A", "Size \u2193"],
                          width=11, font=("Segoe UI", 9))
        cb.pack(side="right", padx=(0, 12))
        cb.bind("<<ComboboxSelected>>", lambda _: self._apply_filter())

        tk.Label(bar, text="\U0001f50d", bg=BG_PANEL, fg=FG_DIM,
                 font=("Segoe UI", 11)).pack(side="right", padx=(0, 2))
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._apply_filter())
        ent = tk.Entry(bar, textvariable=self._search_var,
                       bg="#282840", fg=FG, insertbackground=FG,
                       relief="flat", font=("Segoe UI", 10), width=24)
        ent.pack(side="right", padx=(0, 6), ipady=4)
        ent.bind("<Escape>", lambda _: self._search_var.set(""))

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # Three-panel content
        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True)

        # Left: folder tree
        self._ftree = FolderTree(content, self._on_folder_select)
        self._ftree.pack(side="left", fill="y")

        tk.Frame(content, bg=BORDER, width=1).pack(side="left", fill="y")

        # Centre: grid
        self._grid = WorkflowGrid(content, self._on_select, self._on_open)
        self._grid.pack(side="left", fill="both", expand=True)

        tk.Frame(content, bg=BORDER, width=1).pack(side="left", fill="y")

        # Right: preview pane
        self._pane = PreviewPane(content, self._on_open)
        self._pane.pack(side="right", fill="y")

        # Status bar
        tk.Frame(self, bg="#12121c", height=1).pack(fill="x")
        self._status_var = tk.StringVar(value="Ready \u2014 click Browse to open a folder")
        tk.Label(self, textvariable=self._status_var,
                 bg="#12121c", fg=FG_DIM,
                 font=("Segoe UI", 8), anchor="w",
                 padx=10, pady=3).pack(fill="x")

    # ── Folder logic ──────────────────────────────────────────────────────────
    def _browse(self):
        d = filedialog.askdirectory(
            title="Select ComfyUI workflows folder",
            initialdir=self._cfg.get("last_folder"),
        )
        if d:
            self._load_folder(d)

    def _load_folder(self, folder: str):
        p = Path(folder)
        self._folder_var.set(str(p))
        self._cfg["last_folder"] = str(p)
        self._save_cfg()
        self._all = sorted(p.rglob("*.json"),
                           key=lambda f: f.stat().st_mtime, reverse=True)
        # Populate tree (tree select fires _on_folder_select → _apply_filter)
        self._ftree.load(p)

    def _on_folder_select(self, folder: Path):
        self._selected_folder = folder
        self._apply_filter()

    def _apply_filter(self):
        root = Path(self._cfg.get("last_folder", "") or ".")

        # Scope to selected folder
        if self._selected_folder:
            source = [p for p in self._all
                      if p.is_relative_to(self._selected_folder)]
        else:
            source = list(self._all)

        q = self._search_var.get().lower().strip()
        items = [p for p in source if q in p.name.lower()] if q else source

        key = self._sort_var.get()
        if   "Date \u2191" in key: items.sort(key=lambda p: p.stat().st_mtime)
        elif "Name A"       in key: items.sort(key=lambda p: p.name.lower())
        elif "Name Z"       in key: items.sort(key=lambda p: p.name.lower(), reverse=True)
        elif "Size"         in key: items.sort(key=lambda p: p.stat().st_size, reverse=True)

        self._grid.set_items(items, root_dir=root if root.is_dir() else None)
        n, t = len(items), len(source)
        folder_name = self._selected_folder.name if self._selected_folder else "all"
        self._status_var.set(
            f'{n} of {t} in "{folder_name}"  (search: "{q}")' if q
            else f'{t} workflows in "{folder_name}"  \u2022  double-click to open'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _on_select(self, path: Path):
        self._pane.show(path)

    def _on_open(self, path: Path):
        import subprocess
        viewer = Path(__file__).parent / "workflow_preview" / "viewer.py"
        subprocess.Popen([sys.executable, str(viewer), str(path)])

    # ── Config ────────────────────────────────────────────────────────────────
    def _load_cfg(self) -> dict:
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_cfg(self):
        _APP_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(self._cfg, indent=2), encoding="utf-8")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    WorkflowManager(folder).mainloop()
