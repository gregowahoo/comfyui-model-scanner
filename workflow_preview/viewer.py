"""
ComfyUI Workflow Viewer - standalone test tool.
Drag-and-drop a workflow JSON onto this window, or use File > Open.

Run:  python viewer.py [optional_workflow.json]
"""

import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Pillow is required: pip install Pillow")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from renderer import render_workflow


class WorkflowViewer(tk.Tk):
    def __init__(self, initial_path: str | None = None):
        super().__init__()
        self.title("ComfyUI Workflow Viewer")
        self.geometry("1100x800")
        self.configure(bg="#1c1c23")

        self._setup_menu()
        self._setup_canvas()

        self.current_image: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None

        if initial_path:
            self.after(50, lambda: self._load(initial_path))

        self.bind("<Configure>", self._on_resize)

    def _setup_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open…", accelerator="Ctrl+O", command=self._open)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)
        self.bind("<Control-o>", lambda _e: self._open())

    def _setup_canvas(self):
        self.canvas = tk.Canvas(self, bg="#1c1c23", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._drag_start)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self._drag_x = self._drag_y = 0
        self._offset_x = self._offset_y = 0

        self._label = self.canvas.create_text(
            10, 10, anchor="nw",
            text="Open a ComfyUI workflow JSON via File > Open  (Ctrl+O)",
            fill="#666677", font=("Segoe UI", 12),
        )

    def _open(self):
        path = filedialog.askopenfilename(
            title="Open ComfyUI Workflow",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self._load(path)

    def _load(self, path: str):
        self.title(f"ComfyUI Workflow Viewer — {Path(path).name}")
        cw = max(400, self.canvas.winfo_width())
        ch = max(300, self.canvas.winfo_height())
        img = render_workflow(path, cw * 2, ch * 2)  # render at 2× for sharpness
        if img is None:
            messagebox.showerror(
                "Not a ComfyUI workflow",
                f"{Path(path).name}\n\nThis JSON file does not appear to be a ComfyUI workflow.",
            )
            return
        self.current_image = img
        self._offset_x = self._offset_y = 0
        self._redraw()

    def _redraw(self):
        if self.current_image is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        img = self.current_image.resize((cw, ch), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def _on_resize(self, event):
        if self.current_image:
            job = getattr(self, "_resize_job", None)
            if job is not None:
                self.after_cancel(job)
            self._resize_job = self.after(120, self._redraw)

    def _drag_start(self, event):
        self._drag_x, self._drag_y = event.x, event.y

    def _drag_move(self, event):
        dx = event.x - self._drag_x
        dy = event.y - self._drag_y
        self._drag_x, self._drag_y = event.x, event.y
        self.canvas.move("all", dx, dy)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    WorkflowViewer(path).mainloop()
