#!/usr/bin/env python3
"""
ComfyUI Workflow Preview Generator
Generates high-resolution detail images next to each workflow JSON.

Usage:
    python generate_previews.py C:/path/to/workflows
    python generate_previews.py C:/path/to/workflows --force   # regenerate all
    python generate_previews.py C:/path/to/workflows --workers 8
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Pillow is required:  pip install pillow")


# ── Visual constants ──────────────────────────────────────────────────────────
BG_COLOR        = (22, 22, 28)
TITLE_H         = 24.0      # world units
SLOT_H          = 22.0      # world units per slot row
WIDGET_H        = 17.0      # world units per widget row
WIDGET_PAD      = 5.0       # world units between slot area and widget area
SLOT_RADIUS     = 4.5       # world units, dot radius
PAD             = 100       # world units of whitespace around the workflow

TARGET_W        = 5120      # target output width  (actual may differ)
TARGET_H        = 3840      # target output height
MAX_SCALE       = 3.0       # never zoom in beyond 3× (tiny workflows stay sane)
MIN_SCALE       = 0.35      # never zoom out beyond 0.35× (huge workflows still legible)

# ── Node body colors ──────────────────────────────────────────────────────────
_NODE_COLORS: dict[str, tuple] = {
    "CheckpointLoaderSimple":    (74,  52,  94),
    "CheckpointLoader":          (74,  52,  94),
    "unCLIPCheckpointLoader":    (74,  52,  94),
    "CLIPTextEncode":            (48,  64,  90),
    "CLIPTextEncodeSDXL":        (48,  64,  90),
    "CLIPTextEncodeSDXLRefiner": (48,  64,  90),
    "KSampler":                  (94,  48,  48),
    "KSamplerAdvanced":          (94,  48,  48),
    "SamplerCustom":             (94,  48,  48),
    "VAEDecode":                 (48,  90,  48),
    "VAEEncode":                 (48,  80,  48),
    "VAEEncodeForInpaint":       (48,  80,  48),
    "SaveImage":                 (90,  68,  18),
    "PreviewImage":              (80,  60,  18),
    "LoadImage":                 (50,  70,  90),
    "ImageScale":                (50,  70,  80),
    "ImageScaleBy":              (50,  70,  80),
    "ImageResize+":              (50,  70,  80),
    "UpscaleModelLoader":        (50,  65,  85),
    "ImageUpscaleWithModel":     (50,  65,  85),
    "UltimateSDUpscale":         (60,  70,  95),
    "ControlNetLoader":          (55,  78, 100),
    "ControlNetApply":           (55,  78, 100),
    "ControlNetApplyAdvanced":   (55,  78, 100),
    "IPAdapterLoader":           (80,  55, 100),
    "IPAdapter":                 (80,  55, 100),
    "IPAdapterAdvanced":         (80,  55, 100),
    "LoraLoader":                (60,  90,  60),
    "LoraLoaderModelOnly":       (60,  90,  60),
    "LoraLoaderStack":           (55,  85,  55),
    "PrimitiveNode":             (38,  38,  50),
    "Note":                      (38,  55,  38),
    "EmptyLatentImage":          (42,  62,  80),
    "LatentUpscale":             (42,  72,  80),
    "LatentUpscaleBy":           (42,  72,  80),
    "CLIPLoader":                (48,  64,  90),
    "VAELoader":                 (74,  52,  94),
    "FreeU":                     (60,  75,  60),
    "FreeU_V2":                  (60,  75,  60),
    "ModelSamplingDiscrete":     (70,  60,  90),
    "ConditioningCombine":       (58,  74, 100),
    "ConditioningConcat":        (58,  74, 100),
    "ConditioningSetArea":       (58,  74, 100),
    "ConditioningSetMask":       (58,  74, 100),
    "CLIPSetLastLayer":          (48,  64,  90),
    "FluxGuidance":              (90,  55,  55),
    "InpaintModelConditioning":  (55,  80,  80),
}
_DEFAULT_NODE_COLOR = (42, 42, 52)
_TITLE_LIFT = 32   # how many channels to lighten for title bar

# ── Slot / wire colors ────────────────────────────────────────────────────────
_SLOT_COLORS: dict[str, tuple] = {
    "MODEL":         (183, 117, 255),
    "CONDITIONING":  (255, 182,  80),
    "LATENT":        (148,  80, 180),
    "IMAGE":         ( 80, 182, 100),
    "VAE":           (220, 100, 100),
    "CLIP":          (100, 182, 220),
    "INT":           (145, 200, 145),
    "FLOAT":         (145, 200, 145),
    "STRING":        (200, 200, 145),
    "MASK":          (182, 150, 100),
    "CONTROL_NET":   (100, 160, 220),
    "UPSCALE_MODEL": (160, 160, 220),
    "SAMPLER":       (220, 160, 100),
    "SIGMAS":        (200, 140, 120),
    "NOISE":         (180, 180, 200),
    "GUIDER":        (200, 150, 210),
}
_DEFAULT_SLOT_COLOR = (155, 155, 155)

# ── Widget name maps (most common node types) ─────────────────────────────────
_WIDGET_NAMES: dict[str, list[str]] = {
    "KSampler":                ["seed", "control", "steps", "cfg", "sampler", "scheduler", "denoise"],
    "KSamplerAdvanced":        ["add_noise", "seed", "control", "steps", "cfg", "sampler", "scheduler", "start_at", "end_at", "return_noise"],
    "CheckpointLoaderSimple":  ["ckpt_name"],
    "CheckpointLoader":        ["config_name", "ckpt_name"],
    "CLIPTextEncode":          ["text"],
    "CLIPTextEncodeSDXL":      ["width", "height", "crop_w", "crop_h", "target_w", "target_h", "text_g", "text_l"],
    "CLIPTextEncodeSDXLRefiner": ["ascore", "width", "height", "text"],
    "EmptyLatentImage":        ["width", "height", "batch_size"],
    "LatentUpscale":           ["upscale_method", "width", "height", "crop"],
    "LatentUpscaleBy":         ["upscale_method", "scale_by"],
    "ImageScale":              ["upscale_method", "width", "height", "crop"],
    "ImageScaleBy":            ["upscale_method", "scale_by"],
    "LoraLoader":              ["lora_name", "strength_model", "strength_clip"],
    "LoraLoaderModelOnly":     ["lora_name", "strength_model"],
    "VAELoader":               ["vae_name"],
    "ControlNetLoader":        ["control_net_name"],
    "ControlNetApply":         ["strength"],
    "ControlNetApplyAdvanced": ["strength", "start_percent", "end_percent"],
    "IPAdapterLoader":         ["ipadapter_file"],
    "IPAdapter":               ["weight", "noise", "weight_type", "start_at", "end_at", "unfold_batch"],
    "CLIPLoader":              ["clip_name", "type"],
    "SaveImage":               ["filename_prefix"],
    "PreviewImage":            [],
    "UpscaleModelLoader":      ["model_name"],
    "UltimateSDUpscale":       ["upscale_by", "seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "mode_type", "tile_width", "tile_height", "mask_blur", "tile_padding"],
    "FreeU":                   ["b1", "b2", "s1", "s2"],
    "FreeU_V2":                ["b1", "b2", "s1", "s2"],
    "CLIPSetLastLayer":        ["stop_at_clip_layer"],
    "ModelSamplingDiscrete":   ["sampling", "zsnr"],
    "FluxGuidance":            ["guidance"],
    "Note":                    ["text"],
}


# ── Font helpers ──────────────────────────────────────────────────────────────
_FONT_CACHE: dict[tuple, ImageFont.FreeTypeFont] = {}

def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key not in _FONT_CACHE:
        candidates = (
            ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else
            ["segoeui.ttf",  "arial.ttf",   "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]
        )
        for name in candidates:
            try:
                _FONT_CACHE[key] = ImageFont.truetype(name, size)
                break
            except OSError:
                pass
        else:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


# ── Color helpers ─────────────────────────────────────────────────────────────
def _hex_to_rgb(s: str) -> tuple:
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = s[0]*2 + s[1]*2 + s[2]*2
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)

def _lighten(c: tuple, amt: int) -> tuple:
    return tuple(min(255, v + amt) for v in c)

def _darken(c: tuple, amt: int) -> tuple:
    return tuple(max(0, v - amt) for v in c)

def _truncate(text: str, max_chars: int) -> str:
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= max_chars else text[:max_chars - 1] + "\u2026"


# ── Main render function ──────────────────────────────────────────────────────
def render_detailed(json_path: Path) -> "Image.Image | None":
    """
    Render a ComfyUI workflow JSON to a high-resolution PIL Image.
    Returns None if the file is not a valid workflow.
    """
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None

    if not isinstance(data, dict) or "nodes" not in data:
        return None

    nodes:  list[dict] = data.get("nodes",  [])
    links:  list[list] = data.get("links",  [])
    groups: list[dict] = data.get("groups", [])

    if not nodes:
        return None

    # ── Parse node rects ──────────────────────────────────────────────────────
    node_rects: dict[int, tuple] = {}
    for n in nodes:
        nid = n.get("id")
        pos = n.get("pos")
        if nid is None or not pos:
            continue
        x, y = float(pos[0]), float(pos[1])
        sz = n.get("size", {})
        if isinstance(sz, dict):
            w, h = float(sz.get("0", 200)), float(sz.get("1", 100))
        elif isinstance(sz, (list, tuple)) and len(sz) >= 2:
            w, h = float(sz[0]), float(sz[1])
        else:
            w, h = 200.0, 100.0
        node_rects[nid] = (x, y, w, h)

    if not node_rects:
        return None

    # ── World bounds (nodes + groups + padding) ───────────────────────────────
    xs = [r[0] for r in node_rects.values()] + [r[0]+r[2] for r in node_rects.values()]
    ys = [r[1] for r in node_rects.values()] + [r[1]+r[3] for r in node_rects.values()]
    for g in groups:
        bp = g.get("bounding") or g.get("pos", [0, 0])
        bs = g.get("size", [200, 200])
        if bp and len(bp) >= 2:
            xs += [float(bp[0]), float(bp[0]) + float(bs[0])]
            ys += [float(bp[1]), float(bp[1]) + float(bs[1])]

    min_x = min(xs) - PAD
    min_y = min(ys) - PAD
    max_x = max(xs) + PAD
    max_y = max(ys) + PAD
    world_w = max(1.0, max_x - min_x)
    world_h = max(1.0, max_y - min_y)

    scale = min(TARGET_W / world_w, TARGET_H / world_h)
    scale = max(MIN_SCALE, min(MAX_SCALE, scale))

    img_w = max(640, int(world_w * scale))
    img_h = max(480, int(world_h * scale))

    def sx(wx: float) -> int: return int((wx - min_x) * scale)
    def sy(wy: float) -> int: return int((wy - min_y) * scale)
    def sv(v: float)  -> int: return max(1, int(v * scale))

    # ── Canvas ────────────────────────────────────────────────────────────────
    img  = Image.new("RGB", (img_w, img_h), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Subtle dot-grid background
    grid_step = sv(50)
    if grid_step >= 8:
        dot_color = (30, 30, 38)
        for gx in range(0, img_w, grid_step):
            for gy in range(0, img_h, grid_step):
                draw.rectangle([gx, gy, gx+1, gy+1], fill=dot_color)

    # ── Groups ────────────────────────────────────────────────────────────────
    for g in groups:
        bp = g.get("bounding") or g.get("pos", [0, 0])
        bs = g.get("size", [200, 200])
        if not bp or len(bp) < 2:
            continue
        gx, gy = float(bp[0]), float(bp[1])
        gw, gh = float(bs[0]), float(bs[1])
        try:
            gc = _hex_to_rgb(g.get("color", "#555555"))
        except Exception:
            gc = (80, 80, 80)

        x1, y1 = sx(gx), sy(gy)
        x2, y2 = sx(gx + gw), sy(gy + gh)
        if x2 <= x1 + 4 or y2 <= y1 + 4:
            continue

        fill = _darken(gc, 55)
        border_w = max(2, sv(1.5))
        draw.rectangle([x1, y1, x2, y2], fill=fill, outline=gc, width=border_w)

        gtitle = g.get("title", "")
        if gtitle:
            fs = max(11, sv(14))
            draw.text(
                (x1 + sv(7), y1 + sv(5)),
                gtitle,
                fill=_lighten(gc, 70),
                font=_font(fs, bold=True),
            )

    # ── Pre-compute slot screen positions ─────────────────────────────────────
    out_pos: dict[tuple, tuple] = {}
    in_pos:  dict[tuple, tuple] = {}

    for n in nodes:
        nid = n.get("id")
        if nid not in node_rects:
            continue
        nx, ny, nw, _ = node_rects[nid]
        for i in range(len(n.get("outputs", []))):
            out_pos[(nid, i)] = (sx(nx + nw), sy(ny + TITLE_H + SLOT_H * (i + 0.5)))
        for i in range(len(n.get("inputs", []))):
            in_pos[(nid, i)] = (sx(nx), sy(ny + TITLE_H + SLOT_H * (i + 0.5)))

    # ── Links (bezier, shadow + main pass) ────────────────────────────────────
    lw = max(2, sv(2.2))
    for lnk in links:
        if len(lnk) < 5:
            continue
        _, src_id, src_slot, dst_id, dst_slot = lnk[:5]
        ltype = str(lnk[5]).upper() if len(lnk) > 5 else ""
        p0 = out_pos.get((src_id, src_slot))
        p1 = in_pos.get((dst_id, dst_slot))
        if not p0 or not p1:
            continue
        color = _SLOT_COLORS.get(ltype, _DEFAULT_SLOT_COLOR)
        x0, y0 = p0
        x1, y1 = p1
        dx = max(30, abs(x1 - x0) * 0.5)
        steps = max(20, int(abs(x1 - x0) / 4))
        cx0, cy0 = x0 + dx, y0
        cx1, cy1 = x1 - dx, y1
        pts = []
        for i in range(steps + 1):
            t  = i / steps
            mt = 1 - t
            bx = mt**3*x0 + 3*mt**2*t*cx0 + 3*mt*t**2*cx1 + t**3*x1
            by = mt**3*y0 + 3*mt**2*t*cy0 + 3*mt*t**2*cy1 + t**3*y1
            pts.append((int(bx), int(by)))
        shadow = _darken(color, 90)
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i+1]], fill=shadow, width=lw + sv(1.5))
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i+1]], fill=color, width=lw)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    for n in nodes:
        nid = n.get("id")
        if nid not in node_rects:
            continue
        nx, ny, nw, nh = node_rects[nid]
        ntype: str  = n.get("type", "")
        title: str  = n.get("title") or ntype
        muted: bool = n.get("mode", 0) == 2

        x1, y1 = sx(nx),      sy(ny)
        x2, y2 = sx(nx + nw), sy(ny + nh)
        if x2 - x1 < 6 or y2 - y1 < 6:
            continue

        # ── Node colors ───────────────────────────────────────────────────────
        if n.get("bgcolor"):
            try:   bg = _hex_to_rgb(n["bgcolor"])
            except Exception: bg = _NODE_COLORS.get(ntype, _DEFAULT_NODE_COLOR)
        else:
            bg = _NODE_COLORS.get(ntype, _DEFAULT_NODE_COLOR)
        if muted:
            bg = _darken(bg, 45)

        if n.get("color"):
            try:   title_bg = _hex_to_rgb(n["color"])
            except Exception: title_bg = _lighten(bg, _TITLE_LIFT)
        else:
            title_bg = _lighten(bg, _TITLE_LIFT)

        border_col = _lighten(bg, 18)

        # ── Body rect ─────────────────────────────────────────────────────────
        draw.rectangle([x1, y1, x2, y2], fill=bg, outline=border_col, width=max(1, sv(1)))

        # ── Title bar ─────────────────────────────────────────────────────────
        title_y2 = sy(ny + TITLE_H)
        if y1 < title_y2 <= y2:
            draw.rectangle([x1, y1, x2, title_y2], fill=title_bg)
            fs_t  = max(9, sv(11))
            avail = max(4, (x2 - x1 - sv(8)) // max(1, fs_t // 2))
            draw.text(
                (x1 + sv(6), y1 + sv(4)),
                _truncate(title, avail),
                fill=(230, 230, 230),
                font=_font(fs_t, bold=True),
            )
            # Dim node-type tag in the far right of title bar if title != ntype
            if n.get("title") and n["title"] != ntype:
                fs_tag = max(7, sv(8))
                tag    = _truncate(ntype, 22)
                tag_bb = draw.textbbox((0, 0), tag, font=_font(fs_tag))
                tag_w  = tag_bb[2] - tag_bb[0]
                tx     = x2 - tag_w - sv(5)
                ty     = y1 + sv(5)
                if tx > x1 + sv(60):
                    draw.text((tx, ty), tag, fill=(160, 160, 175), font=_font(fs_tag))

        # ── Slot dots + names ─────────────────────────────────────────────────
        fs_s  = max(7, sv(9))
        font_s = _font(fs_s)
        dr     = max(3, sv(SLOT_RADIUS))
        node_w_px = x2 - x1

        for i, inp in enumerate(n.get("inputs", [])):
            px, py = in_pos.get((nid, i), (0, 0))
            if px == 0 and py == 0:
                continue
            itype = (inp.get("type", "") if isinstance(inp, dict) else "").upper()
            iname =  inp.get("name", "")  if isinstance(inp, dict) else ""
            c = _SLOT_COLORS.get(itype, _DEFAULT_SLOT_COLOR)
            draw.ellipse([px-dr, py-dr, px+dr, py+dr], fill=c, outline=_lighten(c, 50), width=1)
            if iname and node_w_px > sv(45):
                draw.text((px + dr + sv(4), py - fs_s // 2), iname, fill=(185, 185, 200), font=font_s)

        for i, out in enumerate(n.get("outputs", [])):
            px, py = out_pos.get((nid, i), (0, 0))
            if px == 0 and py == 0:
                continue
            otype = (out.get("type", "") if isinstance(out, dict) else "").upper()
            oname =  out.get("name", "")  if isinstance(out, dict) else ""
            c = _SLOT_COLORS.get(otype, _DEFAULT_SLOT_COLOR)
            draw.ellipse([px-dr, py-dr, px+dr, py+dr], fill=c, outline=_lighten(c, 50), width=1)
            if oname and node_w_px > sv(45):
                bb  = draw.textbbox((0, 0), oname, font=font_s)
                tw  = bb[2] - bb[0]
                draw.text((px - dr - sv(4) - tw, py - fs_s // 2), oname, fill=(185, 185, 200), font=font_s)

        # ── Widget values ─────────────────────────────────────────────────────
        widget_vals  = n.get("widgets_values", [])
        widget_names = _WIDGET_NAMES.get(ntype, [])
        if not widget_vals:
            continue

        num_in  = len(n.get("inputs",  []))
        num_out = len(n.get("outputs", []))
        slots_end_wy = ny + TITLE_H + max(num_in, num_out) * SLOT_H + WIDGET_PAD
        slots_end_py = sy(slots_end_wy)
        body_end_py  = y2 - sv(4)

        if slots_end_py >= body_end_py:
            continue

        fs_w      = max(7, sv(9))
        fs_wk     = max(7, sv(8))
        font_w    = _font(fs_w)
        font_wk   = _font(fs_wk)
        avail_w   = node_w_px - sv(14)    # inner width for text
        row_h     = sv(WIDGET_H)
        max_rows  = max(1, (body_end_py - slots_end_py) // max(1, row_h))

        displayed = 0
        for wi, wval in enumerate(widget_vals):
            if displayed >= max_rows:
                break
            wname = widget_names[wi] if wi < len(widget_names) else f"val{wi}"

            # Format value
            if isinstance(wval, float):
                vstr = f"{wval:.5g}"
            else:
                vstr = str(wval)

            # Build "key: " prefix
            key_text = f"{wname}: "
            key_bb   = draw.textbbox((0, 0), key_text, font=font_wk)
            key_w    = key_bb[2] - key_bb[0]
            val_max  = max(4, (avail_w - key_w) // max(1, fs_w // 2))
            vstr_t   = _truncate(vstr, val_max)

            row_y = slots_end_py + displayed * row_h
            if row_y + fs_w > body_end_py:
                break

            draw.text((x1 + sv(7), row_y), key_text, fill=(115, 115, 145), font=font_wk)
            draw.text((x1 + sv(7) + key_w, row_y), vstr_t, fill=(215, 205, 175), font=font_w)
            displayed += 1

        # If there are more values than we could show, add "…" hint
        if displayed < len(widget_vals) and displayed == max_rows:
            hint_y = slots_end_py + displayed * row_h - row_h
            draw.text(
                (x2 - sv(14), hint_y),
                "\u2026",
                fill=(120, 120, 150),
                font=_font(fs_w),
            )

    return img


# ── File processing ───────────────────────────────────────────────────────────
def _process(json_path: Path, force: bool) -> tuple[Path, str]:
    out = json_path.with_suffix(".png")
    if not force and out.exists() and out.stat().st_mtime >= json_path.stat().st_mtime:
        return json_path, "skip"
    img = render_detailed(json_path)
    if img is None:
        return json_path, "not_workflow"
    try:
        img.save(out, "PNG")
    except Exception as exc:
        return json_path, f"error: {exc}"
    return json_path, "ok"


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate high-res ComfyUI workflow preview images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("folder", help="Root folder to scan (recursive)")
    ap.add_argument("--force", "-f", action="store_true",
                    help="Re-render even if a PNG already exists and is up to date")
    ap.add_argument("--workers", "-w", type=int, default=4,
                    help="Parallel worker threads (default: 4)")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"Not a directory: {folder}")

    files = sorted(folder.rglob("*.json"))
    # Exclude any JSON files that are themselves generated (none are .json→.png)
    if not files:
        print("No JSON files found.")
        return

    total = len(files)
    print(f"Scanning {folder}")
    print(f"Found {total} JSON file{'s' if total != 1 else ''}\n")

    ok = skip = not_wf = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_process, f, args.force): f for f in files}
        done = 0
        for fut in as_completed(futs):
            done += 1
            path, status = fut.result()
            rel = path.relative_to(folder)
            if status == "ok":
                ok += 1
                print(f"[{done:>{len(str(total))}}/{total}]  rendered   {rel}")
            elif status == "skip":
                skip += 1
                # Only show skips if verbose; keep output clean
            elif status == "not_workflow":
                not_wf += 1
            else:
                err += 1
                print(f"[{done:>{len(str(total))}}/{total}]  ERROR      {rel}  ({status})")

    print(f"\nDone.")
    if ok:        print(f"  {ok} image{'s' if ok != 1 else ''} rendered")
    if skip:      print(f"  {skip} already up to date (use --force to re-render)")
    if not_wf:    print(f"  {not_wf} skipped (not ComfyUI workflows)")
    if err:       print(f"  {err} error{'s' if err != 1 else ''}")


if __name__ == "__main__":
    main()
