"""
ComfyUI Workflow Renderer
Converts a ComfyUI workflow JSON file into a PIL Image.
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# Default node background colors by type
_NODE_COLORS: dict[str, tuple[int, int, int]] = {
    "CheckpointLoaderSimple": (74, 52, 94),
    "CheckpointLoader": (74, 52, 94),
    "unCLIPCheckpointLoader": (74, 52, 94),
    "CLIPTextEncode": (48, 64, 90),
    "CLIPTextEncodeSDXL": (48, 64, 90),
    "CLIPTextEncodeSDXLRefiner": (48, 64, 90),
    "KSampler": (94, 48, 48),
    "KSamplerAdvanced": (94, 48, 48),
    "SamplerCustom": (94, 48, 48),
    "VAEDecode": (48, 90, 48),
    "VAEEncode": (48, 80, 48),
    "VAEEncodeForInpaint": (48, 80, 48),
    "SaveImage": (90, 68, 18),
    "PreviewImage": (80, 60, 18),
    "LoadImage": (50, 70, 90),
    "ImageScale": (50, 70, 80),
    "ImageResize": (50, 70, 80),
    "ControlNetLoader": (55, 78, 100),
    "ControlNetApply": (55, 78, 100),
    "ControlNetApplyAdvanced": (55, 78, 100),
    "IPAdapterLoader": (80, 55, 100),
    "IPAdapter": (80, 55, 100),
    "LoraLoader": (60, 90, 60),
    "LoraLoaderModelOnly": (60, 90, 60),
    "PrimitiveNode": (38, 38, 50),
    "Note": (38, 55, 38),
}
_DEFAULT_COLOR = (42, 42, 52)
_TITLE_LIFT = 28  # how many units to lighten each channel for the title bar

# Connector type → wire/dot color
_SLOT_COLORS: dict[str, tuple[int, int, int]] = {
    "MODEL":        (183, 117, 255),
    "CONDITIONING": (255, 182, 80),
    "LATENT":       (148, 80, 180),
    "IMAGE":        (80, 182, 100),
    "VAE":          (220, 100, 100),
    "CLIP":         (100, 182, 220),
    "INT":          (145, 200, 145),
    "FLOAT":        (145, 200, 145),
    "STRING":       (200, 200, 145),
    "MASK":         (182, 150, 100),
    "CONTROL_NET":  (100, 160, 220),
    "UPSCALE_MODEL":(160, 160, 220),
}
_DEFAULT_WIRE = (155, 155, 155)


def _hex_to_rgb(s: str) -> tuple[int, int, int]:
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = s[0]*2 + s[1]*2 + s[2]*2
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _lighten(c: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
    return tuple(min(255, v + amount) for v in c)


def _darken(c: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
    return tuple(max(0, v - amount) for v in c)


def _try_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_workflow(
    json_path: str | Path,
    width: int = 960,
    height: int = 720,
) -> Image.Image | None:
    """
    Render a ComfyUI workflow JSON to a PIL Image.
    Returns None if the file is not a valid ComfyUI workflow.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None

    if not isinstance(data, dict) or "nodes" not in data:
        return None

    nodes: list[dict] = data.get("nodes", [])
    links: list[list] = data.get("links", [])
    groups: list[dict] = data.get("groups", [])

    if not nodes:
        return None

    # ── Parse node bounding boxes ───────────────────────────────────────────────
    node_rects: dict[int, tuple[float, float, float, float]] = {}
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

    # ── Calculate world bounds ──────────────────────────────────────────────────
    all_x = [r[0] for r in node_rects.values()] + [r[0] + r[2] for r in node_rects.values()]
    all_y = [r[1] for r in node_rects.values()] + [r[1] + r[3] for r in node_rects.values()]

    for g in groups:
        gpos = g.get("bounding") or g.get("pos", [0, 0])
        gsz = g.get("size", [200, 200])
        if gpos and len(gpos) >= 2:
            all_x += [float(gpos[0]), float(gpos[0]) + float(gsz[0])]
            all_y += [float(gpos[1]), float(gpos[1]) + float(gsz[1])]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    world_w = max(1.0, max_x - min_x)
    world_h = max(1.0, max_y - min_y)
    PAD = 40
    scale = min((width - PAD * 2) / world_w, (height - PAD * 2) / world_h)

    def to_screen(wx: float, wy: float) -> tuple[int, int]:
        return int((wx - min_x) * scale + PAD), int((wy - min_y) * scale + PAD)

    def sv(v: float) -> int:  # scale value
        return max(1, int(v * scale))

    # ── Canvas ──────────────────────────────────────────────────────────────────
    img = Image.new("RGB", (width, height), (28, 28, 35))
    draw = ImageDraw.Draw(img)

    # ── Draw groups ─────────────────────────────────────────────────────────────
    for g in groups:
        gpos = g.get("bounding") or g.get("pos", [0, 0])
        gsz = g.get("size", [200, 200])
        if not gpos or len(gpos) < 2:
            continue
        gx, gy = float(gpos[0]), float(gpos[1])
        gw, gh = float(gsz[0]), float(gsz[1])
        gc_hex = g.get("color", "#555555")
        try:
            gc = _hex_to_rgb(gc_hex)
        except Exception:
            gc = (80, 80, 80)
        sx1, sy1 = to_screen(gx, gy)
        sx2, sy2 = to_screen(gx + gw, gy + gh)
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        # Fill with dimmed group color
        fill = _darken(gc, 50)
        draw.rectangle([sx1, sy1, sx2, sy2], fill=fill, outline=gc, width=max(1, sv(1)))
        if g.get("title") and scale > 0.2:
            fs = max(8, sv(13))
            font = _try_font(fs)
            draw.text((sx1 + 4, sy1 + 2), g["title"], fill=(210, 210, 210), font=font)

    # ── Build slot screen positions ─────────────────────────────────────────────
    TITLE_H = 24.0   # world-space height of the title bar
    SLOT_H = 20.0    # world-space height per slot row

    out_pos: dict[tuple[int, int], tuple[int, int]] = {}
    in_pos: dict[tuple[int, int], tuple[int, int]] = {}

    for n in nodes:
        nid = n.get("id")
        if nid not in node_rects:
            continue
        nx, ny, nw, nh = node_rects[nid]
        for i in range(len(n.get("outputs", []))):
            out_pos[(nid, i)] = to_screen(nx + nw, ny + TITLE_H + SLOT_H * (i + 0.5))
        for i in range(len(n.get("inputs", []))):
            in_pos[(nid, i)] = to_screen(nx, ny + TITLE_H + SLOT_H * (i + 0.5))

    # ── Draw links (bezier-approximated with line segments) ──────────────────────
    lw = max(1, sv(2))
    for lnk in links:
        if len(lnk) < 5:
            continue
        _, src_id, src_slot, dst_id, dst_slot = lnk[0], lnk[1], lnk[2], lnk[3], lnk[4]
        ltype = str(lnk[5]).upper() if len(lnk) > 5 else ""
        p0 = out_pos.get((src_id, src_slot))
        p1 = in_pos.get((dst_id, dst_slot))
        if not p0 or not p1:
            continue
        color = _SLOT_COLORS.get(ltype, _DEFAULT_WIRE)
        x0, y0 = p0
        x1, y1 = p1
        dx = abs(x1 - x0) * 0.5
        steps = max(8, abs(x1 - x0) // 8)
        cx0, cy0 = x0 + dx, y0
        cx1, cy1 = x1 - dx, y1
        pts = []
        for i in range(steps + 1):
            t = i / steps
            mt = 1 - t
            bx = mt**3*x0 + 3*mt**2*t*cx0 + 3*mt*t**2*cx1 + t**3*x1
            by = mt**3*y0 + 3*mt**2*t*cy0 + 3*mt*t**2*cy1 + t**3*y1
            pts.append((int(bx), int(by)))
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=color, width=lw)

    # ── Draw nodes ───────────────────────────────────────────────────────────────
    for n in nodes:
        nid = n.get("id")
        if nid not in node_rects:
            continue
        nx, ny, nw, nh = node_rects[nid]
        ntype: str = n.get("type", "")
        title: str = n.get("title") or ntype
        muted: bool = n.get("mode", 0) == 2

        sx1, sy1 = to_screen(nx, ny)
        sx2, sy2 = to_screen(nx + nw, ny + nh)
        if sx2 - sx1 < 4 or sy2 - sy1 < 4:
            continue

        # Background
        if n.get("bgcolor"):
            try:
                bg = _hex_to_rgb(n["bgcolor"])
            except Exception:
                bg = _NODE_COLORS.get(ntype, _DEFAULT_COLOR)
        else:
            bg = _NODE_COLORS.get(ntype, _DEFAULT_COLOR)
        if muted:
            bg = _darken(bg, 30)

        draw.rectangle([sx1, sy1, sx2, sy2], fill=bg, outline=(88, 88, 108), width=1)

        # Title bar
        title_sy2 = sy1 + sv(TITLE_H)
        if n.get("color"):
            try:
                title_bg = _hex_to_rgb(n["color"])
            except Exception:
                title_bg = _lighten(bg, _TITLE_LIFT)
        else:
            title_bg = _lighten(bg, _TITLE_LIFT)
        if title_sy2 > sy1 and title_sy2 <= sy2:
            draw.rectangle([sx1, sy1, sx2, title_sy2], fill=title_bg)

        # Title text
        if scale > 0.22:
            fs = max(7, sv(11))
            font = _try_font(fs)
            avail_chars = max(3, (sx2 - sx1 - 8) // max(1, fs // 2))
            label = title[:avail_chars] + ("…" if len(title) > avail_chars else "")
            draw.text((sx1 + 4, sy1 + max(2, sv(4))), label, fill=(218, 218, 218), font=font)

        # Slot dots
        if scale > 0.35:
            dr = max(2, sv(4))
            for i, inp in enumerate(n.get("inputs", [])):
                pos = in_pos.get((nid, i))
                if pos:
                    itype = (inp.get("type", "") if isinstance(inp, dict) else "").upper()
                    c = _SLOT_COLORS.get(itype, (175, 175, 175))
                    draw.ellipse([pos[0]-dr, pos[1]-dr, pos[0]+dr, pos[1]+dr], fill=c)
            for i, out in enumerate(n.get("outputs", [])):
                pos = out_pos.get((nid, i))
                if pos:
                    otype = (out.get("type", "") if isinstance(out, dict) else "").upper()
                    c = _SLOT_COLORS.get(otype, (175, 175, 175))
                    draw.ellipse([pos[0]-dr, pos[1]-dr, pos[0]+dr, pos[1]+dr], fill=c)

    return img
