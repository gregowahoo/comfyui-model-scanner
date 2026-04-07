# ComfyUI Model Scanner

A standalone Python GUI tool that reads a ComfyUI workflow `.json` file, extracts all model references, and locates those models across your drives — with color-coded results showing exactly what's found, what's missing, and where duplicates live.

---

## Features

- 📂 **Parses both workflow formats** — API format (flat node dict) and UI format (`nodes` array with `widgets_values`), plus a deep-scan fallback for unusual layouts
- 🔍 **Catches models by field name** — `ckpt_name`, `unet_name`, `vae_name`, `lora_name`, `clip_name`, `control_net_name`, IP-Adapter fields, upscaler fields, and more
- 🧲 **Also catches by file extension** — any `.safetensors`, `.ckpt`, `.gguf`, `.pt`, `.pth`, `.bin`, `.sft` string found anywhere in the workflow
- 🗂️ **Indexes both drives recursively** — walks all subdirectories under your configured model roots
- 🎨 **Color-coded results**:
  - 🟢 Green = Found
  - 🔴 Red = Missing
  - 🟡 Yellow-green = Found in multiple locations (duplicate)
- 🖱️ **Detail panel** — click any row to see node type, node ID, all copy locations, and which drive each is on
- 📋 **Copy Path button** — one click copies the full path of the selected model to clipboard
- 🔎 **Filter bar + Found/Missing toggle** — quickly narrow down what you're looking at

---

## Requirements

- Python 3.x (already included with ComfyUI portable)
- No additional installs required — uses only Python standard library (`tkinter`, `json`, `os`, `pathlib`, `threading`)

---

## Usage

### Standard Python
```bash
python comfyui_model_scanner.py
```

### ComfyUI Portable Python (Windows)
```bash
C:\ComfyUI_windows_portable\python_embeded\python.exe comfyui_model_scanner.py
```

---

## Default Model Paths

The script scans these two locations by default:

```
C:\ComfyUI.Data\models
F:\ComfyUI.stuff\models
```

To change these, edit the `MODEL_ROOTS` list near the top of the script:

```python
MODEL_ROOTS = [
    r"C:\ComfyUI.Data\models",
    r"F:\ComfyUI.stuff\models",
]
```

---

## How It Works

1. **Load a workflow** — click the Browse button and select any ComfyUI `.json` workflow file
2. **Index your drives** — the app walks your model roots and builds a filename lookup index
3. **Extract model references** — the workflow is parsed for all model field names and file extensions
4. **Match and display** — each reference is matched against the index and displayed with status and full path

---

## Supported Model Types

| Field Name | Category |
|---|---|
| `ckpt_name` | Checkpoint |
| `unet_name` | UNet |
| `vae_name` | VAE |
| `lora_name`, `lora_01/02/03` | LoRA |
| `clip_name`, `clip_name1/2` | CLIP |
| `control_net_name` | ControlNet |
| `upscale_model` | Upscaler |
| `ip_adapter_file`, `ipadapter_file` | IP-Adapter |
| `model_name` | Model |
| `encoder_name` / `decoder_name` | Encoder / Decoder |
| Any `.safetensors`, `.ckpt`, `.gguf`, `.pt`, `.pth`, `.bin`, `.sft` value | Auto-detected |

---

## Notes

- Subfolder paths like `flux/model.safetensors` are handled — the script matches on filename only
- Duplicate detection flags models that exist on both drives
- The index is rebuilt each time you load a new workflow

---

## License

MIT — free to use, modify, and share.
