"""
ComfyUI Workflow Preview Handler
Windows Shell Extension that shows a visual graph of a ComfyUI workflow JSON
in the File Explorer preview pane.

Usage:
    python preview_handler.py --register      # run as Administrator
    python preview_handler.py --unregister    # run as Administrator
    python preview_handler.py                 # run as COM local server (called by Windows)
"""

import ctypes
import ctypes.wintypes as wt
import os
import sys
import traceback
import winreg
from pathlib import Path

# ── CLSID for this handler ──────────────────────────────────────────────────────
CLSID_STR = "{A4B3C2D1-E5F6-4789-8A2B-3C4D5E6F7A8B}"
PROGID = "ComfyUIWorkflow.PreviewHandler.1"
PROGID_VER = "ComfyUIWorkflow.PreviewHandler"
HANDLER_DESC = "ComfyUI Workflow Preview Handler"

# ── COM constants ───────────────────────────────────────────────────────────────
S_OK = 0
S_FALSE = 1
E_NOTIMPL = ctypes.c_long(0x80004001).value
E_NOINTERFACE = ctypes.c_long(0x80004002).value
E_POINTER = ctypes.c_long(0x80004003).value
E_FAIL = ctypes.c_long(0x80004005).value
CLASS_E_NOAGGREGATION = ctypes.c_long(0x80040110).value

CLSCTX_LOCAL_SERVER = 0x4
REGCLS_SINGLEUSE    = 0x0
REGCLS_MULTIPLEUSE  = 0x1

HRESULT = ctypes.c_long
ULONG = ctypes.c_ulong
LPVOID = ctypes.c_void_p

# ── GUID ────────────────────────────────────────────────────────────────────────
class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_uint8 * 8),
    ]

    def __init__(self, s: str = ""):
        super().__init__()
        if s:
            s = s.strip("{}")
            parts = s.split("-")
            self.Data1 = int(parts[0], 16)
            self.Data2 = int(parts[1], 16)
            self.Data3 = int(parts[2], 16)
            raw = bytes.fromhex(parts[3] + parts[4])
            for i, b in enumerate(raw):
                self.Data4[i] = b

    def __eq__(self, other):
        return (isinstance(other, GUID)
                and self.Data1 == other.Data1
                and self.Data2 == other.Data2
                and self.Data3 == other.Data3
                and bytes(self.Data4) == bytes(other.Data4))


REFIID = ctypes.POINTER(GUID)

# Well-known IIDs
IID_IUnknown = GUID("{00000000-0000-0000-C000-000000000046}")
IID_IClassFactory = GUID("{00000001-0000-0000-C000-000000000046}")
IID_IPreviewHandler = GUID("{8895b1c6-b41f-4c1c-a562-0d564250836f}")
IID_IInitializeWithFile = GUID("{b7d14566-0509-4cce-a71f-0a554233bd9b}")
CLSID_HANDLER = GUID(CLSID_STR)

# ── RECT ────────────────────────────────────────────────────────────────────────
class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

# ── Vtable function-pointer types ────────────────────────────────────────────────
QI_T   = ctypes.WINFUNCTYPE(HRESULT, LPVOID, REFIID, ctypes.POINTER(LPVOID))
AR_T   = ctypes.WINFUNCTYPE(ULONG,   LPVOID)
REL_T  = ctypes.WINFUNCTYPE(ULONG,   LPVOID)

# IClassFactory
CI_T   = ctypes.WINFUNCTYPE(HRESULT, LPVOID, LPVOID, REFIID, ctypes.POINTER(LPVOID))
LS_T   = ctypes.WINFUNCTYPE(HRESULT, LPVOID, ctypes.c_bool)

# IPreviewHandler
SW_T   = ctypes.WINFUNCTYPE(HRESULT, LPVOID, wt.HWND, ctypes.POINTER(RECT))
SR_T   = ctypes.WINFUNCTYPE(HRESULT, LPVOID, ctypes.POINTER(RECT))
DP_T   = ctypes.WINFUNCTYPE(HRESULT, LPVOID)
UL_T   = ctypes.WINFUNCTYPE(HRESULT, LPVOID)
SF_T   = ctypes.WINFUNCTYPE(HRESULT, LPVOID)
QF_T   = ctypes.WINFUNCTYPE(HRESULT, LPVOID, ctypes.POINTER(wt.HWND))
TA_T   = ctypes.WINFUNCTYPE(HRESULT, LPVOID, LPVOID)

# IInitializeWithFile
INIT_T = ctypes.WINFUNCTYPE(HRESULT, LPVOID, ctypes.c_wchar_p, ctypes.c_uint32)

# ── Vtable structures ────────────────────────────────────────────────────────────
class _VtblIPH(ctypes.Structure):
    _fields_ = [
        ("QueryInterface", QI_T), ("AddRef", AR_T), ("Release", REL_T),
        ("SetWindow", SW_T), ("SetRect", SR_T), ("DoPreview", DP_T),
        ("Unload", UL_T), ("SetFocus", SF_T), ("QueryFocus", QF_T),
        ("TranslateAccelerator", TA_T),
    ]

class _VtblIIWF(ctypes.Structure):
    _fields_ = [
        ("QueryInterface", QI_T), ("AddRef", AR_T), ("Release", REL_T),
        ("Initialize", INIT_T),
    ]

class _VtblICF(ctypes.Structure):
    _fields_ = [
        ("QueryInterface", QI_T), ("AddRef", AR_T), ("Release", REL_T),
        ("CreateInstance", CI_T), ("LockServer", LS_T),
    ]

# ── COM object structs (just a vtable pointer) ────────────────────────────────
class _ObjIPH(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(_VtblIPH))]

class _ObjIIWF(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(_VtblIIWF))]

class _ObjICF(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(_VtblICF))]

# ── Global instance map (COM ptr address → Python impl) ───────────────────────
_instances: dict[int, "PreviewHandlerImpl"] = {}

# ── Win32 helpers ────────────────────────────────────────────────────────────────
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_ole32  = ctypes.WinDLL("ole32",  use_last_error=True)

def _log(msg: str):
    try:
        log_path = Path(os.environ.get("TEMP", "C:\\Temp")) / "comfyui_preview.log"
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
    except Exception:
        pass


# ── Preview Handler implementation ────────────────────────────────────────────────
class PreviewHandlerImpl:
    """
    Python implementation of IPreviewHandler + IInitializeWithFile.
    Two COM object structs share this Python object.
    """

    def __init__(self):
        self.ref_count = 1
        self.file_path: str | None = None
        self.parent_hwnd: int = 0
        self.rect = RECT()
        self.preview_hwnd: int = 0
        self.pil_image = None

        # Build vtable for IPreviewHandler
        self._vtbl_iph = _VtblIPH(
            QueryInterface=QI_T(self._iph_qi),
            AddRef=AR_T(self._addref),
            Release=REL_T(self._release),
            SetWindow=SW_T(self._set_window),
            SetRect=SR_T(self._set_rect),
            DoPreview=DP_T(self._do_preview),
            Unload=UL_T(self._unload),
            SetFocus=SF_T(self._set_focus),
            QueryFocus=QF_T(self._query_focus),
            TranslateAccelerator=TA_T(self._translate_acc),
        )
        self._obj_iph = _ObjIPH()
        self._obj_iph.lpVtbl = ctypes.pointer(self._vtbl_iph)

        # Build vtable for IInitializeWithFile
        self._vtbl_iiwf = _VtblIIWF(
            QueryInterface=QI_T(self._iiwf_qi),
            AddRef=AR_T(self._addref),
            Release=REL_T(self._release),
            Initialize=INIT_T(self._initialize),
        )
        self._obj_iiwf = _ObjIIWF()
        self._obj_iiwf.lpVtbl = ctypes.pointer(self._vtbl_iiwf)

        # Register both COM pointers → self
        _instances[ctypes.addressof(self._obj_iph)] = self
        _instances[ctypes.addressof(self._obj_iiwf)] = self

    # ── IUnknown ──────────────────────────────────────────────────────────────
    def _addref(self, this: int) -> int:
        self.ref_count += 1
        return self.ref_count

    def _release(self, this: int) -> int:
        self.ref_count -= 1
        rc = max(0, self.ref_count)
        if rc == 0:
            _instances.pop(ctypes.addressof(self._obj_iph), None)
            _instances.pop(ctypes.addressof(self._obj_iiwf), None)
        return rc

    def _iph_qi(self, this: int, riid, ppv) -> int:
        if ppv is None:
            return E_POINTER
        iid = riid.contents
        if iid == IID_IUnknown or iid == IID_IPreviewHandler:
            ppv[0] = this
            self._addref(this)
            return S_OK
        elif iid == IID_IInitializeWithFile:
            ppv[0] = ctypes.addressof(self._obj_iiwf)
            self._addref(this)
            return S_OK
        ppv[0] = 0
        return E_NOINTERFACE

    def _iiwf_qi(self, this: int, riid, ppv) -> int:
        if ppv is None:
            return E_POINTER
        iid = riid.contents
        if iid == IID_IUnknown or iid == IID_IInitializeWithFile:
            ppv[0] = this
            self._addref(this)
            return S_OK
        elif iid == IID_IPreviewHandler:
            ppv[0] = ctypes.addressof(self._obj_iph)
            self._addref(this)
            return S_OK
        ppv[0] = 0
        return E_NOINTERFACE

    # ── IInitializeWithFile ───────────────────────────────────────────────────
    def _initialize(self, this: int, path: str, mode: int) -> int:
        try:
            self.file_path = path
            return S_OK
        except Exception:
            _log(traceback.format_exc())
            return E_FAIL

    # ── IPreviewHandler ───────────────────────────────────────────────────────
    def _set_window(self, this: int, hwnd: int, prc) -> int:
        try:
            self.parent_hwnd = hwnd
            if prc:
                r = prc.contents
                self.rect = RECT(r.left, r.top, r.right, r.bottom)
            return S_OK
        except Exception:
            _log(traceback.format_exc())
            return E_FAIL

    def _set_rect(self, this: int, prc) -> int:
        try:
            if prc:
                r = prc.contents
                self.rect = RECT(r.left, r.top, r.right, r.bottom)
                if self.preview_hwnd:
                    _user32.SetWindowPos(
                        self.preview_hwnd, 0,
                        r.left, r.top, r.right - r.left, r.bottom - r.top,
                        0x0014,  # SWP_NOZORDER | SWP_NOACTIVATE
                    )
            return S_OK
        except Exception:
            _log(traceback.format_exc())
            return S_OK

    def _do_preview(self, this: int) -> int:
        try:
            if not self.file_path:
                return E_FAIL
            w = max(100, self.rect.right - self.rect.left)
            h = max(100, self.rect.bottom - self.rect.top)

            from renderer import render_workflow
            self.pil_image = render_workflow(self.file_path, w, h)

            self._create_preview_window()
            return S_OK
        except Exception:
            _log(traceback.format_exc())
            return E_FAIL

    def _create_preview_window(self):
        import win32gui, win32con

        cls_name = "ComfyUIPreviewWindow"

        def wnd_proc(hwnd, msg, wparam, lparam):
            if msg == 0x000F:  # WM_PAINT
                self._on_paint(hwnd)
                return 0
            elif msg == 0x0014:  # WM_ERASEBKGND
                return 1
            elif msg == 0x0002:  # WM_DESTROY
                self.preview_hwnd = 0
                return 0
            elif msg == 0x0005:  # WM_SIZE
                win32gui.InvalidateRect(hwnd, None, False)
                return 0
            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

        wc = win32gui.WNDCLASS()
        wc.lpszClassName = cls_name
        wc.lpfnWndProc = wnd_proc
        wc.hbrBackground = 0
        try:
            win32gui.RegisterClass(wc)
        except Exception:
            pass  # already registered in this process

        # Get size from the parent's actual client area rather than trusting
        # the RECT coordinates (which may be screen-space on some hosts).
        try:
            cr = win32gui.GetClientRect(self.parent_hwnd)
            w = max(1, cr[2])
            h = max(1, cr[3])
        except Exception:
            r = self.rect
            w = max(1, r.right - r.left)
            h = max(1, r.bottom - r.top)

        _log(f"_create_preview_window: parent={self.parent_hwnd}  size={w}x{h}")

        # Create as a borderless popup first — avoids cross-process WS_CHILD
        # restrictions on modern Windows / DWM.
        hwnd = win32gui.CreateWindowEx(
            win32con.WS_EX_NOACTIVATE,
            cls_name,
            "",
            win32con.WS_POPUP,
            0, 0, w, h,
            None, None, None, None,
        )

        # Reparent into the preview container (cross-process SetParent is supported).
        win32gui.SetParent(hwnd, self.parent_hwnd)

        # Switch to child style so DWM clips it correctly inside the parent.
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
        win32gui.SetWindowLong(
            hwnd, win32con.GWL_STYLE,
            (style | win32con.WS_CHILD | win32con.WS_CLIPSIBLINGS) & ~win32con.WS_POPUP,
        )

        # Position at (0,0) within the parent client area and show.
        win32gui.SetWindowPos(
            hwnd, None,
            0, 0, w, h,
            win32con.SWP_NOZORDER | win32con.SWP_FRAMECHANGED | win32con.SWP_SHOWWINDOW,
        )

        self.preview_hwnd = hwnd
        win32gui.InvalidateRect(hwnd, None, True)
        win32gui.UpdateWindow(hwnd)
        _log(f"preview window hwnd={hwnd}")

    def _on_paint(self, hwnd: int):
        import win32gui
        hdc, ps = win32gui.BeginPaint(hwnd)
        try:
            cr = win32gui.GetClientRect(hwnd)
            cw, ch = cr[2], cr[3]
            # Black background
            _user32.FillRect(hdc, ctypes.byref(RECT(cr[0], cr[1], cr[2], cr[3])),
                             _user32.GetStockObject(4))  # BLACK_BRUSH = 4
            if self.pil_image and cw > 0 and ch > 0:
                from PIL import Image, ImageWin
                img = self.pil_image
                iw, ih = img.size
                sc = min(cw / iw, ch / ih)
                nw, nh = max(1, int(iw * sc)), max(1, int(ih * sc))
                scaled = img.resize((nw, nh), Image.Resampling.LANCZOS)
                xo = (cw - nw) // 2
                yo = (ch - nh) // 2
                dib = ImageWin.Dib(scaled)
                dib.draw(hdc, (xo, yo, xo + nw, yo + nh))
        except Exception:
            _log(traceback.format_exc())
        finally:
            win32gui.EndPaint(hwnd, ps)

    def _unload(self, this: int) -> int:
        try:
            import win32gui
            if self.preview_hwnd:
                win32gui.DestroyWindow(self.preview_hwnd)
                self.preview_hwnd = 0
            self.pil_image = None
            return S_OK
        except Exception:
            return S_OK

    def _set_focus(self, this: int) -> int:
        try:
            import win32gui
            if self.preview_hwnd:
                win32gui.SetFocus(self.preview_hwnd)
        except Exception:
            pass
        return S_OK

    def _query_focus(self, this: int, phwnd) -> int:
        try:
            import win32gui
            phwnd[0] = win32gui.GetFocus()
        except Exception:
            pass
        return S_OK

    def _translate_acc(self, this: int, pmsg: int) -> int:
        return S_FALSE

    def get_iph_ptr(self) -> int:
        return ctypes.addressof(self._obj_iph)


# ── Class Factory ────────────────────────────────────────────────────────────────
class ClassFactoryImpl:
    def __init__(self):
        self.ref_count = 1
        self._vtbl = _VtblICF(
            QueryInterface=QI_T(self._qi),
            AddRef=AR_T(self._addref),
            Release=REL_T(self._release),
            CreateInstance=CI_T(self._create_instance),
            LockServer=LS_T(self._lock_server),
        )
        self._obj = _ObjICF()
        self._obj.lpVtbl = ctypes.pointer(self._vtbl)

    def get_ptr(self) -> int:
        return ctypes.addressof(self._obj)

    def _addref(self, this: int) -> int:
        self.ref_count += 1
        return self.ref_count

    def _release(self, this: int) -> int:
        self.ref_count -= 1
        return max(0, self.ref_count)

    def _qi(self, this: int, riid, ppv) -> int:
        if ppv is None:
            return E_POINTER
        iid = riid.contents
        if iid == IID_IUnknown or iid == IID_IClassFactory:
            ppv[0] = this
            self._addref(this)
            return S_OK
        ppv[0] = 0
        return E_NOINTERFACE

    def _create_instance(self, this: int, pUnkOuter: int, riid, ppv) -> int:
        if pUnkOuter:
            if ppv:
                ppv[0] = 0
            return CLASS_E_NOAGGREGATION
        handler = PreviewHandlerImpl()
        iid = riid.contents
        if iid == IID_IUnknown or iid == IID_IPreviewHandler:
            ppv[0] = handler.get_iph_ptr()
            return S_OK
        elif iid == IID_IInitializeWithFile:
            ppv[0] = ctypes.addressof(handler._obj_iiwf)
            return S_OK
        if ppv:
            ppv[0] = 0
        return E_NOINTERFACE

    def _lock_server(self, this: int, lock: bool) -> int:
        return S_OK


# ── COM server entry point ────────────────────────────────────────────────────────
def run_server():
    import pythoncom
    import win32gui

    pythoncom.CoInitialize()
    factory = ClassFactoryImpl()  # keep alive
    cookie = ctypes.c_ulong(0)

    _log(f"COM server starting  pid={os.getpid()}  args={sys.argv}")

    hr = _ole32.CoRegisterClassObject(
        ctypes.byref(CLSID_HANDLER),
        factory.get_ptr(),
        CLSCTX_LOCAL_SERVER,
        REGCLS_MULTIPLEUSE,
        ctypes.byref(cookie),
    )
    if hr != 0:
        _log(f"CoRegisterClassObject failed: {hr:#010x}")
        pythoncom.CoUninitialize()
        return

    _log("COM server ready — waiting for connections")

    # Message pump
    msg = wt.MSG()
    while _user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        _user32.TranslateMessage(ctypes.byref(msg))
        _user32.DispatchMessageW(ctypes.byref(msg))

    _ole32.CoRevokeClassObject(cookie)
    pythoncom.CoUninitialize()


# ── Registration helpers ──────────────────────────────────────────────────────────
def _python_cmd() -> str:
    script = str(Path(__file__).resolve())
    exe = sys.executable
    return f'"{exe}" "{script}"'


def _get_json_progid() -> str | None:
    """Return the current ProgID for .json files (e.g. 'VSCode.json', 'jsonfile')."""
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for path in (r"SOFTWARE\Classes\.json", r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\FileExts\.json\UserChoice"):
            try:
                with winreg.OpenKey(hive, path) as k:
                    val, _ = winreg.QueryValueEx(k, "ProgId" if "UserChoice" in path else "")
                    if val:
                        return val
            except OSError:
                pass
    return None


def register():
    """
    Register this preview handler.  Must be run as Administrator.
    Registers for .json files via SystemFileAssociations (works regardless of
    which app owns .json — VS Code, Notepad++, etc.).
    """
    cmd = _python_cmd()
    clsid = CLSID_STR
    iid_preview = "{8895B1C6-B41F-4C1C-A562-0D564250836F}"

    # 1 – CLSID entry
    with winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE,
                          rf"SOFTWARE\Classes\CLSID\{clsid}") as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, HANDLER_DESC)

    with winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE,
                          rf"SOFTWARE\Classes\CLSID\{clsid}\LocalServer32") as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, cmd)

    with winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE,
                          rf"SOFTWARE\Classes\CLSID\{clsid}\ProgID") as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, PROGID)

    with winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE,
                          rf"SOFTWARE\Classes\CLSID\{clsid}\VersionIndependentProgID") as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, PROGID_VER)

    # 2 – AppID so COM knows this is a local server
    with winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE,
                          rf"SOFTWARE\Classes\CLSID\{clsid}\AppID") as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, clsid)
    with winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE,
                          rf"SOFTWARE\Classes\AppID\{clsid}") as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, HANDLER_DESC)

    # 3 – Approve the shell extension
    with winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE,
                          r"SOFTWARE\Microsoft\Windows\CurrentVersion"
                          r"\Shell Extensions\Approved") as k:
        winreg.SetValueEx(k, clsid, 0, winreg.REG_SZ, HANDLER_DESC)

    # 4 – Register in the PreviewHandlers known list
    with winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE,
                          r"SOFTWARE\Microsoft\Windows\CurrentVersion\PreviewHandlers") as k:
        winreg.SetValueEx(k, clsid, 0, winreg.REG_SZ, HANDLER_DESC)

    # 5 – SystemFileAssociations  ← the key entry Explorer actually uses
    #     This works regardless of which app is the default for .json
    with winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE,
                          rf"SOFTWARE\Classes\SystemFileAssociations\.json\ShellEx\{iid_preview}") as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, clsid)

    # 6 – Also register under the bare extension and common ProgIDs as fallbacks
    for reg_path in (
        rf"SOFTWARE\Classes\.json\ShellEx\{iid_preview}",
        rf"SOFTWARE\Classes\jsonfile\ShellEx\{iid_preview}",
        rf"SOFTWARE\Classes\json_auto_file\ShellEx\{iid_preview}",
    ):
        try:
            with winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as k:
                winreg.SetValueEx(k, "", 0, winreg.REG_SZ, clsid)
        except Exception:
            pass

    # 7 – Also register under the current system ProgID for .json
    progid = _get_json_progid()
    if progid and progid not in ("jsonfile", "json_auto_file"):
        try:
            with winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE,
                                  rf"SOFTWARE\Classes\{progid}\ShellEx\{iid_preview}") as k:
                winreg.SetValueEx(k, "", 0, winreg.REG_SZ, clsid)
            print(f"     Also registered for ProgID: {progid}")
        except Exception:
            pass

    print(f"[OK] Registered {HANDLER_DESC}")
    print(f"     CLSID : {clsid}")
    print(f"     Server: {cmd}")
    print("Restart File Explorer (or sign out/in) for changes to take effect.")


def unregister():
    clsid = CLSID_STR
    iid_preview = "{8895B1C6-B41F-4C1C-A562-0D564250836F}"

    def _del_tree(hive, path):
        try:
            with winreg.OpenKey(hive, path, 0, winreg.KEY_ALL_ACCESS) as k:
                # Delete sub-keys first
                while True:
                    try:
                        sub = winreg.EnumKey(k, 0)
                        _del_tree(hive, path + "\\" + sub)
                    except OSError:
                        break
            winreg.DeleteKey(hive, path)
        except FileNotFoundError:
            pass

    _del_tree(winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\Classes\CLSID\{clsid}")
    _del_tree(winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\Classes\AppID\{clsid}")

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SOFTWARE\Microsoft\Windows\CurrentVersion\PreviewHandlers",
                            0, winreg.KEY_SET_VALUE) as k:
            winreg.DeleteValue(k, clsid)
    except FileNotFoundError:
        pass

    for path in (
        rf"SOFTWARE\Classes\SystemFileAssociations\.json\ShellEx\{iid_preview}",
        rf"SOFTWARE\Classes\.json\ShellEx\{iid_preview}",
        rf"SOFTWARE\Classes\jsonfile\ShellEx\{iid_preview}",
        rf"SOFTWARE\Classes\json_auto_file\ShellEx\{iid_preview}",
    ):
        try:
            winreg.DeleteKey(winreg.HKEY_LOCAL_MACHINE, path)
        except FileNotFoundError:
            pass

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SOFTWARE\Microsoft\Windows\CurrentVersion"
                            r"\Shell Extensions\Approved",
                            0, winreg.KEY_SET_VALUE) as k:
            winreg.DeleteValue(k, clsid)
    except FileNotFoundError:
        pass

    print(f"[OK] Unregistered {HANDLER_DESC}")


# ── Entry point ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--register" in sys.argv:
        register()
    elif "--unregister" in sys.argv:
        unregister()
    else:
        # Called by COM / Windows with -Embedding or no args → run as server
        sys.path.insert(0, str(Path(__file__).parent))
        try:
            run_server()
        except Exception:
            _log(traceback.format_exc())
