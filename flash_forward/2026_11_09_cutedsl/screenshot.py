# screenshot.py
# Small X11 helpers (python-xlib + XTest + Pillow) to drive nsys-ui / ncu-ui and save a
# screenshot cropped to one window. Used interactively while writing the article:
#
#   python screenshot.py list                       # windows and their geometry
#   python screenshot.py shot "Nsight Compute" images/x.png
#   python screenshot.py click "Nsight Compute" 120 340 [right]
#   python screenshot.py key "Nsight Compute" ctrl+f
import os
import sys
import time

from PIL import ImageGrab
from Xlib import X, XK, display, protocol
from Xlib.ext import xtest

DISPLAY = os.environ.get("DISPLAY", ":1")
d = display.Display(DISPLAY)
root = d.screen().root
A = lambda n: d.intern_atom(n)  # noqa: E731


def windows():
    out = []
    for wid in root.get_full_property(A("_NET_CLIENT_LIST"), X.AnyPropertyType).value:
        w = d.create_resource_object("window", wid)
        name = w.get_full_property(A("_NET_WM_NAME"), 0)
        name = name.value.decode() if name else ""
        g = w.get_geometry()
        pos = w.translate_coords(root, 0, 0)
        out.append((wid, name, -pos.x, -pos.y, g.width, g.height))
    return out


def find(title):
    for wid, name, x, y, w, h in windows():
        if title.lower() in name.lower():
            return wid, name, x, y, w, h
    raise SystemExit(f"no window matching {title!r}")


def activate(title):
    wid = find(title)[0]
    ev = protocol.event.ClientMessage(window=d.create_resource_object("window", wid),
                                      client_type=A("_NET_ACTIVE_WINDOW"), data=(32, [2, X.CurrentTime, 0, 0, 0]))
    root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
    d.flush()
    time.sleep(0.4)


def move_resize(title, x, y, w, h):
    wid = find(title)[0]
    win = d.create_resource_object("window", wid)
    for state in ("_NET_WM_STATE_MAXIMIZED_VERT", "_NET_WM_STATE_MAXIMIZED_HORZ"):
        ev = protocol.event.ClientMessage(window=win, client_type=A("_NET_WM_STATE"),
                                          data=(32, [0, A(state), 0, 1, 0]))
        root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
    ev = protocol.event.ClientMessage(window=win, client_type=A("_NET_MOVERESIZE_WINDOW"),
                                      data=(32, [(1 << 8) | (1 << 9) | (1 << 10) | (1 << 11) | 10, x, y, w, h]))
    root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
    d.flush()
    time.sleep(0.8)


def click(title, rx, ry, button=1, double=False):
    _, _, x, y, _, _ = find(title)
    xtest.fake_input(d, X.MotionNotify, x=x + rx, y=y + ry)
    d.sync()
    time.sleep(0.15)
    for _ in range(2 if double else 1):
        xtest.fake_input(d, X.ButtonPress, button)
        xtest.fake_input(d, X.ButtonRelease, button)
        d.sync()
        time.sleep(0.08)
    time.sleep(0.4)


def hover(title, rx, ry):
    _, _, x, y, _, _ = find(title)
    xtest.fake_input(d, X.MotionNotify, x=x + rx, y=y + ry)
    d.sync()
    time.sleep(1.0)


def scroll(title, rx, ry, clicks):
    hover(title, rx, ry)
    btn = 5 if clicks > 0 else 4
    for _ in range(abs(clicks)):
        xtest.fake_input(d, X.ButtonPress, btn)
        xtest.fake_input(d, X.ButtonRelease, btn)
        d.sync()
        time.sleep(0.05)
    time.sleep(0.4)


def key(combo):
    names = combo.split("+")
    codes = []
    for n in names:
        n = {"ctrl": "Control_L", "shift": "Shift_L", "alt": "Alt_L", "enter": "Return", "esc": "Escape"}.get(n, n)
        codes.append(d.keysym_to_keycode(XK.string_to_keysym(n)))
    for c in codes:
        xtest.fake_input(d, X.KeyPress, c)
    for c in reversed(codes):
        xtest.fake_input(d, X.KeyRelease, c)
    d.sync()
    time.sleep(0.3)


def type_text(s):
    for ch in s:
        sym = XK.string_to_keysym({" ": "space", ".": "period", "/": "slash", "_": "underscore", "-": "minus",
                                   "=": "equal", ",": "comma"}.get(ch, ch))
        code = d.keysym_to_keycode(sym)
        shift = ch.isupper() or ch == "_"
        if shift:
            xtest.fake_input(d, X.KeyPress, d.keysym_to_keycode(XK.XK_Shift_L))
        xtest.fake_input(d, X.KeyPress, code)
        xtest.fake_input(d, X.KeyRelease, code)
        if shift:
            xtest.fake_input(d, X.KeyRelease, d.keysym_to_keycode(XK.XK_Shift_L))
        d.sync()
        time.sleep(0.02)


def shot(title, path, crop=None):
    """crop = (left, top, right, bottom) relative to the window, optional."""
    _, _, x, y, w, h = find(title)
    box = (x, y, x + w, y + h) if crop is None else (x + crop[0], y + crop[1], x + crop[2], y + crop[3])
    ImageGrab.grab(bbox=box, xdisplay=DISPLAY).save(path)
    return path


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        for row in windows():
            print(row)
    elif cmd == "shot":
        crop = tuple(int(v) for v in sys.argv[4].split(",")) if len(sys.argv) > 4 else None
        print(shot(sys.argv[2], sys.argv[3], crop))
    elif cmd == "click":
        click(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), 3 if len(sys.argv) > 5 and sys.argv[5] == "right" else 1,
              double=len(sys.argv) > 5 and sys.argv[5] == "double")
    elif cmd == "hover":
        hover(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "scroll":
        scroll(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif cmd == "key":
        activate(sys.argv[2])
        key(sys.argv[3])
    elif cmd == "type":
        activate(sys.argv[2])
        type_text(sys.argv[3])
    elif cmd == "activate":
        activate(sys.argv[2])
    elif cmd == "move":
        move_resize(sys.argv[2], *(int(v) for v in sys.argv[3:7]))
