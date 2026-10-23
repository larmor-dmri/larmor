"""Sets the viewer's menu-bar name on macOS and keeps AppKit off its arguments."""

import ctypes
import ctypes.util

_P = ctypes.c_void_p


def prepare(name: str) -> None:
    """Names the app in the menu bar and keeps AppKit off its arguments.

    Call it before Qt creates its application. Without it, AppKit opens every
    command-line word it does not know as a file, such as `bv` in
    `python -m larmor.main bv`.
    """
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc") or "libobjc.dylib")
    ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation") or "Foundation")
    objc.objc_getClass.restype = objc.sel_registerName.restype = _P
    objc.objc_getClass.argtypes = objc.sel_registerName.argtypes = [ctypes.c_char_p]
    address = ctypes.cast(objc.objc_msgSend, _P).value
    assert address is not None
    # objc_msgSend must be called through a pointer of the method's own signature.
    send = ctypes.CFUNCTYPE(_P, _P, _P)(address)
    send_text = ctypes.CFUNCTYPE(_P, _P, _P, ctypes.c_char_p)(address)
    send_two = ctypes.CFUNCTYPE(None, _P, _P, _P, _P)(address)
    send_is = ctypes.CFUNCTYPE(ctypes.c_bool, _P, _P, _P)(address)
    send_one = ctypes.CFUNCTYPE(None, _P, _P, _P)(address)
    send_two_ret = ctypes.CFUNCTYPE(_P, _P, _P, _P, _P)(address)
    cls, sel = objc.objc_getClass, objc.sel_registerName

    def ns_string(value: str) -> object:
        """Returns value as an NSString."""
        return send_text(
            cls(b"NSString"), sel(b"stringWithUTF8String:"), value.encode()
        )

    bundle = send(cls(b"NSBundle"), sel(b"mainBundle"))
    mutable = cls(b"NSMutableDictionary")
    for getter in (b"infoDictionary", b"localizedInfoDictionary"):
        info = send(bundle, sel(getter))
        if info and send_is(info, sel(b"isKindOfClass:"), mutable):
            for key in ("CFBundleName", "CFBundleDisplayName"):
                send_two(
                    info, sel(b"setObject:forKey:"), ns_string(name), ns_string(key)
                )
    defaults = send(cls(b"NSUserDefaults"), sel(b"standardUserDefaults"))
    refuse = send_two_ret(
        cls(b"NSDictionary"),
        sel(b"dictionaryWithObject:forKey:"),
        ns_string("NO"),
        ns_string("NSTreatUnknownArgumentsAsOpen"),
    )
    send_one(defaults, sel(b"registerDefaults:"), refuse)
