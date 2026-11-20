"""Small LaTeX helpers shared by the renderers."""

_SPECIAL = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
            ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
            ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
            ("’", "'"), ("‘", "'"), ("“", "``"), ("”", "''")]


def esc(s, em="---", en="--"):
    """Escape text for LaTeX; em/en set how — and – are written."""
    s = "" if s is None else str(s)
    for a, b in _SPECIAL + [("—", em), ("–", en)]:
        s = s.replace(a, b)
    return s


def trunc(s, n, ellipsis="..."):
    s = str(s or "")
    return s[:n] + (ellipsis if len(s) > n else "")


def plural(n, word):
    return f"{n} {word}" + ("s" if n != 1 else "")
