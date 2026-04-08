#!/usr/bin/env python3
"""Generate all meaningful input combinations for resolve_color_metadata.

Without flags: prints a table of inputs + resolved outputs for review.
With --pytest:  emits a complete pytest file to stdout.
"""

import sys
from itertools import product

MATRIX_RAW = ["bt2020nc", "bt2020c", "bt709", "bt470bg", "smpte170m", None]
TRANSFER_RAW = ["smpte2084", "arib-std-b67", "bt709", "bt470bg", "bt470m", "smpte170m", None]
PRIMARIES_RAW = ["bt2020", "bt709", "bt470bg", "smpte170m", "bt470m", None]
SYSTEMS = ["PAL", "NTSC", "HD"]
HAS_HDR = [True, False]


def is_realistic(mx, tr, pri, sys, hdr):
    """Filter out obviously impossible combos."""
    if hdr and mx in ("bt709", "bt470bg", "smpte170m"):
        return False
    if pri == "bt470m" and sys == "HD":
        return False
    if pri == "bt2020" and mx in ("bt470bg", "smpte170m"):
        return False
    if pri in ("bt470bg", "smpte170m", "bt470m") and mx in ("bt2020nc", "bt2020c", "bt709"):
        return False
    if tr in ("smpte2084", "arib-std-b67") and mx in ("bt470bg", "smpte170m"):
        return False
    if tr in ("bt470bg", "bt470m", "smpte170m") and mx in ("bt2020nc", "bt2020c"):
        return False
    # BT.2020/BT.709 matrix with SD system is not realistic
    if mx in ("bt2020nc", "bt2020c", "bt709") and sys in ("PAL", "NTSC"):
        return False
    # SD matrix with HD system is not realistic
    if mx in ("bt470bg", "smpte170m") and sys == "HD":
        return False
    return True


def hdr_matters(mx, tr):
    """has_hdr only changes output when we need to infer family or transfer."""
    if mx is None:
        return True
    if mx in ("bt2020nc", "bt2020c") and tr is None:
        return True
    return False


def systems_for(mx, tr, pri, hdr):
    """Return relevant systems for a given combo."""
    if mx in ("bt2020nc", "bt2020c", "bt709"):
        return ["HD"]
    if mx in ("bt470bg", "smpte170m"):
        # System only matters when inferring (pri or tr is None)
        if pri is None or tr is None:
            return ["PAL", "NTSC"]
        return ["PAL"]  # representative, output identical for both
    # mx=None
    if hdr:
        return ["HD"]  # has_hdr → family=bt2020 regardless
    # Need to distinguish all three families
    if pri is None or tr is None:
        return ["PAL", "NTSC", "HD"]
    return ["PAL", "HD"]  # PAL representative for SD, HD for rest


def iter_cases():
    """Yield (matrix_raw, transfer_raw, primaries_raw, system, has_hdr)."""
    for mx, tr, pri in product(MATRIX_RAW, TRANSFER_RAW, PRIMARIES_RAW):
        if hdr_matters(mx, tr):
            hdrs = [True, False]
        else:
            hdrs = [False]

        for hdr in hdrs:
            for sys in systems_for(mx, tr, pri, hdr):
                if is_realistic(mx, tr, pri, sys, hdr):
                    yield (mx, tr, pri, sys, hdr)


def _family(mx, hdr, system):
    """Determine color family from matrix_raw, has_hdr, system."""
    if mx in ("bt2020nc", "bt2020c"):
        return "bt2020"
    if mx == "bt709":
        return "bt709"
    if mx in ("bt470bg", "smpte170m"):
        return "bt601"
    # mx is None
    if hdr:
        return "bt2020"
    if system == "HD":
        return "bt709"
    # PAL or NTSC
    return "bt601"


def resolve(mx, tr, pri, system, hdr):
    """Resolve color metadata → (matrix, transfer, primaries)."""
    fam = _family(mx, hdr, system)

    # --- matrix ---
    if mx is not None:
        matrix = mx
    elif fam == "bt2020":
        matrix = "bt2020nc"
    elif fam == "bt709":
        matrix = "bt709"
    elif system == "PAL":
        matrix = "bt470bg"
    else:
        matrix = "smpte170m"

    # --- primaries ---
    if pri is not None:
        primaries = pri
    elif fam == "bt2020":
        primaries = "bt2020"
    elif fam == "bt709":
        primaries = "bt709"
    elif system == "PAL":
        primaries = "bt470bg"
    else:
        primaries = "smpte170m"

    # --- transfer ---
    if tr is not None:
        transfer = tr
    elif fam == "bt2020":
        transfer = "smpte2084" if hdr else "bt709"
    elif fam == "bt709":
        transfer = "bt709"
    else:
        # bt601: infer from resolved primaries
        _pri_transfer = {
            "bt470bg": "bt470bg",
            "smpte170m": "smpte170m",
            "bt470m": "bt470m",
            "bt709": "bt709",
        }
        if primaries in _pri_transfer:
            transfer = _pri_transfer[primaries]
        elif system == "PAL":
            transfer = "bt470bg"
        else:
            transfer = "smpte170m"

    return (matrix, transfer, primaries)


def print_table(cases):
    """Print a human-readable table of inputs and resolved outputs."""
    print(
        f"{'#':>4}  {'matrix_raw':<12} {'transfer_raw':<14} {'primaries_raw':<14} {'sys':<5} {'hdr':<5}"
        f"  →  {'matrix':<12} {'transfer':<14} {'primaries':<14}"
    )
    print("-" * 102)
    for i, (mx, tr, pri, s, hdr) in enumerate(cases, 1):
        r_mx, r_tr, r_pri = resolve(mx, tr, pri, s, hdr)
        print(
            f"{i:>4}  {str(mx):<12} {str(tr):<14} {str(pri):<14} {s:<5} {str(hdr):<5}"
            f"  →  {r_mx:<12} {r_tr:<14} {r_pri:<14}"
        )
    print(f"\nTotal: {len(cases)}")


def _fmt(val):
    """Format a value as a Python repr for code generation."""
    if val is None:
        return "None"
    return repr(val)


def _sys_enum(s):
    """Convert system string to VideoSystem enum repr."""
    return f"VideoSystem.{s}"


def print_pytest(cases):
    """Emit a complete pytest file to stdout."""
    lines = [
        '"""Auto-generated by scripts/gen_color_tests.py --pytest"""',
        "from __future__ import annotations",
        "",
        "import pytest",
        "",
        "from furnace.core.detect import ResolvedColor, VideoSystem, resolve_color_metadata",
        "",
        "",
        "# fmt: off",
        "CASES = [",
    ]
    for mx, tr, pri, s, hdr in cases:
        r_mx, r_tr, r_pri = resolve(mx, tr, pri, s, hdr)
        line = (
            f"    ({_fmt(mx)}, {_fmt(tr)}, {_fmt(pri)}, "
            f"{_sys_enum(s)}, {hdr}, "
            f'ResolvedColor({_fmt(r_mx)}, {_fmt(r_tr)}, {_fmt(r_pri)})),'
        )
        lines.append(line)
    lines.append("]")
    lines.append("# fmt: on")
    lines.append("")
    lines.append("")
    lines.append("@pytest.mark.parametrize(")
    lines.append('    "matrix_raw, transfer_raw, primaries_raw, system, has_hdr, expected",')
    lines.append("    CASES,")
    lines.append(")")
    lines.append("def test_resolve_color_metadata(matrix_raw, transfer_raw, primaries_raw, system, has_hdr, expected):")
    lines.append("    result = resolve_color_metadata(matrix_raw, transfer_raw, primaries_raw, system, has_hdr)")
    lines.append("    assert result == expected, (")
    lines.append('        f"Input: mx={matrix_raw}, tr={transfer_raw}, pri={primaries_raw}, sys={system}, hdr={has_hdr}\\n"')
    lines.append('        f"Expected: {expected}\\n"')
    lines.append('        f"Got:      {result}"')
    lines.append("    )")
    lines.append("")

    print("\n".join(lines))


def main():
    cases = list(iter_cases())
    if "--pytest" in sys.argv:
        print_pytest(cases)
    else:
        print_table(cases)


if __name__ == "__main__":
    main()
