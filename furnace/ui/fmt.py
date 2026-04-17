"""Shared formatting helpers for the UI layer."""

from __future__ import annotations


def fmt_size(n: int | None) -> str:
    """Format byte count as ``'N,NNN MB'``, or ``'?'`` when unknown."""
    if n is None or n == 0:
        return "?"
    mb = n / (1024 * 1024)
    return f"{mb:,.0f} MB"
