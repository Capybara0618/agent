from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_vendor() -> None:
    """Add the local dependency folder to sys.path when present."""
    root = Path(__file__).resolve().parent.parent
    vendor = root / ".vendor"
    if vendor.exists():
        vendor_path = str(vendor)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)
