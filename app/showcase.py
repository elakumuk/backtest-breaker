"""The demo showcase page (docs/index.html) with its data inlined, for embedding in Streamlit offline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_site import bundle  # noqa: E402

TAG = '<script src="data.js"></script>'


def site_html() -> str:
    """Return the showcase page with demo data inlined (no network needed except optional web fonts)."""
    page = (ROOT / "docs" / "index.html").read_text()
    data = "<script>window.BB_DATA = " + json.dumps(bundle(), separators=(",", ":")).replace("</", "<\\/") + ";</script>"
    if TAG not in page:
        raise ValueError("docs/index.html must load data.js")
    return page.replace(TAG, data)
