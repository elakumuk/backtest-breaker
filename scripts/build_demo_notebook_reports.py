"""Generate saved Muse Spark notebook-audit reports for demo replay (run once, live)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from dotenv import load_dotenv

from bb.notebook_mode import audit_notebook

load_dotenv()
DEMO = Path(__file__).resolve().parents[1] / "demo" / "notebooks"
for name in ("leaky_norm", "leaky_shift", "leaky_universe"):
    start = time.monotonic()
    report = audit_notebook(DEMO / f"{name}.ipynb", clean_notebook=DEMO / "clean.ipynb", timeout_seconds=60).to_dict()
    report["notebook"] = f"{name}.ipynb"
    report["seconds"] = round(time.monotonic() - start, 1)
    (DEMO / f"{name}.report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(name, [(f.get("status"), f.get("cell_index")) for f in report["findings"]], report["seconds"])
