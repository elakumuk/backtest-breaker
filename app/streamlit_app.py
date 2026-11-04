"""Backtest Breaker app. Demo replay (default, offline) shows the showcase page; Live mode audits an uploaded notebook."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

import demo_data as dd  # noqa: E402
from cards import render_finding  # noqa: E402
from showcase import site_html  # noqa: E402

st.set_page_config(page_title="Backtest Breaker", page_icon="🔨", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {display:none !important}
header[data-testid="stHeader"] {background: transparent; height: 0}
.block-container {padding: 0 !important; max-width: 100% !important}
iframe {display:block; border:0}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Backtest Breaker")
    mode = st.radio("Mode", ["Demo replay", "Live"], index=0,
                    help="Demo replay shows saved live Muse Spark runs from demo/ and makes no model calls.")
    demo_mode = mode == "Demo replay"
    if demo_mode:
        os.environ["BB_DEMO_MODE"] = "1"
        os.environ["PROVIDER"] = "mock"
    else:
        os.environ.pop("BB_DEMO_MODE", None)
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=True)
    m = dd.manifest()
    st.caption(f"Model `{m['model']}` via {m['provider']}. Run `{m['run_id']}`.")
    st.caption("Audits strategies; does not give investment advice.")

if demo_mode:
    components.html(site_html(), height=4900, scrolling=False)
else:
    st.title("Audit my notebook (live)")
    st.caption("Muse Spark reads your notebook. A finding is Proven only if a generated test fails on your code, "
               "passes with the proposed fix, and passes on a clean control notebook.")
    up = st.file_uploader("Your notebook (.ipynb)", type="ipynb")
    clean = st.file_uploader("Optional clean control notebook (.ipynb)", type="ipynb")
    if up and not clean:
        st.warning("Without a clean control, findings can't be proven: the test must also pass on clean code.")
    if up and st.button("Audit with Muse Spark"):
        from bb.notebook_mode import audit_notebook
        with tempfile.TemporaryDirectory() as tmp:
            nb = Path(tmp) / "uploaded.ipynb"
            nb.write_bytes(up.getvalue())
            ctrl = None
            if clean:
                ctrl = Path(tmp) / "clean.ipynb"
                ctrl.write_bytes(clean.getvalue())
            started = time.monotonic()
            with st.spinner("Muse Spark is reading your notebook (reasoning model: 10-60 s)…"):
                try:
                    rep = audit_notebook(nb, clean_notebook=ctrl, timeout_seconds=60).to_dict()
                except Exception as exc:  # show a message, never a stack trace
                    st.error(f"Audit failed: {exc}")
                    rep = None
            if rep is not None:
                st.caption(f"Took {time.monotonic() - started:.0f} s.")
                cells = ["".join(c.get("source", "")) for c in json.loads(up.getvalue()).get("cells", [])]
                for f in rep.get("findings") or []:
                    with st.container(border=True):
                        render_finding(f, cells)
                if not rep.get("findings"):
                    st.success("No look-ahead findings.")
    st.caption("Backtest Breaker audits strategies. It does not give investment advice.")
