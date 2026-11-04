"""Finding card for notebook audits (shared by demo and live modes)."""
from __future__ import annotations

import difflib

import streamlit as st

import html

# status palette: icon + label always, never color alone
BADGE = {"PROVEN": ("#0ca30c", "#ffffff", "✓", "Proven"), "UNPROVEN": ("#fab219", "#0b0b0b", "▲", "Unproven"),
         "DUPLICATE": ("#e1e0d9", "#0b0b0b", "=", "Duplicate"), "DISMISSED": ("#e1e0d9", "#0b0b0b", "–", "Dismissed")}

def render_finding(f: dict, cells: list[str]) -> None:
    status = f.get("status", "UNPROVEN")
    bg, fg, icon, label = BADGE.get(status, ("#e1e0d9", "#0b0b0b", "•", str(status).title()))
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:4px">'
        f'<span style="background:{bg};color:{fg};border-radius:999px;padding:3px 12px;font-weight:700;font-size:13px">{icon} {label}</span>'
        f'<code style="font-size:13px">{html.escape(str(f.get("bug_class", "")))}</code>'
        f'<span style="color:#52514e;font-size:14px">cell {html.escape(str(f.get("cell_index")))}, line {html.escape(str(f.get("line")))}</span></div>',
        unsafe_allow_html=True)
    st.write(f.get("explanation", ""))
    idx = f.get("cell_index")
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        idx = None
    if isinstance(idx, int) and 0 <= idx < len(cells):
        lines = cells[idx].splitlines()
        bad = f.get("line")
        try:
            bad = int(bad)
        except (TypeError, ValueError):
            bad = None
        width = len(str(len(lines)))
        shown = [f"{str(i + 1).zfill(width)} {'▶' if (i + 1) == bad else '│'} {ln}" for i, ln in enumerate(lines)]
        st.code("\n".join(shown), language="python")
    fix = f.get("fix") or {}
    old, new = fix.get("old") or f.get("fix_old"), fix.get("new") or f.get("fix_new")
    if old and new:
        diff = difflib.unified_diff(old.splitlines(), new.splitlines(), "leaky", "fixed", lineterm="")
        st.code("\n".join(diff), language="diff")
    if f.get("reason"):
        st.caption(f"Proof status reason: {f['reason']}")
    proof = f.get("runner_stdout_tail")
    if proof and not f.get("test_code"):
        with st.expander("Proof: generated test output (bug → fails; fixed → passes; clean control → passes)"):
            st.code(proof if isinstance(proof, str) else "\n\n".join(f"[{k}]\n{v}" for k, v in proof.items()), language="text")
    if f.get("test_code"):
        with st.expander("Generated test (fails on the bug, must pass after the fix)"):
            st.code(f["test_code"], language="python")
