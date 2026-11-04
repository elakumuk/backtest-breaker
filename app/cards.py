"""Finding card for notebook audits (shared by demo and live modes)."""
from __future__ import annotations

import difflib

import streamlit as st

BADGE = {"PROVEN": "🟢 PROVEN", "UNPROVEN": "🟡 UNPROVEN", "DUPLICATE": "⚪ DUPLICATE", "DISMISSED": "⚫ DISMISSED"}

def render_finding(f: dict, cells: list[str]) -> None:
    status = f.get("status", "UNPROVEN")
    st.markdown(f"**{BADGE.get(status, status)}** · `{f.get('bug_class', '')}` · cell {f.get('cell_index')}, line {f.get('line')}")
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
        shown = [f"{i + 1:>3} {'▶' if (i + 1) == bad else '│'} {ln}" for i, ln in enumerate(lines)]
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
