"""Backtest Breaker: Streamlit app. Demo replay (default, offline) or Live mode."""
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

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import demo_data as dd  # noqa: E402

st.set_page_config(page_title="Backtest Breaker", page_icon="🔨", layout="wide")

VERDICT_COLOR = {"SURVIVES": "green", "FRAGILE": "orange", "BROKEN": "red"}

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🔨 Backtest Breaker")
    st.caption("An AI that tries to break your backtest before the market does.")
    mode = st.radio("Mode", ["Demo replay", "Live"], index=0,
                    help="Demo replay reads saved live Muse Spark runs from demo/ and makes no network calls.")
    demo_mode = mode == "Demo replay"
    if demo_mode:
        os.environ["BB_DEMO_MODE"] = "1"
        os.environ["PROVIDER"] = "mock"
    else:
        os.environ.pop("BB_DEMO_MODE", None)
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=True)
    m = dd.manifest()
    st.markdown(f"**Model:** `{m['model']}`  \n**Provider:** {m['provider']}")
    st.caption("Built during the Meta AI Developer Hackathon, Oct 26 – Nov 4 2026.")


def footer() -> None:
    st.divider()
    st.caption("Backtest Breaker audits strategies. It does not give investment advice.")


tab1, tab2, tab3 = st.tabs(["1 · Watch it overfit", "2 · Break it", "3 · Audit my notebook"])

# ---------------------------------------------------------------- tab 1
with tab1:
    trials = dd.trials_frame()
    st.subheader(f"Muse Spark proposed {m['n_trials']} strategies, iterating on in-sample data only")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("First trial Sharpe", f"{m['first_trial_sharpe']:.2f}")
    c2.metric("Median trial Sharpe", f"{m['median_trial_sharpe']:.2f}")
    c3.metric("Best in-sample Sharpe", f"{m['in_sample_best_sharpe']:.2f}", help=f"Trial {m['best_trial_iter']}")
    c4.metric(f"Best trial at 10 bps cost", f"{m['best_sharpe_at_10bps']:.2f}",
              delta=f"{m['best_sharpe_at_10bps'] - m['in_sample_best_sharpe']:.2f}")
    left, right = st.columns(2)
    with left:
        st.caption("In-sample Sharpe per trial (and best so far)")
        st.line_chart(trials.set_index("trial")[["sharpe", "best_so_far"]])
    with right:
        st.caption("Trading cost the model assumed in each trial (bps)")
        st.bar_chart(trials.set_index("trial")[["cost_bps"]])
    if m["best_trial_cost_bps"] == 0:
        st.warning(f"The 'best' strategy is trial {m['best_trial_iter']}: the one trial where the model set trading "
                   f"costs to **zero**. At a realistic 10 bps it scores {m['best_sharpe_at_10bps']:.2f}.")
    st.caption("Click a trial to see the spec Muse Spark proposed.")
    for _, row in trials.iterrows():
        label = f"Trial {int(row.trial)} · Sharpe {row.sharpe:.3f} · cost {row.cost_bps:g} bps · {row.signal}"
        with st.expander(label, expanded=False):
            st.markdown(f"**{row['name']}**")
            st.json({k: row[k] for k in ("signal", "lookback_days", "skip_days", "top_n", "long_short", "rebalance", "cost_bps")})
    footer()

# ---------------------------------------------------------------- tab 2
with tab2:
    report = dd.auditor_report()
    st.subheader("The auditor attacks the winner, then opens the sealed holdout exactly once")
    a, b = st.columns([3, 2])
    with a:
        st.markdown("#### Tool-call trace")
        for i, step in enumerate(report["trace"], 1):
            tool = step["tool"]
            res = step.get("result") or {}
            if tool == "walk_forward":
                line = f"median fold Sharpe {res.get('median_sharpe')} across {len(res.get('folds', []))} yearly folds"
            elif tool == "cost_stress":
                line = ", ".join(f"{k} bps → {v}" for k, v in sorted(res.items(), key=lambda kv: float(kv[0])))
            elif tool == "param_sensitivity":
                line = f"spread {res.get('spread')} across ±20% variants"
            elif tool == "open_holdout":
                line = f"🔓 holdout Sharpe {res.get('sharpe')} (opened once; hash-verified)"
            else:
                line = json.dumps(res)[:140]
            auto = " · auto-run by code" if step.get("auto_run") else ""
            st.markdown(f"**{i}. `{tool}`**{auto}  \n{line}")
    with b:
        color = VERDICT_COLOR.get(report["final_verdict"], "gray")
        st.markdown(f"#### Verdict: :{color}[{report['final_verdict']}]")
        st.caption("Decided by a code rubric, not by the model.")
        for r in report["reasons"]:
            st.markdown(f"- {r['rule']} (value {r['value']})")
        if report["model_disagrees"]:
            st.info(f"Muse Spark's own verdict was **{report['model_verdict']}**. The rubric overrules it.")
        st.markdown(f"Numbers in the explanation grounded in tool output: **{'yes' if report['grounded'] else 'no'}**")
        p, q = st.columns(2)
        p.metric("Predicted holdout Sharpe", f"{m['predicted_holdout_sharpe']:.2f}", help="Logged before the vault opened")
        q.metric("Actual holdout Sharpe", f"{m['actual_holdout_sharpe']:.2f}",
                 delta=f"{m['actual_holdout_sharpe'] - m['in_sample_best_sharpe']:.2f} vs in-sample")
        with st.expander("Prediction rationale (written before the holdout opened)"):
            st.write(m["prediction_rationale"])
    st.markdown("#### In-sample growth of $1, selected strategy")
    st.line_chart(dd.best_equity_curve())
    with st.expander("Auditor's plain-English explanation"):
        for bullet in report["bullets"]:
            st.markdown(f"- {bullet}")
    footer()

# ---------------------------------------------------------------- tab 3
from cards import render_finding  # noqa: E402

with tab3:
    st.subheader("Upload a notebook. Muse Spark finds the look-ahead leak and proves it with a test.")
    st.caption("PROVEN = the generated test fails on your code, passes once the proposed fix is applied, and passes on a clean control notebook.")
    if demo_mode:
        names = dd.notebook_names()
        if not names:
            st.info("No saved notebook reports in demo/notebooks.")
        else:
            name = st.selectbox("Bundled example notebook (saved live Muse Spark report)", names)
            rep = dd.notebook_report(name)
            cells = dd.notebook_cells(name)
            findings = rep.get("findings") or []
            dismissed = rep.get("dismissed") or []
            if rep.get("seconds"):
                st.caption(f"Live audit took {rep['seconds']} s with Muse Spark.")
            if not findings:
                st.success("No look-ahead findings.")
            for f in findings:
                with st.container(border=True):
                    render_finding(f, cells)
            if dismissed:
                with st.expander(f"{len(dismissed)} hypotheses the model checked and dismissed"):
                    for f in dismissed:
                        st.markdown(f"- {f.get('bug_class')}: {f.get('explanation', '')}")
    else:
        up = st.file_uploader("Your notebook (.ipynb)", type="ipynb")
        clean = st.file_uploader("Optional clean control notebook (.ipynb)", type="ipynb")
        if up and not clean:
            st.warning("Without a clean control, findings can't be PROVEN: the test must also pass on clean code.")
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
    footer()
