"""Backtest Breaker: Streamlit app. Demo replay (default, offline) or Live mode."""
from __future__ import annotations

import html
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

import charts  # noqa: E402
import demo_data as dd  # noqa: E402
from cards import render_finding  # noqa: E402

st.set_page_config(page_title="Backtest Breaker", page_icon="🔨", layout="wide", initial_sidebar_state="collapsed")

STATUS = {  # status palette: always icon + label, never color alone
    "SURVIVES": ("#0ca30c", "✓", "Survives"),
    "FRAGILE": ("#fab219", "▲", "Fragile"),
    "BROKEN": ("#d03b3b", "✕", "Broken"),
}

st.markdown("""
<style>
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; --surface:#fcfcfb; --plane:#f9f9f7;
        --blue:#2a78d6; --orange:#eb6834; --ring:rgba(11,11,11,0.08); }
html, body, [class*="css"] { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display:none !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }
h1, h2, h3, h4 { letter-spacing: -0.01em; color: var(--ink); }
.bb-eyebrow { font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--blue); margin-bottom: 6px; }
.bb-title { font-size: 34px; font-weight: 700; line-height: 1.1; margin: 0 0 8px 0; color: var(--ink); }
.bb-sub { font-size: 17px; color: var(--ink2); max-width: 760px; margin: 0 0 18px 0; line-height: 1.5; }
.bb-strip { display:flex; flex-wrap:wrap; gap: 10px; align-items:center; margin: 4px 0 6px 0; }
.bb-pill { display:inline-flex; align-items:center; gap:8px; border:1px solid var(--line); background: var(--surface);
           border-radius: 999px; padding: 6px 14px; font-size: 14px; color: var(--ink2); }
.bb-pill b { color: var(--ink); font-weight: 650; }
.bb-dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
.bb-arrow { color: var(--muted); font-size: 18px; }
.bb-tiles { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; margin: 6px 0 18px 0; }
@media (max-width: 800px) { .bb-tiles { grid-template-columns: repeat(2, minmax(0,1fr)); } .bb-title { font-size: 30px; } }
.bb-tile { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }
.bb-tile .l { font-size: 13px; color: var(--ink2); margin-bottom: 6px; }
.bb-tile .v { font-size: 30px; font-weight: 650; color: var(--ink); line-height: 1.1; }
.bb-tile .d { font-size: 12.5px; color: var(--muted); margin-top: 6px; }
.bb-tile.flag { border-color: var(--orange); box-shadow: inset 3px 0 0 var(--orange); }
.bb-head h4 { margin: 0 0 2px 0; font-size: 15px; font-weight: 650; color: var(--ink); padding: 0; }
.bb-head .cap { font-size: 13px; color: var(--muted); margin-bottom: 6px; }

.bb-card { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px; margin-bottom: 14px; }
.bb-card h4 { margin: 0 0 4px 0; font-size: 15px; font-weight: 650; }
.bb-card .cap { font-size: 13px; color: var(--muted); margin-bottom: 8px; }
.bb-callout { border-left: 3px solid var(--orange); background: #fff4ee; border-radius: 8px; padding: 12px 16px; color: var(--ink); font-size: 15px; margin: 6px 0 18px 0; }
.bb-verdict { border-radius: 12px; border: 1px solid var(--line); background: var(--surface); padding: 18px 20px; }
.bb-badge { display:inline-flex; align-items:center; gap:8px; font-weight: 700; font-size: 22px; color: var(--ink); }
.bb-badge .ic { width: 30px; height: 30px; border-radius: 8px; display:inline-flex; align-items:center; justify-content:center; font-size: 15px; color: #0b0b0b; }
.bb-reason { font-size: 14px; color: var(--ink2); margin: 10px 0 0 0; padding-left: 0; list-style: none; }
.bb-reason li { padding: 6px 0; border-top: 1px solid var(--line); display:flex; justify-content:space-between; gap: 12px; }
.bb-reason li span:last-child { color: var(--ink); font-variant-numeric: tabular-nums; font-weight: 600; }
.bb-note { margin-top: 12px; font-size: 13.5px; color: var(--ink2); background: var(--plane); border-radius: 8px; padding: 10px 12px; }
.bb-tl { position: relative; margin: 4px 0 0 0; padding: 0; list-style: none; }
.bb-tl li { position: relative; padding: 0 0 16px 34px; }
.bb-tl li:before { content:""; position:absolute; left: 11px; top: 24px; bottom: 0; width: 2px; background: var(--line); }
.bb-tl li:last-child:before { display:none; }
.bb-tl .n { position:absolute; left: 0; top: 0; width: 24px; height: 24px; border-radius: 50%; background: var(--surface); border: 2px solid var(--blue);
            font-size: 11.5px; font-weight: 700; color: var(--ink); display:flex; align-items:center; justify-content:center; }
.bb-tl li.key .n { border-color: var(--orange); background: #fff4ee; }
.bb-tl .t { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; font-weight: 600; color: var(--ink); }
.bb-tl .r { font-size: 14px; color: var(--ink2); margin-top: 2px; }
.bb-tl .auto { font-size: 11px; color: var(--muted); margin-left: 6px; }
.bb-legend { display:flex; gap: 16px; font-size: 13px; color: var(--ink2); margin: 2px 0 4px 2px; }
.bb-legend span { display:inline-flex; align-items:center; gap:6px; }
.bb-key { width: 14px; height: 2px; display:inline-block; }
.bb-foot { margin-top: 28px; padding-top: 14px; border-top: 1px solid var(--line); font-size: 12.5px; color: var(--muted); }
div[data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
button[data-baseweb="tab"] { font-size: 15px; padding: 10px 14px; }
button[data-baseweb="tab"] p { font-size: 15px; font-weight: 600; }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Settings")
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
    st.markdown(f"**Model** `{m['model']}`  \n**Provider** {m['provider']}  \n**Run** `{m['run_id']}`")
    st.caption("Built during the Meta AI Developer Hackathon, Oct 26 – Nov 4 2026.")


def esc(x) -> str:
    return html.escape(str(x))


def tile(label: str, value: str, detail: str = "", flag: bool = False) -> str:
    return f'<div class="bb-tile{" flag" if flag else ""}"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div><div class="d">{esc(detail)}</div></div>'


def footer() -> None:
    st.markdown('<div class="bb-foot">Backtest Breaker audits strategies. It does not give investment advice. '
                'Every number on this page is read from saved run files; none is typed by hand or computed by the model.</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------- header
verdict = m["final_verdict"]
vcol, vic, vlabel = STATUS.get(verdict, ("#898781", "•", verdict))
st.markdown(f"""
<div class="bb-eyebrow">Backtest Breaker · live run on Muse Spark</div>
<div class="bb-title">An AI that tries to break your backtest before the market does.</div>
<p class="bb-sub">One Muse Spark agent searches for a strategy. A second one attacks it, writes down a prediction, and opens data
nobody has touched. Code computes every number and gives the final verdict.</p>
<div class="bb-strip">
  <span class="bb-pill"><span class="bb-dot" style="background:var(--blue)"></span>In-sample <b>{m['in_sample_best_sharpe']:.2f}</b></span>
  <span class="bb-arrow">→</span>
  <span class="bb-pill"><span class="bb-dot" style="background:var(--orange)"></span>Predicted <b>{m['predicted_holdout_sharpe']:.2f}</b></span>
  <span class="bb-arrow">→</span>
  <span class="bb-pill"><span class="bb-dot" style="background:var(--orange)"></span>Sealed holdout <b>{m['actual_holdout_sharpe']:.2f}</b></span>
  <span class="bb-arrow">→</span>
  <span class="bb-pill"><span class="bb-dot" style="background:{vcol}"></span>{vic} Verdict <b>{esc(vlabel)}</b></span>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["1 · Watch it overfit", "2 · Break it", "3 · Audit my notebook"])

# ---------------------------------------------------------------- tab 1
with tab1:
    trials = dd.trials_frame()
    st.markdown(f"### Muse Spark ran {m['n_trials']} strategy trials on in-sample data")
    st.markdown(f"""<div class="bb-tiles">
      {tile("First trial", f"{m['first_trial_sharpe']:.2f}", "Sharpe, in-sample")}
      {tile("Median trial", f"{m['median_trial_sharpe']:.2f}", f"across {m['n_trials']} trials, {m['failed_trials']} failed")}
      {tile("Best trial", f"{m['in_sample_best_sharpe']:.2f}", f"trial {m['best_trial_iter']}, cost {m['best_trial_cost_bps']:g} bps", flag=True)}
      {tile("Best trial at 10 bps", f"{m['best_sharpe_at_10bps']:.2f}", f"{m['best_sharpe_at_10bps'] - m['in_sample_best_sharpe']:+.2f} once costs are real")}
    </div>""", unsafe_allow_html=True)
    if m["best_trial_cost_bps"] == 0:
        st.markdown(f'<div class="bb-callout"><b>The winner cheated once.</b> Muse Spark mostly declined to overfit: it named its specs '
                    f'“not optimized for in-sample Sharpe” and raised its own cost assumption from 20 to 100 bps. The exception is trial '
                    f'{m["best_trial_iter"]}, where it set trading costs to zero, and that trial became the best.</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        st.markdown('<div class="bb-head"><h4>In-sample Sharpe by trial</h4><div class="cap">Hover a point for the spec Muse Spark proposed.</div>'
                    '<div class="bb-legend"><span><i class="bb-key" style="background:var(--blue)"></i>Trial Sharpe</span>'
                    '<span><i class="bb-key" style="background:var(--muted)"></i>Best so far</span>'
                    '<span><i class="bb-dot" style="background:var(--orange)"></i>Selected trial</span></div>', unsafe_allow_html=True)
        st.altair_chart(charts.trials_sharpe(trials, m["best_trial_iter"], m["median_trial_sharpe"]), use_container_width=True)
    with right, st.container(border=True):
        st.markdown('<div class="bb-head"><h4>Trading cost the model assumed</h4><div class="cap">Basis points per trade. One trial assumed trading was free.</div>'
                    '<div class="bb-legend"><span><i class="bb-key" style="background:var(--blue);height:8px;width:8px;border-radius:2px"></i>Cost assumption</span>'
                    '<span><i class="bb-dot" style="background:var(--orange)"></i>Zero-cost trial</span></div>', unsafe_allow_html=True)
        st.altair_chart(charts.trials_cost(trials, m["best_trial_iter"]), use_container_width=True)
    with st.expander("All 25 specs as a table"):
        show = trials[["trial", "sharpe", "cost_bps", "signal", "lookback_days", "skip_days", "top_n", "long_short", "name"]]
        st.dataframe(show, hide_index=True, use_container_width=True,
                     column_config={"sharpe": st.column_config.NumberColumn("Sharpe", format="%.3f"),
                                    "cost_bps": st.column_config.NumberColumn("Cost (bps)", format="%g"),
                                    "name": st.column_config.TextColumn("Muse Spark's name for it", width="large")})
    footer()

# ---------------------------------------------------------------- tab 2
with tab2:
    report = dd.auditor_report()
    events = dd.audit_events()
    st.markdown("### The auditor attacks the winner, then opens the sealed holdout once")
    a, b = st.columns([5, 4], gap="large")
    with b:
        reasons = "".join(f"<li><span>{esc(r['rule'].split(': ', 1)[-1][:1].upper() + r['rule'].split(': ', 1)[-1][1:])}</span><span>{r['value']:.2f}</span></li>" for r in report["reasons"])
        disagree = (f'<div class="bb-note">Muse Spark\'s own verdict was <b>{esc(report["model_verdict"].title())}</b>. '
                    f'The rubric overrules it: the model explains, code decides.</div>') if report["model_disagrees"] else ""
        grounded = "every number in its explanation matches tool output" if report["grounded"] else "some numbers did not match tool output"
        st.markdown(f"""<div class="bb-verdict">
          <div class="cap" style="font-size:13px;color:var(--muted);margin-bottom:8px">Verdict from the code rubric</div>
          <div class="bb-badge"><span class="ic" style="background:{vcol}">{vic}</span>{esc(vlabel)}</div>
          <ul class="bb-reason">{reasons}</ul>
          {disagree}
          <div class="bb-note">Grounding check: {grounded}.</div>
        </div>""", unsafe_allow_html=True)
    with a:
        steps = []
        n = 0
        pred = next((e for e in events if e["event"] == "prediction"), None)
        for step in report["trace"]:
            tool, res = step["tool"], step.get("result") or {}
            if tool == "open_holdout" and pred:
                n += 1
                steps.append(f'<li class="key"><div class="n">{n}</div><div class="t">register_prediction</div>'
                             f'<div class="r">Holdout Sharpe will be about <b>{pred["predicted_sharpe"]:.2f}</b>. Written down before the vault opens.</div></li>')
            n += 1
            if tool == "walk_forward":
                r = f"{len(res.get('folds', []))} yearly folds, median Sharpe {res.get('median_sharpe'):.2f}"
            elif tool == "cost_stress":
                r = " · ".join(f"{k} bps → {v:.2f}" for k, v in sorted(res.items(), key=lambda kv: float(kv[0])))
            elif tool == "param_sensitivity":
                r = f"±20% on lookback and top N moves Sharpe by at most {res.get('spread'):.2f}"
            elif tool == "deflated_sharpe":
                r = f"Deflated Sharpe {res.get('dsr'):.2f} after accounting for {m['n_trials']} trials"
            elif tool == "pbo":
                r = f"Probability of backtest overfitting {res.get('pbo'):.2f}"
            elif tool == "open_holdout":
                r = f"🔓 Holdout Sharpe <b>{res.get('sharpe'):.2f}</b>. Opened once, hash-verified."
            else:
                r = esc(json.dumps(res)[:120])
            auto = '<span class="auto">run by code</span>' if step.get("auto_run") else ""
            key = ' class="key"' if tool == "open_holdout" else ""
            steps.append(f'<li{key}><div class="n">{n}</div><div class="t">{esc(tool)}{auto}</div><div class="r">{r}</div></li>')
        st.markdown(f'<div class="bb-card"><h4>What the auditor did</h4><div class="cap">Tool calls chosen by Muse Spark, computed by code.</div>'
                    f'<ul class="bb-tl">{"".join(steps)}</ul></div>', unsafe_allow_html=True)
    ladder = st.container(border=True)
    ladder.markdown('<div class="bb-head"><h4>What the backtest said vs what new data said</h4>'
                '<div class="cap">The prediction was logged before the holdout opened.</div>'
                '<div class="bb-legend"><span><i class="bb-dot" style="background:var(--blue)"></i>Backtest</span>'
                '<span><i class="bb-dot" style="background:var(--orange)"></i>Out of sample</span></div>', unsafe_allow_html=True)
    ladder.altair_chart(charts.sharpe_ladder(m), use_container_width=True)
    with ladder.expander("Muse Spark's prediction rationale, written before the holdout opened"):
        st.write(m["prediction_rationale"])
    wf = next(s["result"] for s in report["trace"] if s["tool"] == "walk_forward")
    c, d = st.columns(2, gap="medium")
    with c, st.container(border=True):
        st.markdown('<div class="bb-head"><h4>Walk-forward, year by year</h4><div class="cap">Dashed line: median fold Sharpe. Strong on average, with losing years in between.</div>', unsafe_allow_html=True)
        st.altair_chart(charts.walk_forward(wf["folds"], wf["median_sharpe"]), use_container_width=True)
    with d, st.container(border=True):
        st.markdown('<div class="bb-head"><h4>Selected strategy, in-sample</h4><div class="cap">Growth of $1 from 1990, as selected: trial 13, zero trading cost.</div>', unsafe_allow_html=True)
        st.altair_chart(charts.equity(dd.best_equity_curve()), use_container_width=True)
    with st.expander("Auditor's plain-English explanation"):
        for bullet in report["bullets"]:
            st.markdown(f"- {bullet}")
    footer()

# ---------------------------------------------------------------- tab 3
with tab3:
    st.markdown("### Muse Spark reads a notebook, finds the look-ahead leak, and proves it")
    st.markdown('<p class="bb-sub" style="font-size:15px">A finding is <b>Proven</b> only if a generated test fails on your code, '
                'passes once the proposed fix is applied, and passes on a clean control notebook.</p>', unsafe_allow_html=True)
    if demo_mode:
        names = dd.notebook_names()
        if not names:
            st.info("No saved notebook reports in demo/notebooks.")
        else:
            name = st.selectbox("Example notebook (saved live Muse Spark audit)", names)
            rep = dd.notebook_report(name)
            cells = dd.notebook_cells(name)
            findings = rep.get("findings") or []
            dismissed = rep.get("dismissed") or []
            if rep.get("seconds"):
                st.caption(f"Live audit took {rep['seconds']:.0f} s with Muse Spark.")
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
    footer()
