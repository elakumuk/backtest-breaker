"""Altair charts for the app. One palette, thin marks, recessive axes, hover tooltips everywhere."""
from __future__ import annotations

import altair as alt
import pandas as pd

# Roles (reference palette, light mode). Text never wears a series color.
BLUE = "#2a78d6"      # series 1: the strategy / backtest
ORANGE = "#eb6834"    # series 2: the flagged trial / out-of-sample
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _style(chart: alt.Chart, height: int) -> alt.Chart:
    return (
        chart.properties(height=height, background=SURFACE, padding={"left": 8, "right": 16, "top": 12, "bottom": 8})
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=MUTED, titleColor=INK_2, labelFont=FONT, titleFont=FONT, labelFontSize=11,
            titleFontSize=12, titleFontWeight="normal", gridColor=GRID, gridWidth=1, domainColor=AXIS,
            tickColor=AXIS, tickSize=4, labelPadding=6, titlePadding=10,
        )
        .configure_text(font=FONT)
        .configure_legend(disable=True)
    )


def trials_sharpe(trials: pd.DataFrame, best_iter: int, median: float) -> alt.Chart:
    """In-sample Sharpe per trial, best-so-far step, median reference, flagged trial called out."""
    df = trials.copy()
    df["flag"] = df["trial"] == best_iter
    x = alt.X("trial:Q", title="Trial", scale=alt.Scale(domain=[0.5, len(df) + 0.5]), axis=alt.Axis(tickMinStep=1, grid=False, values=list(range(1, len(df) + 1, 2))))
    y = alt.Y("sharpe:Q", title="In-sample Sharpe", scale=alt.Scale(domain=[-0.3, 0.8]), axis=alt.Axis(tickCount=6, format=".1f"))
    tip = [alt.Tooltip("trial:Q", title="Trial"), alt.Tooltip("sharpe:Q", title="Sharpe", format=".3f"),
           alt.Tooltip("cost_bps:Q", title="Cost (bps)"), alt.Tooltip("signal:N", title="Signal"),
           alt.Tooltip("lookback_days:Q", title="Lookback (days)"), alt.Tooltip("top_n:Q", title="Top N"),
           alt.Tooltip("name:N", title="Muse Spark's name for it")]
    base = alt.Chart(df)
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=AXIS, strokeWidth=1).encode(y="y:Q")
    med = alt.Chart(pd.DataFrame({"y": [median]}))
    med_rule = med.mark_rule(color=MUTED, strokeDash=[4, 4], strokeWidth=1).encode(y="y:Q")
    med = alt.Chart(pd.DataFrame({"y": [median], "trial": [len(df)]}))
    med_lbl = med.mark_text(align="left", dx=8, dy=8, color=MUTED, fontSize=11).encode(x=x, y="y:Q", text=alt.value(f"median {median:.2f}"))
    best = base.mark_line(interpolate="step-after", color=MUTED, strokeWidth=1.5, opacity=0.8).encode(x=x, y="best_so_far:Q")
    line = base.mark_line(color=BLUE, strokeWidth=2).encode(x=x, y=y)
    pts = base.mark_circle(size=64, color=BLUE, stroke=SURFACE, strokeWidth=2, opacity=1).encode(x=x, y=y, tooltip=tip)
    hit = base.mark_circle(size=500, opacity=0).encode(x=x, y=y, tooltip=tip)
    flag = base.transform_filter(alt.datum.flag).mark_circle(size=150, color=ORANGE, stroke=SURFACE, strokeWidth=2, opacity=1).encode(x=x, y=y, tooltip=tip)
    flag_lbl = base.transform_filter(alt.datum.flag).mark_text(align="center", dy=-16, color=INK, fontSize=12, fontWeight=600).encode(
        x=x, y=y, text=alt.value(f"Trial {best_iter} · costs set to 0"))
    end = df.iloc[[-1]]
    end_lbl = alt.Chart(end).mark_text(align="left", dx=8, color=INK_2, fontSize=11).encode(x=x, y="best_so_far:Q", text=alt.value("best so far"))
    return _style(alt.layer(zero, med_rule, med_lbl, best, line, pts, flag, flag_lbl, end_lbl, hit), 300)


def trials_cost(trials: pd.DataFrame, best_iter: int) -> alt.Chart:
    """Cost assumption per trial: thin bars from a zero baseline; the zero-cost trial is marked."""
    df = trials.copy()
    df["flag"] = df["trial"] == best_iter
    x = alt.X("trial:Q", title="Trial", scale=alt.Scale(domain=[0.5, len(df) + 0.5]), axis=alt.Axis(tickMinStep=1, grid=False, values=list(range(1, len(df) + 1, 2))))
    tip = [alt.Tooltip("trial:Q", title="Trial"), alt.Tooltip("cost_bps:Q", title="Cost (bps)"), alt.Tooltip("sharpe:Q", title="Sharpe", format=".3f")]
    base = alt.Chart(df)
    bars = base.mark_bar(size=12, color=BLUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=x, y=alt.Y("cost_bps:Q", title="Assumed trading cost (bps)", scale=alt.Scale(domain=[0, 110]), axis=alt.Axis(tickCount=5)), tooltip=tip)
    hit = base.mark_bar(size=18, opacity=0).encode(x=x, y=alt.value(0), y2=alt.value(300), tooltip=tip)
    mark = base.transform_filter(alt.datum.flag).mark_point(shape="triangle-up", size=110, filled=True, color=ORANGE).encode(x=x, y=alt.Y("cost_bps:Q"), tooltip=tip)
    lbl = base.transform_filter(alt.datum.flag).mark_text(dy=-34, color=INK, fontSize=12, fontWeight=600).encode(x=x, y="cost_bps:Q", text=alt.value("0 bps"))
    return _style(alt.layer(bars, mark, lbl, hit), 300)


def sharpe_ladder(m: dict) -> alt.Chart:
    """The story in one chart: what the backtest said vs what the sealed holdout said."""
    rows = pd.DataFrame([
        {"step": "In-sample best (trial 13, 0 bps)", "sharpe": m["in_sample_best_sharpe"], "kind": "Backtest", "order": 1},
        {"step": "Same strategy at 10 bps", "sharpe": m["best_sharpe_at_10bps"], "kind": "Backtest", "order": 2},
        {"step": "Auditor's prediction (logged first)", "sharpe": m["predicted_holdout_sharpe"], "kind": "Out of sample", "order": 3},
        {"step": "Sealed holdout, actual (2024 →)", "sharpe": m["actual_holdout_sharpe"], "kind": "Out of sample", "order": 4},
    ])
    y = alt.Y("step:N", sort=alt.EncodingSortField("order"), title=None, axis=alt.Axis(labelColor=INK_2, labelFontSize=12, labelLimit=260, domain=False, ticks=False, grid=False))
    x = alt.X("sharpe:Q", title="Sharpe ratio", scale=alt.Scale(domain=[0, 0.8]), axis=alt.Axis(tickCount=5, format=".1f"))
    color = alt.Color("kind:N", scale=alt.Scale(domain=["Backtest", "Out of sample"], range=[BLUE, ORANGE]))
    tip = [alt.Tooltip("step:N", title=" "), alt.Tooltip("sharpe:Q", title="Sharpe", format=".3f")]
    base = alt.Chart(rows)
    rows["zero"] = 0.0
    base = alt.Chart(rows)
    stem = base.mark_rule(color=GRID, strokeWidth=2).encode(y=y, x="zero:Q", x2="sharpe:Q")
    dots = base.mark_circle(size=180, opacity=1, stroke=SURFACE, strokeWidth=2).encode(y=y, x=x, color=color, tooltip=tip)
    vals = base.mark_text(align="left", dx=12, color=INK, fontSize=13, fontWeight=600).encode(y=y, x=x, text=alt.Text("sharpe:Q", format=".2f"))
    return _style(alt.layer(stem, dots, vals), 210)


def walk_forward(folds: list[dict], median: float) -> alt.Chart:
    """Yearly walk-forward Sharpe: one series from a zero baseline, median as reference."""
    df = pd.DataFrame(folds)
    df["year"] = pd.to_datetime(df["test_start"]).dt.year
    tip = [alt.Tooltip("year:O", title="Test year"), alt.Tooltip("sharpe:Q", title="Sharpe", format=".2f")]
    base = alt.Chart(df)
    bars = base.mark_bar(size=10, color=BLUE, cornerRadiusEnd=3).encode(
        x=alt.X("year:O", title="Walk-forward test year", axis=alt.Axis(labelAngle=0, values=[y for y in df.year if y % 5 == 0], grid=False)),
        y=alt.Y("sharpe:Q", title="Sharpe", axis=alt.Axis(tickCount=5, format=".1f")), tooltip=tip)
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=AXIS).encode(y="y:Q")
    med = alt.Chart(pd.DataFrame({"y": [median]}))
    rule = med.mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
    return _style(alt.layer(zero, bars, rule), 220)


def equity(curve: pd.Series) -> alt.Chart:
    """Growth of $1, log scale, so 1990 and 2023 moves read at the same size."""
    df = curve.rename("value").to_frame().reset_index().rename(columns={"index": "date"})
    df.columns = ["date", "value"]
    df = df.iloc[::5]
    tip = [alt.Tooltip("date:T", title="Date", format="%b %Y"), alt.Tooltip("value:Q", title="Growth of $1", format="$,.2f")]
    line = alt.Chart(df).mark_line(color=BLUE, strokeWidth=1.5).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", grid=False, tickCount=8)),
        y=alt.Y("value:Q", title="Growth of $1 (log scale)", scale=alt.Scale(type="log", domain=[0.5, 300]), axis=alt.Axis(format="$,.0f", values=[1, 10, 100])))
    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
    hit = alt.Chart(df).mark_rule(color=AXIS).encode(x="date:T", opacity=alt.condition(hover, alt.value(1), alt.value(0)), tooltip=tip).add_params(hover)
    dot = alt.Chart(df).mark_circle(size=60, color=BLUE, stroke=SURFACE, strokeWidth=2).encode(x="date:T", y="value:Q", opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    return _style(alt.layer(line, hit, dot), 240)
