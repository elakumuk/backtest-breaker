"""Adversarial, tool-using strategy auditor with code-grounded verdicts."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bb.audit import HoldoutVault, cost_stress, param_sensitivity, walk_forward
from bb.data import DataGate
from bb.engine import run_backtest
from bb.llm import chat
from bb.metrics import deflated_sharpe_ratio, pbo_cscv, sharpe
from bb.spec import StrategySpec

# These verdict thresholds are judgment calls; keep all rubric cutoffs here.
RUBRIC = {"broken_holdout": 0.0, "broken_cost": 0.0, "broken_dsr": 0.5, "broken_pbo": 0.5,
          "wf_ratio": 0.5, "sensitivity_spread": 0.5, "fragile_dsr": 0.95,
          "holdout_ratio": 0.5, "fragile_pbo": 0.3}
FINAL_SCHEMA = {"type":"object","properties":{"model_verdict":{"type":"string","enum":["SURVIVES","FRAGILE","BROKEN"]},"bullets":{"type":"array","items":{"type":"string"},"minItems":5},"predicted_holdout_sharpe":{"type":"number"}},"required":["model_verdict","bullets","predicted_holdout_sharpe"],"additionalProperties":False}
TOOL_NAMES = ("walk_forward", "param_sensitivity", "cost_stress", "deflated_sharpe", "pbo", "open_holdout")

def _round(value: Any) -> Any:
    """Round every numeric tool result consistently for model grounding."""
    if isinstance(value, dict): return {str(k): _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_round(v) for v in value]
    if isinstance(value, (float, np.floating)): return round(float(value), 4) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)): return int(value)
    return value

def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

def _ungrounded_numbers(bullets: list[str], evidence: dict[str, Any], exempt: set[float]) -> list[str]:
    """Return numeric claims not supported at the allowed output precision."""
    available: set[float] = set()
    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                try: available.add(float(key))
                except (TypeError, ValueError): pass
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value: collect(item)
        elif isinstance(value, (int, float)):
            available.add(float(value))
    collect(evidence)
    flagged=[]
    for bullet in bullets:
        for token in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?", bullet):
            raw=token.rstrip("%"); value=float(raw)
            places=len(raw.split(".")[1]) if "." in raw else 0
            precision=places if places in {0,1,2,4} else 8
            factor=100 if token.endswith("%") else 1
            matches=any(round(number*factor,precision)==value for number in available)
            if not (1900<=value<=2100) and value not in exempt and not matches:
                flagged.append(token)
    return flagged

def _reasons_and_verdict(evidence: dict[str, Any], in_sample: float) -> tuple[str, list[dict[str, Any]]]:
    reasons=[]; broken=False; fragile=False
    def flag(key, value, threshold, rule, condition):
        nonlocal broken, fragile
        if value is not None and condition(value, threshold):
            reasons.append({"rule":rule,"value":_round(value),"threshold":threshold})
            if rule.startswith("BROKEN"): broken=True
            else: fragile=True
    hold=evidence.get("open_holdout",{}).get("sharpe")
    cost=evidence.get("cost_stress",{}).get("25")
    dsr=evidence.get("deflated_sharpe",{}).get("dsr")
    pbo=evidence.get("pbo",{}).get("pbo")
    flag("",hold,RUBRIC["broken_holdout"],"BROKEN: holdout Sharpe below zero",lambda v,t:v<t)
    flag("",cost,RUBRIC["broken_cost"],"BROKEN: 25 bps Sharpe below zero",lambda v,t:v<t)
    if dsr is not None and pbo is not None and dsr<RUBRIC["broken_dsr"] and pbo>RUBRIC["broken_pbo"]:
        reasons.append({"rule":"BROKEN: DSR below 0.5 and PBO above 0.5","value":{"dsr":dsr,"pbo":pbo},"threshold":{"dsr":.5,"pbo":.5}}); broken=True
    wf=evidence.get("walk_forward",{}).get("median_sharpe")
    flag("",wf,RUBRIC["wf_ratio"],"FRAGILE: median walk-forward Sharpe below half in-sample",lambda v,t:v<t*in_sample)
    flag("",evidence.get("param_sensitivity",{}).get("spread"),.5,"FRAGILE: parameter sensitivity spread above 0.5",lambda v,t:v>t)
    flag("",dsr,.95,"FRAGILE: DSR below 0.95",lambda v,t:v<t)
    flag("",hold,.5,"FRAGILE: holdout Sharpe below half in-sample",lambda v,t:v<t*in_sample)
    flag("",pbo,.3,"FRAGILE: PBO above 0.3",lambda v,t:v>t)
    if not evidence.get("open_holdout"): fragile=True; reasons.append({"rule":"FRAGILE: holdout not opened; not proven out of sample","value":None,"threshold":None})
    return ("BROKEN" if broken else "FRAGILE" if fragile else "SURVIVES"), reasons

def audit_run(run_dir: str | Path, *, mock_script: list[dict[str,Any]] | None = None) -> dict[str,Any]:
    folder=Path(run_dir); spec=StrategySpec.model_validate_json((folder/"best_spec.json").read_text())
    gate=DataGate(); returns,rf=gate.in_sample(); returns=returns.loc[:pd.Timestamp(spec.end)]; rf=rf.reindex(returns.index).fillna(0)
    trials=_jsonl(folder/"trials.jsonl"); n_trials=len(trials); frame=pd.read_parquet(folder/"trial_returns.parquet")
    best_result=run_backtest(spec,returns,rf)
    best=sharpe(best_result.daily_returns,rf.reindex(best_result.daily_returns.index).fillna(0),long_short=spec.long_short)
    vault=HoldoutVault(folder/"audit_log.jsonl"); events=_jsonl(folder/"audit_log.jsonl")
    tools=[{"type":"function","function":{"name":name,"description":name.replace("_"," "),"parameters":{"type":"object","properties":{},"additionalProperties":False}}} for name in TOOL_NAMES]
    system=f"You are an adversarial auditor. Try to break this strategy. Choose which attacks to run and in what order. You may open the sealed holdout exactly once, last. Cite only numbers that tools returned. Spec: {spec.to_json()}; in-sample Sharpe: {_round(best)}; n_trials: {n_trials}."
    messages=[{"role":"system","content":system},{"role":"user","content":"Audit this strategy using tools, then return the required JSON."}]
    evidence={}; trace=[]; required={"walk_forward","cost_stress","deflated_sharpe"}; skipped=[]; opened=False; tokens=0; tool_call_count=0
    for turn in range(8):
        response=chat(messages,tools=None if opened else tools,json_schema=FINAL_SCHEMA,mock_policy={"kind":"auditor","script":mock_script or [],"turn":turn})
        tokens+=response.usage.get("total_tokens",0)
        if not response.tool_calls:
            try: answer=json.loads(response.content)
            except ValueError: answer={"model_verdict":"FRAGILE","bullets":["The model did not return valid JSON."],"predicted_holdout_sharpe":0}
            break
        messages.append({"role":"assistant","content":response.content or None,"tool_calls":[{"id":c.get("id",""),"type":"function","function":{"name":c["name"],"arguments":json.dumps(c.get("arguments",{})) if not isinstance(c.get("arguments"),str) else c["arguments"]}} for c in response.tool_calls]})
        for call in response.tool_calls:
            if tool_call_count >= 8: break
            tool_call_count += 1
            name=call["name"]
            if opened: raise ValueError("tool call rejected after open_holdout")
            if name=="open_holdout" and len(trace)<2: result={"error":"open_holdout requires at least two other tools"}
            elif name not in TOOL_NAMES: result={"error":"unknown tool"}
            else:
                start=time.monotonic()
                try:
                    if name=="walk_forward": result={"folds":walk_forward(spec,returns,rf),"median_sharpe":float(np.nanmedian([x["sharpe"] for x in walk_forward(spec,returns,rf)]))}
                    elif name=="param_sensitivity": result=param_sensitivity(spec,returns,rf)
                    elif name=="cost_stress": result={str(k):v for k,v in cost_stress(spec,returns,rf).items()}
                    elif name=="deflated_sharpe":
                        tr=[(r.get("metrics") or {}).get("sharpe") for r in trials if r.get("status")=="ok"]; tr=[x for x in tr if x is not None]
                        vals=(frame.iloc[:,0].dropna().to_numpy() if len(frame.columns) else np.array([])); sr=float(np.nanmean(vals)/np.nanstd(vals,ddof=1)) if len(vals)>1 and np.nanstd(vals,ddof=1)>0 else 0
                        result={"dsr":deflated_sharpe_ratio(sr/np.sqrt(252),np.asarray(tr)/np.sqrt(252),len(vals),float(pd.Series(vals).skew()),float(pd.Series(vals).kurt()+3))}
                    elif name=="pbo": result={"pbo":pbo_cscv(frame.dropna(axis=0,how="all").dropna(axis=1,how="all").fillna(0))[0]}
                    else:
                        if not any(e.get("event")=="prediction" and e.get("spec")==json.loads(spec.to_json()) for e in events):
                            pred=chat([{"role":"system","content":"Predict holdout Sharpe before it is opened. Return JSON."},{"role":"user","content":f"Spec {spec.to_json()} and in-sample Sharpe {_round(best)}"}],json_schema={"type":"object","properties":{"predicted_holdout_sharpe":{"type":"number"},"rationale":{"type":"string"}},"required":["predicted_holdout_sharpe","rationale"],"additionalProperties":False})
                            p=json.loads(pred.content); vault.register_prediction(spec,p["predicted_holdout_sharpe"],p["rationale"]); tokens+=pred.usage.get("total_tokens",0)
                            trace.append({"tool":"prediction","args":{},"result":_round(p),"seconds":0.0,"tokens":pred.usage.get("total_tokens",0)})
                        result=vault.open_once(spec,replay=True); opened=True
                    result=_round(result); evidence[name]=result
                except Exception as exc: result={"error":str(exc)}
                trace.append({"tool":name,"args":call.get("arguments",{}),"result":result,"seconds":round(time.monotonic()-start,4),"tokens":response.usage.get("total_tokens",0)})
            messages.append({"role":"tool","tool_call_id":call.get("id",""),"content":json.dumps(result)})
            if name=="open_holdout" and opened: break
        if opened:
            messages.append({"role":"user","content":"The sealed holdout has been evaluated. Give your final JSON now; call no tools."})
    else: answer={"model_verdict":"FRAGILE","bullets":["The audit reached its tool call limit."],"predicted_holdout_sharpe":0}
    for name in sorted(required-set(evidence)):
        skipped.append(name)
        # Required evidence is completed by code even if the model skipped it.
        if name=="walk_forward": val=walk_forward(spec,returns,rf); result={"folds":val,"median_sharpe":float(np.nanmedian([x["sharpe"] for x in val])) if val else None}
        elif name=="cost_stress": result={str(k):v for k,v in cost_stress(spec,returns,rf).items()}
        else:
            vals=[(t.get("metrics") or {}).get("sharpe") for t in trials if t.get("status")=="ok"]; vals=[x for x in vals if x is not None]
            result={"dsr":deflated_sharpe_ratio(best/np.sqrt(252),np.asarray(vals)/np.sqrt(252),len(returns),0,3)}
        result=_round(result); evidence[name]=result; trace.append({"tool":name,"args":{},"result":result,"seconds":0.0,"tokens":0,"auto_run":True})
    verdict,reasons=_reasons_and_verdict(evidence,best)
    if len(answer.get("bullets",[]))<5:
        repair=chat(messages+[{"role":"user","content":"Rewrite the final answer as exactly 5 plain-English bullets for a non-quant, using only the tool results above."}],json_schema=FINAL_SCHEMA)
        try: answer=json.loads(repair.content)
        except ValueError: pass
        tokens+=repair.usage.get("total_tokens",0)
    exempt={float(spec.lookback_days),float(spec.skip_days),float(spec.top_n),float(n_trials)}
    nums=_ungrounded_numbers(answer.get("bullets",[]),evidence,exempt)
    if nums:
        repair=chat(messages+[{"role":"user","content":f"These numbers are not in the tool outputs: {nums}; rewrite using only tool numbers."}],json_schema=FINAL_SCHEMA)
        try: answer=json.loads(repair.content)
        except ValueError: pass
        tokens+=repair.usage.get("total_tokens",0)
    still=_ungrounded_numbers(answer.get("bullets",[]),evidence,exempt)
    report={"final_verdict":verdict,"model_verdict":answer.get("model_verdict"),"model_disagrees":answer.get("model_verdict")!=verdict,"reasons":reasons,"bullets":answer.get("bullets",[]),"grounded":not still,"ungrounded_numbers":still,"skipped_required":skipped,"trace":trace,"total_tokens":tokens,"provider":response.provider,"model":response.model}
    (folder/"auditor_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-dir"); parser.add_argument("--control-12-1",action="store_true"); args=parser.parse_args()
    if args.control_12_1:
        gate=DataGate(); returns,rf=gate.in_sample(); start=pd.Timestamp("1990-01-01").date(); end=min(gate.holdout_start-pd.Timedelta(days=1),returns.index.max().date())
        folder=Path("runs")/f"control-12-1-{int(time.time())}"; folder.mkdir(parents=True,exist_ok=False)
        spec=StrategySpec(name="plain 12-1 momentum",signal_type="momentum",lookback_days=252,skip_days=21,top_n=1,rebalance="M",long_short=False,cost_bps=10,start=start,end=end)
        result=run_backtest(spec,returns.loc[:pd.Timestamp(end)],rf); (folder/"best_spec.json").write_text(spec.to_json()+"\n")
        pd.DataFrame({"control":result.daily_returns}).to_parquet(folder/"trial_returns.parquet")
        tr={"iter":1,"trial_id":"trial_001","spec":json.loads(spec.to_json()),"metrics":{"sharpe":sharpe(result.daily_returns,rf.reindex(result.daily_returns.index).fillna(0))},"status":"ok"}
        (folder/"trials.jsonl").write_text(json.dumps(tr)+"\n")
        vault=HoldoutVault(folder/"audit_log.jsonl"); vault.seal(); vault.register_prediction(spec,0.0,"Control baseline prediction")
        print(json.dumps(audit_run(folder),indent=2)); return
    if not args.run_dir: parser.error("--run-dir or --control-12-1 is required")
    print(json.dumps(audit_run(args.run_dir),indent=2))
if __name__=="__main__": main()
