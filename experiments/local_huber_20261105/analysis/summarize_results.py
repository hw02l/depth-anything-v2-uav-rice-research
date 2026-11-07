import csv, json, math, html
from pathlib import Path
base = Path(__file__).resolve().parents[1]
out = Path(__file__).resolve().parent
with (base / "validation_summary.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
for r in rows:
    for k in ("epoch","validation_images"):
        r[k] = int(r[k])
    for k in ("rmse_m","mae_m","absrel","full14_mae_m","gt_ge20cm_14_mae_m","amplitude_ratio","sign_match_rate"):
        r[k] = float(r[k])
e0 = next(r for r in rows if r["group"] == "E0")
groups = ["E1","E2","E3"]
selected = {}
for g in groups:
    eligible = [r for r in rows if r["group"] == g and r["rmse_m"] <= 1.05*e0["rmse_m"] and r["full14_mae_m"] <= 1.05*e0["full14_mae_m"]]
    if not eligible:
        selected[g] = None
        continue
    best_local = min(r["gt_ge20cm_14_mae_m"] for r in eligible)
    tied = [r for r in eligible if r["gt_ge20cm_14_mae_m"] <= best_local + 1e-6]
    selected[g] = min(tied, key=lambda r:(r["rmse_m"],r["epoch"]))
selected["E0"] = e0
comparison = []
for g in ["E0","E1","E2","E3"]:
    r = selected[g]
    if r is None: continue
    q = {"group":g,"selected_epoch":r["epoch"],"validation_images":r["validation_images"]}
    for key in ("rmse_m","mae_m","absrel","full14_mae_m","gt_ge20cm_14_mae_m","amplitude_ratio","sign_match_rate"):
        q[key]=r[key]
    q["rmse_change_vs_E0_pct"]=(r["rmse_m"]/e0["rmse_m"]-1)*100
    q["full14_change_vs_E0_pct"]=(r["full14_mae_m"]/e0["full14_mae_m"]-1)*100
    q["large14_change_vs_E0_pct"]=(r["gt_ge20cm_14_mae_m"]/e0["gt_ge20cm_14_mae_m"]-1)*100
    e1 = next(x for x in rows if x["group"]=="E1" and x["epoch"]==5)
    q["rmse_change_vs_E1e5_pct"]=(r["rmse_m"]/e1["rmse_m"]-1)*100
    q["full14_change_vs_E1e5_pct"]=(r["full14_mae_m"]/e1["full14_mae_m"]-1)*100
    q["large14_change_vs_E1e5_pct"]=(r["gt_ge20cm_14_mae_m"]/e1["gt_ge20cm_14_mae_m"]-1)*100
    q["sign_change_vs_E0_pp"]=(r["sign_match_rate"]-e0["sign_match_rate"])*100
    comparison.append(q)
with (out/"selected_comparison.csv").open("w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(comparison[0].keys())); w.writeheader(); w.writerows(comparison)
payload={"source":"../validation_summary.csv","selection_rule":"RMSE and full 14px MAE <= 1.05*E0; minimize fixed-sample 14px GT>=0.2m MAE; ties within 1e-6m choose lower RMSE, then earlier epoch","selected":comparison,"epoch_rows":rows,"notes":["Image-level local bootstrap was not computed because per-image local sufficient statistics were not available in the local result bundle.","Validation is Day047 only; image resampling does not establish generalization to unseen acquisition days."]}
(out/"summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
# Compact standalone SVG with one panel per metric and E0 reference.
W,H=1120,1380
left,right=118,1080
top=95
panel_h=238
gap=25
metrics=[
    ("RMSE (m)","rmse_m",False),
    ("Local difference MAE, all 14 px pairs (m)","full14_mae_m",False),
    ("Local difference MAE, |GT difference| >= 0.20 m (m)","gt_ge20cm_14_mae_m",False),
    ("Amplitude ratio, |predicted difference| / |GT difference|","amplitude_ratio",False),
    ("Sign agreement (%)","sign_match_rate",True),
]
colors={"E1":"#2563eb","E2":"#059669","E3":"#d97706"}
def fmt(v, pct=False): return f"{v*100:.1f}" if pct else f"{v:.4f}"
parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="white"/>',
       '<text x="560" y="38" text-anchor="middle" font-family="Arial, sans-serif" font-size="23" font-weight="bold">Local Huber fine-tuning: Day047 validation by epoch</text>',
       '<text x="560" y="65" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">128 images per epoch; E0 is the fixed epoch-20 B0 reference</text>']
# legend
lx=775
for g in groups:
    c=colors[g]
    parts.append(f'<line x1="{lx}" y1="84" x2="{lx+24}" y2="84" stroke="{c}" stroke-width="3"/><text x="{lx+31}" y="89" font-family="Arial" font-size="13">{g}</text>')
    lx+=90
parts.append('<line x1="1035" y1="84" x2="1059" y2="84" stroke="#444" stroke-width="2" stroke-dasharray="6,4"/><text x="1065" y="89" font-family="Arial" font-size="13">E0</text>')
for pi,(title,key,is_pct) in enumerate(metrics):
    y0=top+pi*(panel_h+gap)
    plot_top=y0+35
    plot_bottom=y0+190
    plot_left=left
    plot_right=right-22
    vals=[e0[key]]+[r[key] for r in rows if r["group"] in groups]
    lo,hi=min(vals),max(vals)
    span=hi-lo
    pad=max(span*.16, .0005 if not is_pct else .002)
    lo-=pad; hi+=pad
    def py(v): return plot_bottom-(v-lo)/(hi-lo)*(plot_bottom-plot_top)
    parts.append(f'<text x="{plot_left}" y="{y0+17}" font-family="Arial, sans-serif" font-size="16" font-weight="bold">{html.escape(title)}</text>')
    for j in range(5):
        v=lo+(hi-lo)*j/4
        yy=py(v)
        parts.append(f'<line x1="{plot_left}" y1="{yy:.1f}" x2="{plot_right}" y2="{yy:.1f}" stroke="#dddddd" stroke-width="1"/>')
        parts.append(f'<text x="{plot_left-10}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{fmt(v,is_pct)}</text>')
    parts.append(f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#555"/>')
    parts.append(f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#555"/>')
    e0y=py(e0[key])
    parts.append(f'<line x1="{plot_left}" y1="{e0y:.1f}" x2="{plot_right}" y2="{e0y:.1f}" stroke="#444" stroke-width="2" stroke-dasharray="7,5"/>')
    for i in range(1,6):
        xx=plot_left+(i-1)*(plot_right-plot_left)/4
        parts.append(f'<text x="{xx:.1f}" y="{plot_bottom+19}" text-anchor="middle" font-family="Arial" font-size="12">{i}</text>')
    parts.append(f'<text x="{(plot_left+plot_right)/2:.1f}" y="{plot_bottom+38}" text-anchor="middle" font-family="Arial" font-size="12">Additional-training epoch</text>')
    for g in groups:
        rs=sorted([r for r in rows if r["group"]==g],key=lambda r:r["epoch"])
        pts=[]
        for r in rs:
            xx=plot_left+(r["epoch"]-1)*(plot_right-plot_left)/4
            yy=py(r[key])
            pts.append(f"{xx:.1f},{yy:.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[g]}" stroke-width="2.5"/>')
        for r in rs:
            xx=plot_left+(r["epoch"]-1)*(plot_right-plot_left)/4
            yy=py(r[key])
            parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4.2" fill="{colors[g]}"><title>{g}, epoch {r["epoch"]}: {fmt(r[key],is_pct)}</title></circle>')
parts.append(f'<text x="560" y="{H-20}" text-anchor="middle" font-family="Arial" font-size="12">Reference counts: all-valid 14 px pairs = 5,320,891,492; fixed large-difference 14 px pairs = 1,911,836</text>')
parts.append('</svg>')
(out/"learning_curves.svg").write_text("\n".join(parts),encoding="utf-8")
print(json.dumps({"selected_epochs":{k:(v["epoch"] if v else None) for k,v in selected.items()},"comparison":comparison,"artifacts":[str(out/"selected_comparison.csv"),str(out/"summary.json"),str(out/"learning_curves.svg")]},ensure_ascii=False,indent=2))
