from pathlib import Path
import json,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('--results-dir',type=Path,default=Path(__file__).parent);ap.add_argument('--output',type=Path,default=None);args=ap.parse_args()
root=args.results_dir
s=json.loads((root/'shape_compact.json').read_text())
b=json.loads((root/'border_analysis.json').read_text())
c=json.loads((root/'gpu_phase_receipt.json').read_text())
old=json.loads((root/'previous_validation_summary.json').read_text())
full={'A':old['epoch']['validation'],'B':old['metric_outdoor_original_validation']['metrics'],'C':c['epoch']['validation']}
colors={'A':'#e69f00','B':'#009e73','C':'#d55e00'}
fig,ax=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
x=np.arange(2)
for j,m in enumerate(['A','B','C']):
    ax[0,0].bar(x+(j-1)*.25,[full[m]['mae_m'],full[m]['rmse_m']],width=.25,color=colors[m],label=m)
ax[0,0].set_xticks(x,['MAE','RMSE']);ax[0,0].set_title('Full validation absolute depth error');ax[0,0].set_ylabel('m');ax[0,0].legend()
labels=['0-5cm','5-10cm','10-20cm','>=20cm'];x=np.arange(4)
for j,m in enumerate(['A','B','C']):
    ax[0,1].bar(x+(j-1)*.2,[s['s14'][f'{m}|gt_delta:{label}|s14']['delta_mae_m'] for label in labels],width=.2,color=colors[m],label=m)
ax[0,1].plot(x,[s['s14'][f'A|gt_delta:{label}|s14']['zero_delta_mae_m'] for label in labels],'k--o',label='Zero delta')
ax[0,1].set_xticks(x,labels);ax[0,1].set_title('Local dx/dy14 error by absolute GT delta');ax[0,1].set_ylabel('m');ax[0,1].legend()
zones=['0-128px','128-512px','>=512px'];x=np.arange(3)
for j,m in enumerate(['A','B','C']):ax[1,0].bar(x+(j-1)*.25,[b['stats'][f'{m}|{z}']['delta_mae_m'] for z in zones],width=.25,color=colors[m],label=m)
ax[1,0].set_xticks(x,zones);ax[1,0].set_title('Local error by distance from image border');ax[1,0].set_ylabel('m');ax[1,0].legend()
strides=[1,7,14,28,56,107]
for m in ['A','B','C']:ax[1,1].plot(strides,[s['strides'][f'{m}|all|s{st}']['delta_mae_m'] for st in strides],'o-',color=colors[m],label=m)
ax[1,1].plot(strides,[s['strides'][f'A|all|s{st}']['zero_delta_mae_m'] for st in strides],'k--',label='Zero delta')
ax[1,1].set_xscale('log');ax[1,1].set_xticks(strides,[str(v) for v in strides]);ax[1,1].set_title('Dependence on original-pixel interval');ax[1,1].set_xlabel('Original-image pixels');ax[1,1].set_ylabel('m');ax[1,1].legend()
for a in ax.flat:a.grid(axis='y',alpha=.2)
fig.suptitle('A: relative initialized + 1epoch; B: original Outdoor; C: Outdoor + 1epoch\nAbsolute metrics: full valid pixels. Local metrics: identical sampled pairs.',fontsize=13)
target=args.output if args.output is not None else root/'figures/overview_metrics.png'
target.parent.mkdir(parents=True,exist_ok=True)
assert not target.exists()
fig.savefig(target,dpi=160)
print(target)
