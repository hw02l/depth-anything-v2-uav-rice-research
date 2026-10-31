"""Render measured GT-only reference figures. Does not imply model inference."""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from uav_metric_research import multiscale as m


def main():
    out=m.ROOT/'experiments/multiscale_tile_inference_20261016'
    figures=out/'figures'
    figures.mkdir(exist_ok=True)
    rows=list(csv.DictReader((out/'gt_reference_groups.csv').open(encoding='utf-8')))
    def lookup(scope,group,stride):
        return next(r for r in rows if r['scope']==scope and r['group']==group and int(r['stride_px'])==stride)
    scales=list(m.SCALES)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(scales,[float(lookup('all','all',s)['zero_delta_mae_m']) for s in scales],'o-',label='All fixed sampled pairs')
    axes[0].plot(scales,[float(lookup('all','gt_ge20cm',s)['zero_delta_mae_m']) for s in scales],'s-',label='GT absolute delta >= 0.20 m')
    axes[0].set(xlabel='Separation in original image (px)',ylabel='Zero-delta MAE = mean |GT delta| (m)')
    axes[0].legend(fontsize=8)
    axes[1].plot(scales,[100*int(lookup('all','gt_ge20cm',s)['pair_count'])/int(lookup('all','all',s)['pair_count']) for s in scales],'o-')
    axes[1].set(xlabel='Separation in original image (px)',ylabel='Pairs with |GT delta| >= 0.20 m (%)',ylim=(0,50))
    fig.suptitle('GT-only measured reference | Day047 validation, 128 images | No model inference')
    fig.tight_layout()
    fig.savefig(figures/'gt_reference_scales.png',dpi=150)
    fig.savefig(figures/'gt_reference_scales.svg')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for scope in ['crop=corn','crop=oat','crop=wheat','crop=soybean']:
        axes[0].plot(scales,[float(lookup(scope,'all',s)['zero_delta_mae_m']) for s in scales],'o-',label=scope[5:])
    for scope in ['view=nadir','view=oblique']:
        axes[1].plot(scales,[float(lookup(scope,'all',s)['zero_delta_mae_m']) for s in scales],'o-',label=scope[5:])
    for ax in axes:
        ax.legend()
        ax.set(xlabel='Original-image separation (px)',ylabel='Zero-delta MAE (m)')
    fig.suptitle('GT-only crop/view reference | Shared fixed samples | No model inference')
    fig.tight_layout()
    fig.savefig(figures/'gt_reference_crop_view.png',dpi=150)
    fig.savefig(figures/'gt_reference_crop_view.svg')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for zone in ['border_lt128','middle_128_512','interior_ge512']:
        ax.plot(scales,[float(lookup('all',zone,s)['zero_delta_mae_m']) for s in scales],'o-',label=zone)
    ax.set(xlabel='Original-image separation (px)',ylabel='Zero-delta MAE (m)',title='GT-only edge reference | Minimum edge distance of either endpoint')
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures/'gt_reference_edge.png',dpi=150)
    fig.savefig(figures/'gt_reference_edge.svg')
    plt.close(fig)
    summary=[]
    for s in scales:
        all_=lookup('all','all',s)
        large=lookup('all','gt_ge20cm',s)
        summary.append(dict(stride_px=s,valid_pairs=int(all_['pair_count']),zero_delta_mae_m=float(all_['zero_delta_mae_m']),
                        zero_delta_rmse_m=float(all_['delta_rmse_m']),large_pairs=int(large['pair_count']),
                        large_pair_fraction=int(large['pair_count'])/int(all_['pair_count']),
                        large_mean_abs_gt_delta_m=float(large['mean_abs_gt_delta_m'])))
    m.write_csv(out/'gt_reference_summary.csv',summary)


if __name__=='__main__':main()
