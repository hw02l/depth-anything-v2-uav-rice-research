"""Windows CPU analysis of hash-verified lightweight results. No model or image inference."""
import argparse,csv,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from uav_metric_research import multiscale as m


def table(path):return list(csv.DictReader(Path(path).open(encoding='utf-8-sig')))
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def number(value):return None if value in (None,'') else float(value)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    figures=args.output/'figures'
    figures.mkdir(exist_ok=True)
    selection=read(args.results/'pilot_selection_lock.json')
    tile_label=selection['selected_tile_label']
    labels=['B0','B1_1036',tile_label]+(['B3'] if selection['run_B3'] else [])
    pilot_indices={int(r['image_index']) for r in table(m.ROOT/'experiments/multiscale_tile_inference_20261016/pilot_selection.csv')}
    diagnostic=table(args.results/'diagnostic_A/multiscale_metrics.csv')
    diagnostic_per=table(args.results/'diagnostic_A/per_image_metrics.csv')
    groups,per,comparison,runtimes,paired,depth_groups,local_per=[],[],[],[],[],[],[]
    for epoch in (17,20):
        for label in labels:
            run=args.results/(f'B0_ep{epoch}' if label=='B0' else f'{label}_full_ep{epoch}')
            summary=read(run/'summary.json')
            if summary['images']!=128 or summary['epoch']!=epoch:raise ValueError('Incomplete comparison')
            local=[r for r in diagnostic if r['model']==f'epoch{epoch}'] if label=='B0' else table(run/'multiscale_metrics.csv')
            image_local=[r for r in diagnostic_per if r['model']==f'epoch{epoch}'] if label=='B0' else table(run/'multiscale_per_image.csv')
            groups.extend(dict(epoch=epoch,method=label,**{k:v for k,v in r.items() if k!='model'}) for r in local)
            absolute=table(run/'per_image_metrics.csv')
            absolute_by_index={int(r['image_index']):r for r in absolute}
            for r in image_local:
                row=absolute_by_index[int(r['image_index'])]
                local_per.append(dict(epoch=epoch,method=label,image_relpath=row['image_relpath'],crop=row['crop'],view=row['view'],
                    **{k:v for k,v in r.items() if k not in ('model','crop','view','image_relpath')}))
            for row in absolute:
                i=int(row['image_index'])
                large=next(r for r in image_local if int(r['image_index'])==i and r['stride_px']=='14' and r['group']=='gt_ge20cm')
                border=next(r for r in image_local if int(r['image_index'])==i and r['stride_px']=='14' and r['group']=='border_lt128')
                interior=next(r for r in image_local if int(r['image_index'])==i and r['stride_px']=='14' and r['group']=='interior_ge512')
                per.append(dict(epoch=epoch,method=label,**row,large14_pairs=int(large['pair_count']),
                    large14_mae_m=number(large['delta_mae_m']),large14_amplitude_ratio=number(large['amplitude_ratio']),
                    large14_sign_rate=number(large['sign_match_rate']),border14_mae_m=number(border['delta_mae_m']),
                    interior14_mae_m=number(interior['delta_mae_m'])))
            for r in table(run/'group_metrics.csv'):
                scope=r['scope']
                members=[a for a in absolute if scope in ('crop='+a['crop'],'view='+a['view'],a['crop']+'|'+a['view'])]
                small=next(v for v in local if v['scope']==scope and v['stride_px']=='14' and v['group']=='all')
                large=next(v for v in local if v['scope']==scope and v['stride_px']=='14' and v['group']=='gt_ge20cm')
                depth_groups.append(dict(epoch=epoch,method=label,**{k:v for k,v in r.items() if k!='images'},images=len(members),
                    fixed_sample_14_pairs=int(small['pair_count']),fixed_sample_large14_pairs=int(large['pair_count']),
                    large14_mae_m=number(large['delta_mae_m']),large14_amplitude_ratio=number(large['amplitude_ratio']),large14_sign_rate=number(large['sign_match_rate'])))
            timing=table(run/'runtime_metrics.csv')
            runtimes.extend(dict(epoch=epoch,method=label,**r) for r in timing)
            benchmark=table(args.results/f'B3_end_to_end_benchmark_ep{epoch}.csv') if label=='B3' else [r for r in timing if int(r['image_index']) in pilot_indices]
            if {int(r['image_index']) for r in benchmark}!=pilot_indices:raise ValueError('Cost comparison must use the same 16 images')
            large=next(r for r in local if r['scope']=='all' and r['stride_px']=='14' and r['group']=='gt_ge20cm')
            comparison.append(dict(epoch=epoch,method=label,images=128,**summary['metrics'],large14_pairs=int(large['pair_count']),
                large14_mae_m=number(large['delta_mae_m']),large14_rmse_m=number(large['delta_rmse_m']),
                large14_amplitude_ratio=number(large['amplitude_ratio']),large14_sign_rate=number(large['sign_match_rate']),
                seconds_mean=float(np.mean([float(r['seconds']) for r in timing])),
                seconds_median=float(np.median([float(r['seconds']) for r in timing])),
                peak_allocated_mib=max(int(r['peak_allocated_bytes']) for r in timing)/1024**2,
                benchmark_images=len(benchmark),benchmark_seconds_mean=float(np.mean([float(r['seconds']) for r in benchmark])),
                benchmark_peak_allocated_mib=max(int(r['peak_allocated_bytes']) for r in benchmark)/1024**2,
                benchmark_peak_reserved_mib=max(int(r['peak_reserved_bytes']) for r in benchmark)/1024**2,
                runtime_definition='component sum for B3; measured preprocess/infer/restore/merge otherwise; excludes I/O and scoring'))
    for epoch in (17,20):
        for label in labels[1:]:
            a=sorted([r for r in per if r['epoch']==epoch and r['method']=='B0'],key=lambda r:int(r['image_index']))
            b=sorted([r for r in per if r['epoch']==epoch and r['method']==label],key=lambda r:int(r['image_index']))
            if [r['image_index'] for r in a]!=[r['image_index'] for r in b]:raise ValueError('Paired IDs differ')
            for metric in ('rmse_m','mae_m','local_depth_delta_mae_m','large14_mae_m','border14_mae_m','interior14_mae_m'):
                if any((number(x[metric]) is None)!=(number(y[metric]) is None) for x,y in zip(a,b)):
                    raise ValueError('Method-specific undefined metric; GT mask differs')
                defined=[(x,y) for x,y in zip(a,b) if number(x[metric]) is not None]
                missing=[int(x['image_index']) for x,y in zip(a,b) if number(x[metric]) is None]
                if missing and metric!='large14_mae_m':raise ValueError('Unexpected undefined non-stratified metric')
                diff=[number(y[metric])-number(x[metric]) for x,y in defined]
                result=m.paired_bootstrap(diff)
                ci=result.pop('ci95')
                weighted={}
                if metric=='large14_mae_m':
                    if [r['large14_pairs'] for r in a]!=[r['large14_pairs'] for r in b]:raise ValueError('Pair counts differ')
                    weighted=m.paired_weighted_bootstrap([r[metric] for r in a],[r[metric] for r in b],[r['large14_pairs'] for r in a])
                paired.append(dict(epoch=epoch,method=label,metric=metric,total_images=len(a),undefined_image_ids=missing,
                    ci95_low=ci[0],ci95_high=ci[1],**result,**weighted))
            for stride in m.SCALES:
                a=sorted([r for r in local_per if r['epoch']==epoch and r['method']=='B0' and int(r['stride_px'])==stride and r['group']=='gt_ge20cm'],key=lambda r:int(r['image_index']))
                b=sorted([r for r in local_per if r['epoch']==epoch and r['method']==label and int(r['stride_px'])==stride and r['group']=='gt_ge20cm'],key=lambda r:int(r['image_index']))
                if [r['pair_count'] for r in a]!=[r['pair_count'] for r in b]:raise ValueError('Paired stratum counts differ')
                defined=[(x,y) for x,y in zip(a,b) if int(x['pair_count'])>0]
                missing=[int(x['image_index']) for x,y in zip(a,b) if int(x['pair_count'])==0]
                result=m.paired_bootstrap([float(y['delta_mae_m'])-float(x['delta_mae_m']) for x,y in defined])
                ci=result.pop('ci95')
                weighted=m.paired_weighted_bootstrap([number(r['delta_mae_m']) for r in a],[number(r['delta_mae_m']) for r in b],[int(r['pair_count']) for r in a])
                paired.append(dict(epoch=epoch,method=label,metric='sampled_large_delta_mae_m',stride_px=stride,total_images=len(a),
                    undefined_image_ids=missing,ci95_low=ci[0],ci95_high=ci[1],**result,**weighted))
    m.write_csv(args.output/'method_comparison.csv',comparison)
    m.write_csv(args.output/'multiscale_metrics.csv',groups)
    m.write_csv(args.output/'per_image_metrics.csv',per)
    m.write_csv(args.output/'per_image_multiscale.csv',local_per)
    m.write_csv(args.output/'group_metrics.csv',depth_groups)
    m.write_csv(args.output/'runtime_metrics.csv',runtimes)
    m.write_csv(args.output/'paired_differences.csv',paired)
    variability=[]
    for epoch,label,stride,group in sorted({(r['epoch'],r['method'],int(r['stride_px']),r['group']) for r in local_per if r['group'] in ('all','gt_ge20cm','border_lt128','interior_ge512')}):
        selected=[r for r in local_per if r['epoch']==epoch and r['method']==label and int(r['stride_px'])==stride and r['group']==group]
        values=np.array([float(r['delta_mae_m']) for r in selected if int(r['pair_count'])>0])
        variability.append(dict(epoch=epoch,method=label,stride_px=stride,group=group,images=len(selected),nonempty_images=len(values),
            empty_images=len(selected)-len(values),image_mean_delta_mae_m=float(values.mean()) if len(values) else None,
            image_std_delta_mae_m=float(values.std(ddof=1)) if len(values)>1 else None,
            image_median_delta_mae_m=float(np.median(values)) if len(values) else None,
            image_min_delta_mae_m=float(values.min()) if len(values) else None,image_max_delta_mae_m=float(values.max()) if len(values) else None))
    m.write_csv(args.output/'image_variability.csv',variability)
    m.write_json(args.output/'analysis_summary.json',dict(selection=selection,comparison=comparison,paired=paired,
        independent_test_used=False,historical52_used=False,day='Day047',images_per_method=128))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=dict(zip(labels,['#244c88','#cf7434','#279383','#984e9c']))
    short={'B0':'B0 518','B1_1036':'B1 1036',tile_label:'B2 tiles','B3':'B3 fusion'}
    def find(epoch,label,stride,group,scope='all'):
        return next(r for r in groups if r['epoch']==epoch and r['method']==label and int(r['stride_px'])==stride and r['group']==group and r['scope']==scope)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for epoch in (17,18,20):
        values=[next(r for r in diagnostic if r['model']==f'epoch{epoch}' and r['scope']=='all' and int(r['stride_px'])==s and r['group']=='gt_ge20cm') for s in m.SCALES]
        axes[0].plot(m.SCALES,[float(r['delta_mae_m']) for r in values],'o-',label=f'epoch {epoch}')
        axes[1].plot(m.SCALES,[float(r['amplitude_ratio']) for r in values],'o-',label=f'epoch {epoch}')
    axes[0].set(ylabel='Delta MAE (m)');axes[1].set(ylabel='Sum |prediction delta| / sum |GT delta|')
    for ax in axes:ax.set_xlabel('Original-image separation (px)');ax.legend()
    fig.suptitle('A: B0 multiscale diagnosis | GT absolute delta >= 0.20 m | 128 Day047 images')
    fig.tight_layout();fig.savefig(figures/'A_multiscale_large_delta.png',dpi=160);plt.close(fig)
    for epoch in (17,20):
        fig,axes=plt.subplots(2,2,figsize=(11,8))
        for label in labels:
            for col,group in enumerate(('all','gt_ge20cm')):
                values=[find(epoch,label,s,group) for s in m.SCALES]
                axes[0,col].plot(m.SCALES,[float(r['delta_mae_m']) for r in values],'o-',color=colors[label],label=short[label])
                axes[1,col].plot(m.SCALES,[float(r['amplitude_ratio']) for r in values],'o-',color=colors[label])
            axes[0,0].set_title('All fixed sampled pairs');axes[0,1].set_title('|GT delta| >= 0.20 m')
        for ax in axes[0]:ax.set_ylabel('Delta MAE (m)');ax.legend(fontsize=8)
        for ax in axes[1]:ax.set_ylabel('Amplitude ratio');ax.axhline(1,color='gray',linestyle=':',linewidth=1)
        for ax in axes.flat:ax.set_xlabel('Original-image separation (px)')
        fig.suptitle(f'B: Fixed weights epoch {epoch} | Day047 validation, 128 images')
        fig.tight_layout();fig.savefig(figures/f'B_multiscale_ep{epoch}.png',dpi=160);plt.close(fig)
        bins=['gt_delta_0_0.05m','gt_delta_0.05_0.1m','gt_delta_0.1_0.2m','gt_delta_0.2_0.5m','gt_delta_0.5_infm']
        fig,ax=plt.subplots(figsize=(10,4))
        x=np.arange(5);width=.8/len(labels)
        for j,label in enumerate(labels):
            ax.bar(x+(j-(len(labels)-1)/2)*width,[float(find(epoch,label,14,g)['delta_mae_m']) for g in bins],width,color=colors[label],label=short[label])
        counts=[int(find(epoch,'B0',14,g)['pair_count']) for g in bins]
        ax.set_xticks(x,[f'{name}\nn={count:,}' for name,count in zip(['[0,5)','[5,10)','[10,20)','[20,50)','[50,inf)'],counts)],fontsize=8)
        ax.set(xlabel='GT absolute difference bin (cm); shared GT-valid sample counts',ylabel='14px delta MAE (m)',title=f'GT difference strata | epoch {epoch}')
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figures/f'B_delta_bins_ep{epoch}.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,scopes,names in [(axes[0],['crop=corn','crop=oat','crop=wheat','crop=soybean'],['Corn','Oat','Wheat','Soybean']),
                                (axes[1],['view=nadir','view=oblique'],['Nadir','Oblique'])]:
            x=np.arange(len(scopes))
            for j,label in enumerate(labels):
                ax.bar(x+(j-(len(labels)-1)/2)*width,[float(find(epoch,label,14,'gt_ge20cm',s)['delta_mae_m']) for s in scopes],width,color=colors[label],label=short[label])
            ax.set_xticks(x,names);ax.set_ylabel('GT>=20cm 14px delta MAE (m)');ax.legend(fontsize=7)
        fig.suptitle(f'Crop/view strata | epoch {epoch} | 32 images/crop; 64 images/view')
        fig.tight_layout();fig.savefig(figures/f'B_crop_view_ep{epoch}.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,4))
        zones=['border_lt128','middle_128_512','interior_ge512']
        x=np.arange(3)
        for j,label in enumerate(labels):
            ax.bar(x+(j-(len(labels)-1)/2)*width,[float(find(epoch,label,14,g)['delta_mae_m']) for g in zones],width,color=colors[label],label=short[label])
        ax.set_xticks(x,['edge <128px','128px <= edge <512px','edge >=512px'])
        ax.set(ylabel='Sampled 14px delta MAE (m)',title=f'Image edge / interior | epoch {epoch} | minimum endpoint distance')
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figures/f'B_edge_ep{epoch}.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(12,4))
        base={int(r['image_index']):number(r['large14_mae_m']) for r in per if r['epoch']==epoch and r['method']=='B0'}
        for label in labels[1:]:
            selected=sorted([r for r in per if r['epoch']==epoch and r['method']==label],key=lambda r:int(r['image_index']))
            diff=[float(r['large14_mae_m'])-base[int(r['image_index'])] if r['large14_mae_m'] is not None and base[int(r['image_index'])] is not None else np.nan for r in selected]
            ax.plot([int(r['image_index']) for r in selected],diff,'.-',linewidth=.7,markersize=3,color=colors[label],label=short[label])
        ax.axhline(0,color='black',linewidth=.8)
        for boundary in (31.5,63.5,95.5):ax.axvline(boundary,color='gray',linewidth=.5,linestyle=':')
        ax.set(xlabel='Fixed validation image index; crops: Corn | Oat | Wheat | Soybean',ylabel='Image GT>=20cm delta MAE minus B0 (m)',title=f'Paired differences | epoch {epoch} | 126 defined; IDs101/107 have no sampled GT>=20cm pairs')
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figures/f'B_paired_images_ep{epoch}.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for label in labels:
            row=next(r for r in comparison if r['epoch']==epoch and r['method']==label)
            offset=(4,-13) if label=='B0' else (4,6)
            axes[0].scatter(row['benchmark_seconds_mean'],row['large14_mae_m'],color=colors[label],s=60)
            axes[0].annotate(short[label],(row['benchmark_seconds_mean'],row['large14_mae_m']),xytext=offset,textcoords='offset points',fontsize=8)
            axes[1].scatter(row['benchmark_peak_allocated_mib'],row['rmse_m'],color=colors[label],s=60)
            axes[1].annotate(short[label],(row['benchmark_peak_allocated_mib'],row['rmse_m']),xytext=offset,textcoords='offset points',fontsize=8)
        axes[0].set(xlabel='Mean measured processing time on same 16 RGBs (s/image)',ylabel='GT>=20cm 14px delta MAE (m)')
        axes[1].set(xlabel='PyTorch peak allocated VRAM (MiB)',ylabel='Absolute depth RMSE (m)')
        for ax in axes:ax.margins(.2)
        fig.suptitle(f'Accuracy: 128 images | Cost: same 16 RGBs, warm | epoch {epoch} | measured B3 pipeline')
        fig.tight_layout();fig.savefig(figures/f'B_cost_ep{epoch}.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(comparison=comparison,paired=paired),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
