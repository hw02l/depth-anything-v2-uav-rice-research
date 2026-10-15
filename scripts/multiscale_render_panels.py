"""Compose publication figures on Windows from verified OpenCV color panels."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    cases=json.loads((args.panels/'panel_metadata.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    for case in cases:
        i,epoch=case['image_index'],case['epoch']
        names=case['labels']
        display={name:('B2 2000px / nominal overlap25%' if name.startswith('B2_') else 'B1 whole 1036px' if name=='B1_1036' else 'B3 global/local fusion' if name=='B3' else 'B0 whole 518px' if name=='B0' else name) for name in names}
        for zoom in (False,True):
            fig=plt.figure(figsize=(3.6*len(names)+1,10))
            grid=fig.add_gridspec(3,len(names)+1,width_ratios=[1]*len(names)+[.055],
                                  left=.03,right=.95,top=.89,bottom=.07,wspace=.12,hspace=.14)
            axes=np.array([[fig.add_subplot(grid[row,col]) for col in range(len(names))] for row in range(3)])
            roi=case['roi']
            suffixes=['roi_depth','roi_absolute','roi_delta'] if zoom else ['depth','absolute','delta']
            limits=[(roi['depth_min_m'],roi['depth_max_m']) if zoom else (case['depth_min_m'],case['depth_max_m']),
                    (0,case['absolute_error_max_m']),(0,case['delta_error_max_m'])]
            row_labels=['Z-depth (m)','Absolute Z-depth error (m), values >=2 clipped',
                        'Horizontal 14-original-px delta error (m), values >=0.5 clipped']
            for col,label in enumerate(names):
                for row,suffix in enumerate(suffixes):
                    path=args.panels/f'case_{i:03d}_{label}_{suffix}.png'
                    if zoom:
                        extent=(roi['left']-.5,roi['left']+roi['width']-.5,roi['top']+roi['height']-.5,roi['top']-.5)
                        axes[row,col].imshow(plt.imread(path),extent=extent)
                        vertical={v for box in case['tile_boxes'] for v in (box[1],box[3]) if roi['left']<=v<roi['left']+roi['width']}
                        horizontal={v for box in case['tile_boxes'] for v in (box[0],box[2]) if roi['top']<=v<roi['top']+roi['height']}
                        for position in vertical:axes[row,col].axvline(position-.5,color='#ffaf00',linestyle='--',linewidth=.9)
                        for position in horizontal:axes[row,col].axhline(position-.5,color='#ffaf00',linestyle='--',linewidth=.9)
                        axes[row,col].tick_params(labelsize=6)
                    else:
                        axes[row,col].imshow(plt.imread(path))
                        axes[row,col].set_axis_off()
                axes[0,col].set_title(display[label],fontsize=9)
            for row,(lo,hi) in enumerate(limits):
                sm=ScalarMappable(norm=Normalize(lo,hi),cmap='viridis' if row==0 else 'magma')
                cbar=fig.colorbar(sm,cax=fig.add_subplot(grid[row,len(names)]))
                cbar.set_label(row_labels[row],fontsize=8)
                cbar.ax.tick_params(labelsize=7)
            subtitle=(f'512px ROI: x={roi["left"]}..{roi["left"]+511}, y={roi["top"]}..{roi["top"]+511}; orange lines = tile crop boundary references'
                      if zoom else 'Display samples every 8 original pixels; scoring uses all original-resolution values')
            fig.suptitle(f'Epoch {epoch} | {case["image_relpath"]} | {case["crop"]}/{case["view"]}\n{subtitle}',fontsize=12,y=.98)
            fig.text(.5,.015,'Shared GT-derived depth limits across all methods; white = invalid GT. No independent normalization.\n'
                     'Data: Link-Dev, UAV3DCrop / UAV3DCrop Depth (CC BY 4.0); modified: depth coloring and prediction/error comparison.',ha='center',fontsize=8)
            stem=('tile_boundary' if zoom else 'depth_comparison')+f'_ep{epoch}_image{i:03d}'
            fig.savefig(args.output/(stem+'.png'),dpi=145)
            plt.close(fig)


if __name__=='__main__':main()
