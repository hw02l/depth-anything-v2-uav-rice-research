from run_experiment2 import *
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',type=Path,required=True); ap.add_argument('--work',type=Path,required=True); ap.add_argument('--run-trial',action='store_true'); args=ap.parse_args()
    assert args.run_trial and torch.cuda.is_available()
    project=args.project; work=args.work; source=Path(__file__).parent
    plan=project/'uav_metric_finetuning/config/verified_plan'; lr=Path('/content/uav3dcrop/rgb'); ld=Path('/content/uav3dcrop/depth')
    assert json.loads((work/'ssd_integrity.json').read_text())['status']=='pass'
    assert dataset_status(plan,lr,ld,['train','validation'])['complete']
    assert json.loads((work/'output_cap_probe.json').read_text())['actual_median_ratio']==0.8125
    run=project/'runs/metric_outdoor_trial2_20261013_s42_518_cap80_gt65'; assert not run.exists(), 'Existing run preserved'
    metric=project/'model_cache/models--depth-anything--Depth-Anything-V2-Metric-Outdoor-Small-hf/snapshots/fd2c22027eaf20374204f14099b8341e1925ad39'
    common=['--initialization','metric_outdoor','--metric-model',str(metric),'--max-depth','65','--model-max-depth','80','--image-size','518','--batch-size','1','--amp']
    preflight=work/'metric_outdoor_preflight_retry.json'
    common=['--initialization','metric_outdoor','--metric-model',str(metric),'--max-depth','65','--model-max-depth','80','--image-size','518','--batch-size','1','--amp']
    subprocess.run([sys.executable,'-u',str(source/'memory_preflight.py'),*common,'--output',str(preflight)],check=True)
    assert json.loads(preflight.read_text())['status']=='pass'
    train_start=time.monotonic()
    subprocess.run([sys.executable,'-u',str(source/'train.py'),*common,'--train-manifest',str(plan/'train.csv'),'--val-manifest',str(plan/'validation.csv'),'--rgb-root',str(lr),'--depth-root',str(ld),'--epochs','1','--run-kind','trial','--grad-accumulation','4','--seed','42','--workers','2','--encoder-lr','5e-6','--save-every-epochs','1','--keep-recent-checkpoints','0','--memory-preflight-report',str(preflight),'--output-dir',str(run)],check=True)
    assert json.loads((run/'run_status.json').read_text())['status']=='completed'
    checkpoints={}
    for cp in run.glob('checkpoint_epoch_*'):
        marker=json.loads((cp/'complete.json').read_text())
        assert all(sha(cp/n)==h for n,h in marker['file_sha256'].items())
        checkpoints[cp.name]=marker
    newjson(work/'trial_complete.json',{'run':str(run),'training_total_seconds_including_startup_save':time.monotonic()-train_start,'checkpoints_verified':checkpoints,'model_output_ceiling_m':80,'gt_mask_max_m':65,'epochs':1,'test_used':False})
    spec={'manifest':str(plan/'validation.csv'),'validation_sha256':sha(plan/'validation.csv'),'rgb_root':str(lr),'depth_root':str(ld),'output':str(work/'analysis'),'models':{'A':str(project/'runs/relative_encoder_trial_20261013_s42_518/checkpoint_epoch_0001'),'B':str(metric),'C':str(run/'checkpoint_epoch_0001')},'model_caps':{'A':65,'B':80,'C':80}}
    newjson(work/'analysis_spec.json',spec)
    subprocess.run([sys.executable,'-u',str(source/'shape_analysis.py'),'--spec',str(work/'analysis_spec.json'),'--phase','infer'],check=True)
    print('ALL_GPU_WORK_COMPLETE. Persisted training and prediction cache. Switch to CPU for analysis.',flush=True)

if __name__=='__main__':main()
