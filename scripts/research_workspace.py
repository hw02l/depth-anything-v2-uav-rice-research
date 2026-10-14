"""Offline CLI for local-first research provenance and one-way file exchange."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from uav_metric_research.provenance import verify, make_bundle, import_results

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    check=commands.add_parser('verify')
    check.add_argument('--checkpoint',type=Path)
    bundle=commands.add_parser('bundle')
    bundle.add_argument('--output',type=Path,required=True)
    imports=commands.add_parser('import-results')
    imports.add_argument('--incoming',type=Path,required=True)
    imports.add_argument('--experiment',required=True)
    args=parser.parse_args()
    if args.command=='verify': result=verify(ROOT,args.checkpoint)
    elif args.command=='bundle': result=make_bundle(ROOT,args.output)
    else: result=import_results(ROOT,args.incoming,args.experiment)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
