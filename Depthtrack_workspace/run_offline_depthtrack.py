import os, sys, time
import argparse
import torch
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1]))
from lib.test.evaluation import Tracker
from lib.train.dataset.depth_utils import get_rgbd_frame

ROOT=Path(os.environ.get('CVR_DATA', os.path.join(os.environ.get('CVR_ROOT', str(Path(__file__).parents[1])), 'data'))) / 'depthtrack' / 'test'; OUT=Path(__file__).parent/'offline_results'
def run(yaml_name, checkpoint=None, output_dir=None, sequence=None, config_yaml=None):
    torch.set_num_threads(1)
    # 2026-10-18: config_yaml picks the experiments/vipt/<name>.yaml to build the model from, while yaml_name stays the
    # experiment name used for the output folder (VOT analysis expects 'rgbd-unsupervised'). FED-PRIME checkpoints
    # need the *_prime yaml (token pools); None = old behaviour.
    tr=Tracker('vipt',config_yaml or yaml_name,'vot22',None); params=tr.get_parameters()
    if checkpoint is not None:
        params.checkpoint = checkpoint
    print('[params] checkpoint ->', params.checkpoint, flush=True)
    out_root = Path(output_dir) if output_dir is not None else OUT
    out_root.mkdir(parents=True, exist_ok=True)
    params.visualization=False; params.debug=0; tracker=tr.create_tracker(params)
    sequence_dirs = sorted(p for p in ROOT.iterdir() if p.is_dir())
    if sequence is not None:
        sequence_dirs = [p for p in sequence_dirs if p.name == sequence]
        if len(sequence_dirs) != 1:
            raise ValueError(f'expected exactly one sequence named {sequence!r}')
    for d in sequence_dirs:
        gt=np.loadtxt(d/'groundtruth.txt',delimiter=',',ndmin=2); names=sorted((d/'color').glob('*.jpg')); dep=sorted((d/'depth').glob('*.png')); od=out_root/yaml_name/d.name; od.mkdir(parents=True,exist_ok=True)
        boxes=['1']; scores=['']; times=[0]
        first=get_rgbd_frame(str(names[0]),str(dep[0]),dtype='rgbcolormap',depth_clip=True); tracker.initialize(first,{'init_bbox':gt[0].tolist()})
        for color,depth in zip(names[1:],dep[1:]):
            t0=time.time(); image=get_rgbd_frame(str(color),str(depth),dtype='rgbcolormap',depth_clip=True); out=tracker.track(image)
            boxes.append(','.join(str(float(v)) for v in out['target_bbox'])); scores.append(float(out['best_score'])); times.append(time.time()-t0)
        (od/f'{d.name}_001.txt').write_text('\n'.join(boxes)+'\n',encoding='utf-8'); (od/f'{d.name}_001_confidence.value').write_text('\n'.join(str(v) for v in scores)+'\n',encoding='utf-8'); (od/f'{d.name}_001_time.value').write_text('\n'.join(str(v) for v in times)+'\n',encoding='utf-8')
        print(yaml_name,d.name,flush=True)
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_name', nargs='?', default='deep_rgbd')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output-dir')
    parser.add_argument('--sequence')
    parser.add_argument('--config-yaml', default=None)
    args = parser.parse_args()
    run(args.yaml_name, args.checkpoint, args.output_dir, args.sequence, args.config_yaml)
