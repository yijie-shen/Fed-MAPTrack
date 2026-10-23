"""VisEvent OPE evaluator matching the official MATLAB toolkit.

Rules copied from utils/eval_tracker.m and calc_seq_err_robust.m:
pixel PR thresholds 0..50, IoU SR thresholds 0..1 by .05; first prediction
is replaced by GT; absent frames are removed from the numerator; invalid boxes
from frame 2 onward repeat the previous box; results are truncated to GT
length; per-sequence curves use the original GT length as denominator; final
curves are the unweighted mean over sequences. No NPR or sequence filtering.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from lasher_eval import _matlab_colon

THR_OVERLAP = _matlab_colon(0.0, 0.05, 1.0)

def iou(a, b):
    l = np.maximum(a[:, 0], b[:, 0]); t = np.maximum(a[:, 1], b[:, 1])
    r = np.minimum(a[:, 0] + a[:, 2] - 1, b[:, 0] + b[:, 2] - 1)
    bot = np.minimum(a[:, 1] + a[:, 3] - 1, b[:, 1] + b[:, 3] - 1)
    inter = np.maximum(0, r-l+1) * np.maximum(0, bot-t+1)
    return inter / (a[:,2]*a[:,3] + b[:,2]*b[:,3] - inter)

def evaluate(anno_dir: Path, result_dir: Path, sequence_file: Path):
    seqs = [x.strip() for x in sequence_file.read_text().splitlines() if x.strip()]
    pr = np.zeros((len(seqs), 51)); sr = np.zeros((len(seqs), 21))
    for j, s in enumerate(seqs):
        gt = np.loadtxt(anno_dir/'gt_rect'/f'{s}.txt', delimiter=',', ndmin=2)
        absent = np.loadtxt(anno_dir/'absent'/f'{s}.txt', delimiter=',', ndmin=1)
        res = np.loadtxt(result_dir/f'{s}.txt', delimiter=',', ndmin=2)
        assert res.shape[1] == 4, (s, res.shape)
        assert len(res) >= len(gt), (s, len(res), len(gt))
        res = res[:len(gt)].copy()
        for i in range(1, len(gt)):
            if (np.isnan(res[i]).any() or res[i,2] <= 0 or res[i,3] <= 0) and not np.isnan(gt[i]).any():
                res[i] = res[i-1]
        res[0] = gt[0]
        keep = absent != 1
        rr, gg = res[keep], gt[keep]
        valid = np.all(gg > 0, axis=1)
        ov = np.full(len(gg), -1.0); ov[valid] = iou(rr[valid], gg[valid])
        c = np.column_stack((rr[:,0]+(rr[:,2]-1)/2, rr[:,1]+(rr[:,3]-1)/2))
        cg = np.column_stack((gg[:,0]+(gg[:,2]-1)/2, gg[:,1]+(gg[:,3]-1)/2))
        err = np.sqrt(((c-cg)**2).sum(axis=1)); err[~valid] = -1
        pr[j] = [(err <= x).sum()/(len(gt)+np.finfo(float).eps) for x in range(51)]
        sr[j] = [(ov > x).sum()/(len(gt)+np.finfo(float).eps) for x in THR_OVERLAP]
    pcurve, scurve = pr.mean(0), sr.mean(0)
    # Official plot_draw_save.m calls mean(bb), not numerical integration.
    return pcurve[20], scurve.mean(), pcurve, scurve

def sequence_scores(anno_dir, result_dir, seqs):
    out = {}
    for s in seqs:
        gt = np.loadtxt(Path(anno_dir)/'gt_rect'/f'{s}.txt', delimiter=',', ndmin=2)
        ab = np.loadtxt(Path(anno_dir)/'absent'/f'{s}.txt', delimiter=',', ndmin=1)
        r = np.loadtxt(Path(result_dir)/f'{s}.txt', delimiter=',', ndmin=2)
        assert len(r) >= len(gt), (s, len(r), len(gt)); r=r[:len(gt)].copy()
        for i in range(1,len(gt)):
            if (np.isnan(r[i]).any() or r[i,2]<=0 or r[i,3]<=0) and not np.isnan(gt[i]).any(): r[i]=r[i-1]
        r[0]=gt[0]; keep=ab!=1; rr,gg=r[keep],gt[keep]; valid=np.all(gg>0,axis=1)
        l=np.maximum(rr[:,0],gg[:,0]); t=np.maximum(rr[:,1],gg[:,1]); right=np.minimum(rr[:,0]+rr[:,2]-1,gg[:,0]+gg[:,2]-1); bot=np.minimum(rr[:,1]+rr[:,3]-1,gg[:,1]+gg[:,3]-1)
        inter=np.maximum(0,right-l+1)*np.maximum(0,bot-t+1); ov=np.full(len(gg),-1.); ov[valid]=inter[valid]/(rr[valid,2]*rr[valid,3]+gg[valid,2]*gg[valid,3]-inter[valid])
        c=np.c_[rr[:,0]+(rr[:,2]-1)/2,rr[:,1]+(rr[:,3]-1)/2]; cg=np.c_[gg[:,0]+(gg[:,2]-1)/2,gg[:,1]+(gg[:,3]-1)/2]; e=np.sqrt(((c-cg)**2).sum(1)); e[~valid]=-1
        out[s]=((e<=20).sum()/(len(gt)+np.finfo(float).eps), np.mean([(ov>x).sum()/(len(gt)+np.finfo(float).eps) for x in THR_OVERLAP]))
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--toolkit', required=True); ap.add_argument('--results', required=True); a=ap.parse_args()
    root=Path(a.toolkit); p,s,pc,sc=evaluate(root/'annos',Path(a.results),root/'sequence_evaluation_config'/'VisEvent_testing_subset.txt')
    print(f'PR@20={p:.9f} ({p*100:.3f}%) SR_AUC={s:.9f} ({s*100:.3f}%)')
