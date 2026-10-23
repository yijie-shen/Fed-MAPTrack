"""LasHeR PR / NPR / SR evaluation -- a line-by-line Python port of the official
MATLAB toolkit (LasHeR_Evaluation_Toolkit).

Why this file exists
--------------------
The official toolkit is MATLAB and this machine has no MATLAB/Octave. Rather
than invent a metric, every formula below is transcribed from the toolkit source
so our numbers are comparable to published ones. Source map:

  calc_rect_int.m          -> _rect_iou()
  calc_seq_err_robust.m    -> _seq_err()
  eval_tracker.m           -> per-sequence curves in evaluate()
  plot_draw_save.m         -> the final scalar (mean over sequences, then index)
  run_tracker_performance_evaluation.m -> thresholds, rank_idx, norm_dst

Quirks of the official metric that are REPRODUCED ON PURPOSE (do not "fix"):

  1. Boxes use the inclusive-pixel convention: right = x + w - 1, and the
     intersection adds +1. Centre is x + (w-1)/2.
  2. Frame 0 of the prediction is overwritten with ground truth
     (rect_mat(1,:) = rect_anno(1,:)), i.e. the init frame is never scored.
  3. Frames whose GT row is not strictly all-positive (LasHeR marks an absent
     target as 0,0,0,0) get err_coverage = -1 and err_center = -1. Because the
     counters are (coverage > thr) and (center <= thr), such frames count as
     FAILURE for SR but as SUCCESS for PR at every threshold. This asymmetry is
     in the official code and inflates PR; all published LasHeR PR numbers
     carry it, so we keep it. Reported separately as absent_frac.
  4. The denominator is the full sequence length, not the number of valid
     frames (success_num / (len_all + eps)).
  5. A prediction row that is NaN or has w<=0/h<=0 is replaced by the previous
     row, but only when the GT row has no NaN.
  6. plot_draw_save drops any sequence whose whole curve sums to <= eps before
     averaging (aa = aa(sum(aa,2)>eps, :)). Done here too, per metric.

Scalars, matching run_tracker_performance_evaluation.m:
  PR   = error curve at threshold 20 px            (rank_idx 21, 1-indexed)
  NPR  = normalised error curve at threshold 0.20  (norm_dst = true)
  SR50 = overlap curve at threshold 0.50           (rank_idx 11, 1-indexed)
  AUC  = mean of the overlap curve over 0:0.05:1   (ranking_type = AUC)

The toolkit Readme.txt warns that the SR curve produced in the same run as PR is
"not accurate" -- that refers to needing ranking_type=AUC for the success plot.
We compute both SR50 and AUC so there is no ambiguity about which number is
being quoted.
"""

import argparse
import os
import sys

import numpy as np

EPS = np.finfo(float).eps


def _matlab_colon(a, d, b):
    """Reproduce MATLAB's a:d:b bit-for-bit.

    MATLAB does NOT build the vector by repeated a+i*d; it fills the lower half
    from `a` and the upper half from `b` so that both endpoints come out exact:

        for i <= n/2 : v(i) = a + i*d
        for i >  n/2 : v(i) = b - (n-i)*d

    This is not pedantry. With 0:0.05:1 the two constructions disagree at five
    positions, and at index 13 the naive value 0.65000000000000002 is ABOVE the
    double nearest 0.65 while MATLAB's is 0.64999999999999991, which is below.
    Tracker boxes are integers, so IoU values like 13/20 land exactly on
    double(0.65) -- and `err_coverage > threshold` then flips. Measured on
    mfDiMP this moved 4 frames of `doginrain` and changed that sequence's curve
    by 1.09e-2. Using np.arange here made the port disagree with the toolkit;
    using this makes it exact (verified in scripts/lasher_eval_validate.py
    against MATLAB's own tmp_mat/*.mat over 12 trackers x 1224 sequences).
    """
    n = int(np.floor((b - a) / d))
    v = np.empty(n + 1, dtype=float)
    half = n // 2
    for i in range(n + 1):
        v[i] = a + i * d if i <= half else b - (n - i) * d
    return v


THR_OVERLAP = _matlab_colon(0.0, 0.05, 1.0)    # 0:0.05:1   -> 21 values
THR_ERROR = np.arange(0, 51, dtype=float)      # 0:50       -> 51 values, integers
                                               # are exact so no colon subtlety;
                                               # norm_dst divides this by 100,
                                               # exactly as the toolkit does.

RANK_IDX_ERROR = 20    # MATLAB rank_idx 21 -> threshold 20 px
RANK_IDX_OVERLAP = 10  # MATLAB rank_idx 11 -> threshold 0.50


def _loadtxt(path):
    """dlmread() equivalent: sniff the delimiter instead of assuming one.

    The toolkit annos/ are comma-separated CRLF; our tracker writes
    space-separated via np.savetxt. dlmread auto-detects, so we must too.
    """
    with open(path, "r") as f:
        head = f.readline()
    if not head.strip():
        return np.zeros((0, 4))
    delim = "," if "," in head else ("\t" if "\t" in head else None)
    return np.loadtxt(path, delimiter=delim, ndmin=2)


def _rect_iou(a, b):
    """calc_rect_int.m -- inclusive-pixel IoU (note the +1 terms)."""
    left_a, bottom_a = a[:, 0], a[:, 1]
    right_a = left_a + a[:, 2] - 1
    top_a = bottom_a + a[:, 3] - 1

    left_b, bottom_b = b[:, 0], b[:, 1]
    right_b = left_b + b[:, 2] - 1
    top_b = bottom_b + b[:, 3] - 1

    inter = (np.maximum(0, np.minimum(right_a, right_b) - np.maximum(left_a, left_b) + 1)
             * np.maximum(0, np.minimum(top_a, top_b) - np.maximum(bottom_a, bottom_b) + 1))
    area_a = a[:, 2] * a[:, 3]
    area_b = b[:, 2] * b[:, 3]
    return inter / (area_a + area_b - inter)


def _seq_err(results, anno, norm_dst):
    """calc_seq_err_robust.m"""
    seq_length = anno.shape[0]

    # MATLAB: results = results(1:size(rect_anno,1), :) -- truncate only.
    # Fewer prediction rows than GT rows would be an index error there; we
    # surface it instead of silently padding, because it means the image list
    # and the annotation disagree (see the frame-alignment pitfall in CLAUDE.md).
    if results.shape[0] < seq_length:
        raise ValueError("prediction has %d rows, GT has %d -- cannot score"
                         % (results.shape[0], seq_length))
    results = np.array(results[:seq_length], dtype=float, copy=True)
    anno = np.asarray(anno, dtype=float)

    # replace invalid predictions with the previous frame (sequential, in place)
    for i in range(1, seq_length):
        r = results[i]
        bad = bool(np.isnan(r).any() or r[2] <= 0 or r[3] <= 0)
        if bad and not np.isnan(anno[i]).any():
            results[i] = results[i - 1]

    rect_mat = results
    rect_mat[0] = anno[0]      # init frame is never scored

    center_gt = np.stack([anno[:, 0] + (anno[:, 2] - 1) / 2.0,
                          anno[:, 1] + (anno[:, 3] - 1) / 2.0], axis=1)
    center = np.stack([rect_mat[:, 0] + (rect_mat[:, 2] - 1) / 2.0,
                       rect_mat[:, 1] + (rect_mat[:, 3] - 1) / 2.0], axis=1)

    idx = (anno > 0).sum(axis=1) == 4

    # norm_dst divides the centres by the GT width/height. On absent frames
    # LasHeR writes 0,0,0,0, so this is 0/0 -> NaN, and the NaN then propagates
    # through the subtraction below. That is exactly what MATLAB does, and it is
    # harmless because every row with a zero in it has idx == False and its
    # err_center is overwritten with -1 a few lines down. Silenced so that a
    # real numerical problem later is not lost in expected noise.
    with np.errstate(divide="ignore", invalid="ignore"):
        if norm_dst:
            center = center / anno[:, 2:4]
            center_gt = center_gt / anno[:, 2:4]
        err_center = np.sqrt(np.sum((center - center_gt) ** 2, axis=1))
    err_coverage = -np.ones(seq_length)
    if idx.any():
        err_coverage[idx] = _rect_iou(rect_mat[idx], anno[idx])
    err_center[~idx] = -1.0
    return err_coverage, err_center, idx


def evaluate(seq_names, anno_dir, result_dir, norm_dst=False, verbose=False):
    """eval_tracker.m -- returns per-sequence curves plus bookkeeping."""
    thr_err = THR_ERROR / 100.0 if norm_dst else THR_ERROR

    curves_ovl, curves_err, kept, absent = [], [], [], []
    missing, failed = [], []

    for name in seq_names:
        ap = os.path.join(anno_dir, name + ".txt")
        rp = os.path.join(result_dir, name + ".txt")
        if not os.path.isfile(rp):
            missing.append(name)
            continue
        anno = _loadtxt(ap)
        res = _loadtxt(rp)
        if res.size == 0:            # MATLAB: if isempty(res), break; end
            failed.append((name, "empty result file"))
            continue
        try:
            cov, cen, idx = _seq_err(res, anno, norm_dst)
        except ValueError as exc:
            failed.append((name, str(exc)))
            continue

        len_all = anno.shape[0]
        c_ovl = np.array([(cov > t).sum() for t in THR_OVERLAP]) / (len_all + EPS)
        c_err = np.array([(cen <= t).sum() for t in thr_err]) / (len_all + EPS)
        curves_ovl.append(c_ovl)
        curves_err.append(c_err)
        kept.append(name)
        absent.append(1.0 - idx.mean())
        if verbose:
            print("  %-40s frames=%-6d PR20=%.4f SR50=%.4f"
                  % (name, len_all, c_err[RANK_IDX_ERROR], c_ovl[RANK_IDX_OVERLAP]))

    return {
        "curves_ovl": np.array(curves_ovl),
        "curves_err": np.array(curves_err),
        "names": kept,
        "absent_frac": np.array(absent),
        "missing": missing,
        "failed": failed,
    }


def _mean_curve(curves):
    """plot_draw_save.m -- drop all-zero sequences, then average."""
    if curves.size == 0:
        return np.zeros(0), 0
    keep = curves.sum(axis=1) > EPS
    return curves[keep].mean(axis=0), int(keep.sum())


def score(seq_names, anno_dir, result_dir, verbose=False):
    plain = evaluate(seq_names, anno_dir, result_dir, norm_dst=False, verbose=verbose)
    norm = evaluate(seq_names, anno_dir, result_dir, norm_dst=True)

    mc_err, n_err = _mean_curve(plain["curves_err"])
    mc_ovl, n_ovl = _mean_curve(plain["curves_ovl"])
    mc_nerr, _ = _mean_curve(norm["curves_err"])

    return {
        "PR": float(mc_err[RANK_IDX_ERROR]) if mc_err.size else float("nan"),
        "NPR": float(mc_nerr[RANK_IDX_ERROR]) if mc_nerr.size else float("nan"),
        "SR50": float(mc_ovl[RANK_IDX_OVERLAP]) if mc_ovl.size else float("nan"),
        "AUC": float(mc_ovl.mean()) if mc_ovl.size else float("nan"),
        "n_scored": len(plain["names"]),
        "n_after_zero_filter_err": n_err,
        "n_after_zero_filter_ovl": n_ovl,
        "missing": plain["missing"],
        "failed": plain["failed"],
        "absent_frac_mean": (float(plain["absent_frac"].mean())
                             if len(plain["absent_frac"]) else 0.0),
        "curves_ovl": plain["curves_ovl"],
        "curves_err": plain["curves_err"],
        "names": plain["names"],
    }


def main():
    p = argparse.ArgumentParser(
        description="LasHeR PR/NPR/SR (port of the official MATLAB toolkit)")
    p.add_argument("--results", required=True, help="directory of <seq>.txt predictions")
    p.add_argument("--toolkit", default=os.path.join(os.environ.get("CVR_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "tools", "LasHeR_Toolkit"))
    p.add_argument("--seq-list", default="all_dataset_lashertest.txt",
                   help="file under sequence_evaluation_config/ "
                        "(default: the 245-sequence test set)")
    p.add_argument("--verbose", action="store_true", help="print per-sequence scores")
    p.add_argument("--save-npz", default="", help="also dump the raw curves here")
    args = p.parse_args()

    anno_dir = os.path.join(args.toolkit, "annos")
    list_path = os.path.join(args.toolkit, "sequence_evaluation_config", args.seq_list)
    for path in (anno_dir, list_path):
        if not os.path.exists(path):
            sys.exit("not found: %s" % path)

    with open(list_path) as f:
        seqs = [ln.strip() for ln in f if ln.strip()]

    print("sequences in list : %d" % len(seqs))
    print("annos             : %s" % anno_dir)
    print("results           : %s" % args.results)
    print("")

    r = score(seqs, anno_dir, args.results, verbose=args.verbose)

    print("")
    print("scored            : %d / %d" % (r["n_scored"], len(seqs)))
    if r["missing"]:
        print("MISSING result    : %d  e.g. %s" % (len(r["missing"]), r["missing"][:5]))
    if r["failed"]:
        print("FAILED to score   : %d" % len(r["failed"]))
        for n, why in r["failed"][:10]:
            print("    %-40s %s" % (n, why))
    print("after zero-filter : err=%d  ovl=%d" % (r["n_after_zero_filter_err"],
                                                  r["n_after_zero_filter_ovl"]))
    print("absent-GT frames  : %.2f%% (these count as PR success, see module docstring)"
          % (100.0 * r["absent_frac_mean"]))
    print("")
    print("  PR   (20 px)   = %.4f   (%.1f)" % (r["PR"], 100 * r["PR"]))
    print("  NPR  (0.20)    = %.4f   (%.1f)" % (r["NPR"], 100 * r["NPR"]))
    print("  SR50 (IoU 0.5) = %.4f   (%.1f)" % (r["SR50"], 100 * r["SR50"]))
    print("  AUC  (success) = %.4f   (%.1f)" % (r["AUC"], 100 * r["AUC"]))

    if args.save_npz:
        np.savez(args.save_npz, curves_ovl=r["curves_ovl"], curves_err=r["curves_err"],
                 names=np.array(r["names"]))
        print("")
        print("curves -> %s" % args.save_npz)


if __name__ == "__main__":
    main()
