"""DepthTrack-only client splits for the partition-sensitivity supplement.

split_builder draws every dataset from ONE random.Random(seed), in the order
lasher -> depthtrack -> visevent, and the RGB-only pick (stratified) runs after
all three. Building DepthTrack alone (--skip-dataset lasher visevent) therefore
does NOT reproduce the DepthTrack part of the main 12-client split. To keep the
alpha=0.5 cell identical to the main experiment's DepthTrack clients, this script
runs the full 12-client construction with lasher/visevent at alpha=0.5 (as in the
main split) and only the DepthTrack alpha changed, then keeps the 4 DepthTrack
clients. At alpha=0.5 the result must equal the main split's rgbd_* clients;
this is asserted.

    python scripts/supp/build_supp_splits.py --alphas 0.1 0.5 1.0 --outdir configs/fed/client_splits
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from lib.federated import split_builder as sb  # noqa: E402

MAIN_SPLIT = os.path.join(REPO, "configs", "fed", "client_splits",
                          "lasher_depthtrack_visevent_seed42_v2.json")


def build_depth_only(alpha, seed=42):
    args = sb.parse_args(["--seed", str(seed), "--clients-per-dataset", "4",
                          "--strategy", "dirichlet", "--alpha", "0.5",
                          "--rgb-only-ratio", "0.25", "--rgb-only-strategy", "stratified",
                          "--sample-missing-prob", "0.1"])
    orig = sb.split_dirichlet
    calls = {"n": 0}

    def patched(items, n_clients, rng, a):
        calls["n"] += 1                      # 1 = lasher, 2 = depthtrack, 3 = visevent
        return orig(items, n_clients, rng, alpha if calls["n"] == 2 else a)

    sb.split_dirichlet = patched
    try:
        clients, pools = sb.build_clients(args, verbose=False)
    finally:
        sb.split_dirichlet = orig
    assert calls["n"] == 3, calls
    depth = [c for c in clients if c["dataset"] == "depthtrack"]
    for c in depth:
        c["non_iid_tag"] = "dirichlet_alpha_%s" % alpha
    doc = OrderedDict(
        schema_version=1, seed=seed, strategy="dirichlet", alpha=alpha,
        clients_per_dataset=4, rgb_only_ratio=0.25, rgb_only_strategy="stratified",
        sample_missing_prob=0.1, split_level="sequence",
        datasets=OrderedDict(depthtrack=pools["depthtrack"]),
        fingerprint=sb.fingerprint(depth), n_clients=len(depth),
        total_samples=sum(c["num_samples"] for c in depth), clients=depth,
        note=("DepthTrack-only split. Built by the full 12-client split_builder "
              "pipeline (shared RNG stream) with only the DepthTrack Dirichlet alpha "
              "changed; lasher/visevent drawn at alpha=0.5 and discarded."))
    return doc


def check_against_main(doc):
    main = json.load(open(MAIN_SPLIT, encoding="utf-8"))
    ref = {c["client_id"]: c for c in main["clients"] if c["dataset"] == "depthtrack"}
    got = {c["client_id"]: c for c in doc["clients"]}
    assert set(ref) == set(got), (set(ref), set(got))
    for cid in ref:
        for k in ("aux_type", "sequence_ids", "num_samples", "num_sequences", "missing_prob"):
            assert ref[cid][k] == got[cid][k], (cid, k, ref[cid][k], got[cid][k])
    return True


def centralized(doc):
    """One client holding every DepthTrack training sequence, aux fully present
    (apart from the same 10% per-sample dropout every sensor client has)."""
    cs = doc["clients"]
    seqs = sorted(s for c in cs for s in c["sequence_ids"])
    one = OrderedDict(
        client_id="rgbd_all", dataset="depthtrack", aux_type="depth",
        sequence_ids=seqs, num_samples=sum(c["num_samples"] for c in cs),
        num_sequences=len(seqs), missing_prob=0.1, non_iid_tag="centralized",
        train_weight=sum(c["num_samples"] for c in cs))
    out = OrderedDict((k, v) for k, v in doc.items() if k != "clients")
    out.update(strategy="centralized", alpha=None, clients_per_dataset=1,
               rgb_only_ratio=0.0, n_clients=1, total_samples=one["num_samples"],
               fingerprint=sb.fingerprint([one]), clients=[one],
               note="Centralized reference: the union of the DepthTrack clients' sequences as one client.")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    p.add_argument("--outdir", default=os.path.join(REPO, "configs", "fed", "client_splits"))
    a = p.parse_args()
    for alpha in a.alphas:
        doc = build_depth_only(alpha)
        if alpha == 0.5:
            check_against_main(doc)
            print("alpha=0.5 reproduces the main split's DepthTrack clients exactly")
        probs = sb.sanity_check(doc["clients"])
        print("alpha=%s fingerprint=%s %s" % (alpha, doc["fingerprint"], probs or "sanity OK"))
        for c in doc["clients"]:
            print("   %-9s %-5s seqs=%3d frames=%7d" % (c["client_id"], c["aux_type"],
                                                         c["num_sequences"], c["num_samples"]))
        out = os.path.join(a.outdir, "depthtrack_only_alpha%s_seed42.json" % alpha)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print("   ->", out)
        if alpha == 0.5:
            cdoc = centralized(doc)
            cout = os.path.join(a.outdir, "depthtrack_only_centralized_seed42.json")
            with open(cout, "w", encoding="utf-8", newline="\n") as f:
                json.dump(cdoc, f, ensure_ascii=False, indent=1)
                f.write("\n")
            print("   -> %s  (%d seqs, %d frames, fp %s)" % (
                cout, cdoc["clients"][0]["num_sequences"], cdoc["total_samples"], cdoc["fingerprint"]))


if __name__ == "__main__":
    main()
