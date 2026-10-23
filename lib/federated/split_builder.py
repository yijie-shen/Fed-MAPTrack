#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build deterministic federated client splits (spec sections 5, 5.1, 5.2).

Output is one client_splits.json that fully determines who holds what. Every
later run reads that file instead of re-randomising, so an experiment can be
reproduced months later from the json alone.

-------------------------------------------------------------------------
Rule 0, and it is not negotiable: SPLIT BY SEQUENCE, NEVER BY FRAME
-------------------------------------------------------------------------
Frames 100 and 101 of the same video are nearly identical. Putting one in a
training client and the other in a test set is textbook temporal leakage --
the tracker has literally seen the answer. Every number after that is inflated
and the paper is dead. Federated splitting recipes written for image
classification shuffle samples freely; that advice is wrong here.

This module therefore only ever partitions *sequence names*, and the sequence
pool comes from the repo's own train lists, which already exclude the test
benchmarks (spec 5.2: "若官方已划分 train/test,严格遵守官方划分").

-------------------------------------------------------------------------
Client construction (spec section 5)
-------------------------------------------------------------------------
Dataset-wise modality heterogeneity first -- each dataset becomes its own pool
of clients holding one sensor:

    LasHeR      -> aux_type 'thermal'   client ids rgbt_000 ...
    DepthTrack  -> aux_type 'depth'     client ids rgbd_000 ...
    VisEvent    -> aux_type 'event'     client ids rgbe_000 ...

Then a configurable fraction of clients is converted to RGB-only
(aux_type='none') by masking their auxiliary modality, which is what the
"incomplete modality" experiments need.

Non-IID within a dataset:
    uniform    every client gets a near-equal number of sequences (IID baseline)
    dirichlet  the PROPORTION of sequences per client is drawn from
               Dirichlet(alpha). Small alpha -> a few clients hold most of the
               data. This is quantity skew, the axis that actually exists for
               tracking: there are no class labels to skew over, and sequence
               length varies by more than 30x across LasHeR, so quantity skew
               is both natural and severe.

`num_samples` is the real frame count, read from each sequence's annotation
file -- not the number of sequences. It is what the aggregator weights by
(spec 9.2: weight_i = num_samples_i), and sequences differ enormously in
length, so using sequence counts would badly misweight the average.

-------------------------------------------------------------------------
Usage
-------------------------------------------------------------------------
    python -m lib.federated.split_builder --preview
    python -m lib.federated.split_builder --seed 42 --clients-per-dataset 4 \
        --strategy dirichlet --alpha 0.5 --rgb-only-ratio 0.25 \
        --out configs/fed/client_splits/lasher_depthtrack_visevent_seed42.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import OrderedDict
from datetime import datetime

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CVR_ROOT = os.environ.get("CVR_ROOT", REPO)
DATA = os.environ.get("CVR_DATA", os.path.join(CVR_ROOT, "data"))
SPECS = os.path.join(REPO, "lib", "train", "data_specs")

NO_AUX = "none"

#: dataset -> (aux_type, client id prefix, sequence root, how to list sequences,
#:             annotation file used to count frames)
DATASETS = OrderedDict(
    lasher=dict(
        aux_type="thermal", prefix="rgbt",
        root=os.path.join(DATA, "lasher", "trainingset"),
        list_file=os.path.join(SPECS, "lasher_train.txt"),
        anno="visible.txt",
    ),
    depthtrack=dict(
        aux_type="depth", prefix="rgbd",
        root=os.path.join(DATA, "depthtrack", "train"),
        list_file=os.path.join(SPECS, "depthtrack_train.txt"),
        anno="groundtruth.txt",
    ),
    visevent=dict(
        aux_type="event", prefix="rgbe",
        root=os.path.join(DATA, "visevent", "train_subset"),
        list_file=os.path.join(DATA, "visevent", "train_subset", "trainlist.txt"),
        anno="groundtruth.txt",
    ),
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------- io helpers

def read_sequence_list(path):
    """Read a sequence-name list. Handles the CRLF line endings the repo ships."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f.read().splitlines() if ln.strip()]


def count_frames(seq_dir, anno_name):
    """Number of frames for which both modalities and annotation exist.

    Returns 0 if the file is missing, and the caller decides what to do; we do
    NOT silently invent a count, because num_samples drives aggregation weights.
    """
    p = os.path.join(seq_dir, anno_name)
    if not os.path.isfile(p):
        return 0
    n = 0
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                n += 1
    if anno_name == "visible.txt":
        dirs = ("visible", "infrared")
    elif os.path.isdir(os.path.join(seq_dir, "color")):
        dirs = ("color", "depth")
    elif os.path.isdir(os.path.join(seq_dir, "vis_imgs")):
        dirs = ("vis_imgs", "event_imgs")
    else:
        return n
    available = [sum(os.path.splitext(x)[1].lower() in _IMAGE_EXTS
                     for x in os.listdir(os.path.join(seq_dir, d)))
                 if os.path.isdir(os.path.join(seq_dir, d)) else 0 for d in dirs]
    return min([n] + available)


def load_pool(name, cfg, cache=None, verbose=True):
    """[(sequence_name, n_frames)] for one dataset, sorted by name.

    Sorting matters: it makes the split depend only on the seed, not on
    filesystem enumeration order.
    """
    if cache and name in cache:
        return [tuple(x) for x in cache[name]]

    seqs = sorted(read_sequence_list(cfg["list_file"]))
    out, missing = [], []
    for s in seqs:
        n = count_frames(os.path.join(cfg["root"], s), cfg["anno"])
        if n == 0:
            missing.append(s)
            continue
        out.append((s, n))
    if verbose and missing:
        print(f"  [{name}] {len(missing)} sequences skipped (no {cfg['anno']}): "
              f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")
    return out


# ---------------------------------------------------------------- splitting

def split_uniform(items, n_clients, rng):
    """Shuffle once, then deal round-robin so client sizes differ by <= 1."""
    items = list(items)
    rng.shuffle(items)
    buckets = [[] for _ in range(n_clients)]
    for i, it in enumerate(items):
        buckets[i % n_clients].append(it)
    return buckets


def split_dirichlet(items, n_clients, rng, alpha):
    """Proportions ~ Dirichlet(alpha), then hand out sequences to match.

    Implemented without numpy: Dirichlet(alpha) == normalised independent
    Gamma(alpha, 1) draws, and random.gammavariate is seeded by the same rng,
    so the result is reproducible from the seed alone.

    Every client is guaranteed at least one sequence -- a zero-sequence client
    would divide by zero in the aggregator's sample weighting and is never what
    the experiment intends.
    """
    items = list(items)
    rng.shuffle(items)
    if n_clients > len(items):
        raise ValueError(
            f"asked for {n_clients} clients but only {len(items)} sequences; "
            f"a client with no sequence cannot be weighted by num_samples"
        )

    props = [rng.gammavariate(alpha, 1.0) for _ in range(n_clients)]
    total = sum(props) or 1.0
    props = [p / total for p in props]

    # Largest-remainder apportionment, so the counts sum exactly to len(items).
    raw = [p * len(items) for p in props]
    counts = [max(1, int(r)) for r in raw]
    while sum(counts) > len(items):                 # give back from the biggest
        counts[counts.index(max(counts))] -= 1
    rema = sorted(range(n_clients), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    i = 0
    while sum(counts) < len(items):
        counts[rema[i % n_clients]] += 1
        i += 1

    buckets, pos = [], 0
    for c in counts:
        buckets.append(items[pos:pos + c])
        pos += c
    return buckets


# ---------------------------------------------------------------- build

def build_clients(args, verbose=True):
    rng = random.Random(args.seed)
    clients, pools_meta = [], OrderedDict()

    for name, cfg in DATASETS.items():
        if name in args.skip_dataset:
            continue
        pool = load_pool(name, cfg, verbose=verbose)
        if not pool:
            print(f"  [{name}] !! empty pool, skipped (is the data extracted?)")
            continue
        pools_meta[name] = dict(
            root=cfg["root"], aux_type=cfg["aux_type"],
            n_sequences=len(pool), n_frames=sum(n for _, n in pool),
        )

        if args.strategy == "uniform":
            buckets = split_uniform(pool, args.clients_per_dataset, rng)
        else:
            buckets = split_dirichlet(pool, args.clients_per_dataset, rng, args.alpha)

        for i, bucket in enumerate(buckets):
            clients.append(dict(
                client_id=f"{cfg['prefix']}_{i:03d}",
                dataset=name,
                aux_type=cfg["aux_type"],
                sequence_ids=sorted(s for s, _ in bucket),
                num_samples=sum(n for _, n in bucket),
                num_sequences=len(bucket),
                missing_prob=0.0,
                non_iid_tag=(f"dirichlet_alpha_{args.alpha}"
                             if args.strategy == "dirichlet" else "uniform"),
                train_weight=1.0,
            ))

    # -- convert a fraction of clients to RGB-only (spec section 5) ------------
    #
    # Selection is STRATIFIED BY DATASET by default, and that is a deliberate
    # departure from "just sample k clients at random".
    #
    # Why: with 12 clients and ratio 0.2, plain random sampling picked both
    # RGB-only clients out of LasHeR (seed 42). Thermal then lost 30% of its
    # frames while depth and event lost none. That confounds the variable under
    # study ("how much missing modality") with a variable nobody intended
    # ("which modality went missing"), and the ablation can no longer attribute
    # the effect. Stratifying spreads the masking across sensors so the only
    # thing changing is the ratio.
    #
    # --rgb-only-strategy random reproduces the naive behaviour, for anyone who
    # explicitly wants to study an unbalanced-missingness scenario.
    if args.rgb_only_ratio > 0 and clients:
        k = int(round(len(clients) * args.rgb_only_ratio))
        picked = []
        if k and args.rgb_only_strategy == "stratified":
            by_ds = OrderedDict()
            for i, c in enumerate(clients):
                by_ds.setdefault(c["dataset"], []).append(i)
            # Largest-remainder apportionment of k across datasets, so the
            # per-dataset counts sum exactly to k.
            total = len(clients)
            raw = {ds: k * len(idxs) / total for ds, idxs in by_ds.items()}
            quota = {ds: int(v) for ds, v in raw.items()}
            order = sorted(by_ds, key=lambda ds: raw[ds] - quota[ds], reverse=True)
            j = 0
            while sum(quota.values()) < k:
                quota[order[j % len(order)]] += 1
                j += 1
            for ds, idxs in by_ds.items():
                n = min(quota[ds], len(idxs))
                if n:
                    picked.extend(rng.sample(sorted(idxs), n))
        elif k:
            picked = rng.sample(sorted(range(len(clients))), k)

        for idx in picked:
            clients[idx]["aux_type"] = NO_AUX
            clients[idx]["aux_masked_from"] = DATASETS[clients[idx]["dataset"]]["aux_type"]

    # -- sample-level missing (spec 3.2) --------------------------------------
    for c in clients:
        if c["aux_type"] != NO_AUX:
            c["missing_prob"] = args.sample_missing_prob

    return clients, pools_meta


def sanity_check(clients):
    """Fail loudly on the mistakes that would invalidate the experiment."""
    problems = []

    seen = {}
    for c in clients:
        for s in c["sequence_ids"]:
            key = (c["dataset"], s)
            if key in seen:
                problems.append(f"sequence {s} of {c['dataset']} is in both "
                                f"{seen[key]} and {c['client_id']}")
            seen[key] = c["client_id"]

    for c in clients:
        if c["num_sequences"] == 0:
            problems.append(f"{c['client_id']} holds no sequence")
        if c["num_samples"] <= 0:
            problems.append(f"{c['client_id']} has num_samples={c['num_samples']}")

    ids = [c["client_id"] for c in clients]
    if len(ids) != len(set(ids)):
        problems.append("duplicate client_id")
    return problems


def fingerprint(clients):
    """Stable hash of the assignment, to prove two runs used the same split."""
    payload = json.dumps(
        [[c["client_id"], c["dataset"], c["aux_type"], c["sequence_ids"]]
         for c in sorted(clients, key=lambda x: x["client_id"])],
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def make_document(args, clients, pools_meta):
    return OrderedDict(
        schema_version=1,
        created_at=datetime.now().isoformat(timespec="seconds"),
        seed=args.seed,
        strategy=args.strategy,
        alpha=args.alpha if args.strategy == "dirichlet" else None,
        clients_per_dataset=args.clients_per_dataset,
        rgb_only_ratio=args.rgb_only_ratio,
        rgb_only_strategy=args.rgb_only_strategy,
        sample_missing_prob=args.sample_missing_prob,
        split_level="sequence",   # never 'frame' -- see module docstring
        datasets=pools_meta,
        fingerprint=fingerprint(clients),
        n_clients=len(clients),
        total_samples=sum(c["num_samples"] for c in clients),
        clients=clients,
    )


def print_report(doc):
    print(f"\nseed={doc['seed']}  strategy={doc['strategy']}"
          f"{'  alpha=' + str(doc['alpha']) if doc['alpha'] else ''}"
          f"  fingerprint={doc['fingerprint']}")
    print(f"{doc['n_clients']} clients, {doc['total_samples']:,} annotated frames total\n")
    print(f"{'client_id':<12}{'dataset':<12}{'aux_type':<9}{'seqs':>6}{'frames':>12}{'share':>8}")
    total = doc["total_samples"] or 1
    for c in doc["clients"]:
        print(f"{c['client_id']:<12}{c['dataset']:<12}{c['aux_type']:<9}"
              f"{c['num_sequences']:>6}{c['num_samples']:>12,}"
              f"{c['num_samples']/total*100:>7.1f}%")
    by_aux = OrderedDict()
    for c in doc["clients"]:
        a = by_aux.setdefault(c["aux_type"], [0, 0])
        a[0] += 1
        a[1] += c["num_samples"]
    print("\nby aux_type:")
    for a, (n, f) in by_aux.items():
        print(f"  {a:<9} {n:>3} clients  {f:>12,} frames  {f/total*100:5.1f}%")


# ---------------------------------------------------------------- cli

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build federated client splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clients-per-dataset", type=int, default=4)
    p.add_argument("--strategy", choices=["uniform", "dirichlet"], default="dirichlet")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Dirichlet concentration; smaller = more skewed")
    p.add_argument("--rgb-only-ratio", type=float, default=0.0,
                   help="fraction of clients whose aux modality is masked away")
    p.add_argument("--rgb-only-strategy", choices=["stratified", "random"],
                   default="stratified",
                   help="stratified spreads the masking evenly across datasets so "
                        "the ablation isolates 'how much is missing' from 'which "
                        "modality is missing'; random is the naive version")
    p.add_argument("--sample-missing-prob", type=float, default=0.0,
                   help="per-sample aux dropout inside a client that has aux")
    p.add_argument("--skip-dataset", nargs="*", default=[],
                   choices=list(DATASETS), help="datasets to leave out")
    p.add_argument("--out", default=None, help="where to write client_splits.json")
    p.add_argument("--preview", action="store_true",
                   help="print the split but do not write a file")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("Building client splits (sequence-level, never frame-level)")
    clients, pools_meta = build_clients(args)
    if not clients:
        print("!! no clients built -- check that the datasets are extracted")
        return 1

    problems = sanity_check(clients)
    doc = make_document(args, clients, pools_meta)
    print_report(doc)

    if problems:
        print("\n!! sanity check FAILED:")
        for p in problems[:20]:
            print(f"   {p}")
        return 1
    print("\nsanity check OK: no sequence shared between clients, "
          "every client non-empty")

    if args.preview:
        print("(--preview, nothing written)")
        return 0

    out = args.out or os.path.join(
        REPO, "configs", "fed", "client_splits",
        f"seed{args.seed}_{args.strategy}_{args.clients_per_dataset}pc.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
