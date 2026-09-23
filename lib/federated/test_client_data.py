#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for client_data.py.

    D:\\Conda\\envs\\sdstrack\\python.exe lib/federated/test_client_data.py

Uses the REAL datasets on disk and the REAL client_splits.json. A stub would
prove nothing here: the whole point of this module is that the three official
dataset classes each list their sequences differently, so the tests have to
exercise the actual classes.

Set FED_DATA_FAST=1 to skip the tests that decode image files.
"""

from __future__ import annotations

import os
import sys
import traceback

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from lib.federated.client_data import (  # noqa: E402
    ClientDataError,
    build_client_dataset,
    build_dataset,
    load_split,
    restrict_to_sequences,
)

FAST = os.environ.get("FED_DATA_FAST") == "1"
SPLIT = os.path.join(REPO, "configs", "fed", "client_splits",
                     "lasher_depthtrack_visevent_seed42.json")

_CACHE = {}


def _ds(name):
    """Datasets are expensive to construct; build each once per process.

    Callers MUST NOT mutate the cached object -- restrict_to_sequences works in
    place. Tests that narrow a dataset build their own.
    """
    if name not in _CACHE:
        _CACHE[name] = build_dataset(name)
    return _CACHE[name]


# ------------------------------------------------------------------ building

def test_all_three_datasets_build():
    for name, expect_min in (("lasher", 800), ("depthtrack", 140), ("visevent", 400)):
        d = _ds(name)
        n = d.get_num_sequences()
        assert n >= expect_min, f"{name}: only {n} sequences"
        assert len(d.sequence_list) == n


def test_unknown_dataset_is_rejected():
    try:
        build_dataset("nonexistent")
    except ClientDataError:
        return
    raise AssertionError("unknown dataset name was accepted")


def test_bad_root_is_rejected():
    try:
        build_dataset("lasher", root=r"D:\cvresearch\data\does_not_exist")
    except ClientDataError:
        return
    raise AssertionError("nonexistent root was accepted")


# --------------------------------------------------------------- restricting

def test_restrict_selects_exactly_the_named_sequences():
    d = build_dataset("lasher")
    want = [d.sequence_list[i] for i in (5, 100, 3)]
    idx = restrict_to_sequences(d, want)
    assert d.sequence_list == want, d.sequence_list
    assert idx == [5, 100, 3], idx
    assert d.get_num_sequences() == 3


def test_restrict_preserves_the_requested_order():
    d = build_dataset("lasher")
    want = [d.sequence_list[i] for i in (200, 1, 50)]
    restrict_to_sequences(d, want)
    assert d.sequence_list == want


def test_restrict_rejects_unknown_name():
    d = build_dataset("lasher")
    try:
        restrict_to_sequences(d, [d.sequence_list[0], "this_sequence_does_not_exist"])
    except ClientDataError as exc:
        assert "not in the dataset list" in str(exc)
        return
    raise AssertionError("an unknown sequence name was accepted")


def test_restrict_rebuilds_depthtrack_class_structures():
    """DepthTrack maps class names to POSITIONS in sequence_list. Stale maps
    would point at the wrong sequences, silently."""
    d = build_dataset("depthtrack")
    assert d.has_class_info(), "expected DepthTrack to carry class info"
    want = [d.sequence_list[i] for i in (10, 11, 30, 31, 32)]
    restrict_to_sequences(d, want)

    # every index in seq_per_class must be in range and carry that class
    for cls, ids in d.seq_per_class.items():
        for i in ids:
            assert 0 <= i < len(d.sequence_list), f"{cls}: index {i} out of range"
            assert d.sequence_list[i].split("_")[0] == cls, \
                f"{cls}: index {i} is {d.sequence_list[i]}"
    # class_to_id must agree with class_list
    assert set(d.class_to_id) == set(d.class_list)
    assert sorted(d.class_to_id.values()) == list(range(len(d.class_list)))
    # get_class_name must not raise for any surviving sequence
    for i in range(d.get_num_sequences()):
        assert isinstance(d.get_class_name(i), str)


# ------------------------------------------------------------------- the json

def test_split_file_loads():
    doc, by_id = load_split(SPLIT)
    assert doc["n_clients"] == len(by_id) == 12, doc["n_clients"]
    assert doc["fingerprint"] == "bf493c676d666971", doc["fingerprint"]


def test_every_client_in_the_real_split_can_be_built():
    """The one test that would have caught a name/index mismatch."""
    _, by_id = load_split(SPLIT)
    for cid, c in by_id.items():
        d = build_client_dataset(c)
        assert d.get_num_sequences() == len(c["sequence_ids"]), cid
        assert d.sequence_list == list(c["sequence_ids"]), cid


def test_clients_of_one_dataset_do_not_share_sequences():
    """Sequence-level splitting means no sequence may appear twice."""
    _, by_id = load_split(SPLIT)
    seen = {}
    for cid, c in by_id.items():
        for s in c["sequence_ids"]:
            key = (c["dataset"], s)
            assert key not in seen, f"{s} in both {seen[key]} and {cid}"
            seen[key] = cid


def test_declared_num_samples_matches_annotation_frames():
    """num_samples drives the aggregation weight, so it must not drift from the
    data. Checked against the annotation files, per dataset."""
    from lib.federated.split_builder import DATASETS, count_frames
    _, by_id = load_split(SPLIT)
    for cid, c in by_id.items():
        cfg = DATASETS[c["dataset"]]
        total = sum(count_frames(os.path.join(cfg["root"], s), cfg["anno"])
                    for s in c["sequence_ids"])
        assert total == c["num_samples"], \
            f"{cid}: declared {c['num_samples']}, counted {total}"


# ------------------------------------------------------------------- reading

def test_restricted_dataset_returns_six_channel_frames():
    """End-to-end: a narrowed dataset must still decode images and give ViPT the
    packed 6-channel frame it expects."""
    if FAST:
        return
    _, by_id = load_split(SPLIT)
    # one client per dataset, so all three dtypes are exercised
    picked = {}
    for cid, c in by_id.items():
        picked.setdefault(c["dataset"], c)
    for name, c in picked.items():
        d = build_client_dataset(c)
        info = d.get_sequence_info(0)
        assert "bbox" in info, name
        frames, anno, _ = d.get_frames(0, [0], info)
        img = frames[0]
        assert img.ndim == 3 and img.shape[2] == 6, \
            f"{name}: expected (H,W,6), got {img.shape}"
        assert len(anno["bbox"]) == 1, name


def test_first_sequence_of_a_client_is_the_one_declared():
    """get_frames(0, ...) must read the client's OWN first sequence."""
    if FAST:
        return
    _, by_id = load_split(SPLIT)
    c = by_id["rgbt_001"]
    d = build_client_dataset(c)
    assert d.sequence_list[0] == c["sequence_ids"][0]
    path = d._get_sequence_path(0) if hasattr(d, "_get_sequence_path") else None
    if path is not None:
        assert os.path.basename(os.path.normpath(path)) == c["sequence_ids"][0], path


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    if FAST:
        print("FED_DATA_FAST=1 -> image-decoding tests skipped\n")
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            print(f"  [OK]   {name}")
            passed += 1
        except Exception:
            print(f"  [FAIL] {name}")
            traceback.print_exc()
            failed.append(name)
    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        print("failed: " + ", ".join(failed))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
