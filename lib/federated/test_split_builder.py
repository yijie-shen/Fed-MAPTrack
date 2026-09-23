#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for split_builder.py (pure functions, no dataset needed).

    D:\\Conda\\envs\\sdstrack\\python.exe lib/federated/test_split_builder.py

The split is the one artefact every later experiment depends on, and its two
failure modes are both silent:
  * a sequence landing in two clients -> the same video trains twice, and any
    "held-out" comparison between those clients is meaningless;
  * a non-reproducible split -> two runs that were supposed to differ in one
    variable actually also differ in who held what.
So the tests below check the guards, and each guard has a NEGATIVE test proving
it would actually fire.
"""

from __future__ import annotations

import os
import random
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lib.federated.split_builder import (  # noqa: E402
    fingerprint,
    sanity_check,
    split_dirichlet,
    split_uniform,
)


def _items(n):
    """(sequence_name, n_frames) with deliberately uneven lengths -- real
    tracking sequences differ by more than 30x."""
    return [(f"seq{i:03d}", 100 + (i * 37) % 900) for i in range(n)]


def _client(cid, seqs, samples=1000, dataset="lasher", aux="thermal"):
    return dict(client_id=cid, dataset=dataset, aux_type=aux,
                sequence_ids=list(seqs), num_samples=samples,
                num_sequences=len(seqs))


# ---------------------------------------------------------------- uniform

def test_uniform_uses_every_item_exactly_once():
    items = _items(50)
    buckets = split_uniform(items, 4, random.Random(0))
    flat = [s for b in buckets for s, _ in b]
    assert sorted(flat) == sorted(s for s, _ in items)
    assert len(flat) == len(set(flat)), "a sequence appeared twice"


def test_uniform_sizes_differ_by_at_most_one():
    buckets = split_uniform(_items(50), 4, random.Random(0))
    sizes = sorted(len(b) for b in buckets)
    assert sizes[-1] - sizes[0] <= 1, sizes


def test_uniform_is_seed_deterministic():
    a = split_uniform(_items(50), 4, random.Random(7))
    b = split_uniform(_items(50), 4, random.Random(7))
    c = split_uniform(_items(50), 4, random.Random(8))
    assert [[s for s, _ in x] for x in a] == [[s for s, _ in x] for x in b]
    assert [[s for s, _ in x] for x in a] != [[s for s, _ in x] for x in c]


# ---------------------------------------------------------------- dirichlet

def test_dirichlet_uses_every_item_exactly_once():
    items = _items(200)
    buckets = split_dirichlet(items, 5, random.Random(1), alpha=0.5)
    flat = [s for b in buckets for s, _ in b]
    assert sorted(flat) == sorted(s for s, _ in items)
    assert len(flat) == len(set(flat))


def test_dirichlet_every_client_gets_at_least_one():
    """A zero-sequence client divides by zero in sample weighting."""
    for seed in range(20):
        buckets = split_dirichlet(_items(30), 8, random.Random(seed), alpha=0.05)
        assert all(len(b) >= 1 for b in buckets), f"seed {seed}: {[len(b) for b in buckets]}"


def test_dirichlet_small_alpha_is_more_skewed():
    """The whole point of alpha: it must actually control skew."""
    def top_share(alpha, seed):
        b = split_dirichlet(_items(300), 6, random.Random(seed), alpha=alpha)
        sizes = sorted((len(x) for x in b), reverse=True)
        return sizes[0] / sum(sizes)

    skewed = sum(top_share(0.1, s) for s in range(12)) / 12
    even = sum(top_share(100.0, s) for s in range(12)) / 12
    assert skewed > even + 0.10, f"alpha=0.1 share {skewed:.3f} vs alpha=100 {even:.3f}"


def test_dirichlet_rejects_more_clients_than_sequences():
    try:
        split_dirichlet(_items(3), 8, random.Random(0), alpha=0.5)
    except ValueError:
        return
    raise AssertionError("expected ValueError when clients > sequences")


def test_dirichlet_is_seed_deterministic():
    a = split_dirichlet(_items(100), 4, random.Random(3), alpha=0.5)
    b = split_dirichlet(_items(100), 4, random.Random(3), alpha=0.5)
    assert [[s for s, _ in x] for x in a] == [[s for s, _ in x] for x in b]


# ---------------------------------------------------------------- guards

def test_sanity_check_passes_on_a_clean_split():
    clients = [_client("c0", ["a", "b"]), _client("c1", ["c"])]
    assert sanity_check(clients) == []


def test_sanity_check_detects_shared_sequence():
    """The leakage guard must actually fire -- otherwise the clean-split test
    above proves nothing."""
    clients = [_client("c0", ["a", "b"]), _client("c1", ["b", "c"])]
    problems = sanity_check(clients)
    assert any("both" in p and "b" in p for p in problems), problems


def test_sanity_check_allows_same_name_in_different_datasets():
    """Two datasets may legitimately contain a sequence with the same name;
    only a clash within one dataset is leakage."""
    clients = [
        _client("c0", ["dup"], dataset="lasher", aux="thermal"),
        _client("c1", ["dup"], dataset="visevent", aux="event"),
    ]
    assert sanity_check(clients) == []


def test_sanity_check_detects_empty_client():
    problems = sanity_check([_client("c0", [])])
    assert any("no sequence" in p for p in problems), problems


def test_sanity_check_detects_zero_samples():
    problems = sanity_check([_client("c0", ["a"], samples=0)])
    assert any("num_samples" in p for p in problems), problems


def test_sanity_check_detects_duplicate_client_id():
    problems = sanity_check([_client("c0", ["a"]), _client("c0", ["b"])])
    assert any("duplicate client_id" in p for p in problems), problems


# ---------------------------------------------------------------- fingerprint

def test_fingerprint_is_stable_and_order_independent():
    a = [_client("c0", ["x", "y"]), _client("c1", ["z"])]
    b = [_client("c1", ["z"]), _client("c0", ["x", "y"])]   # different order
    assert fingerprint(a) == fingerprint(b)


def test_fingerprint_changes_when_assignment_changes():
    a = [_client("c0", ["x", "y"]), _client("c1", ["z"])]
    b = [_client("c0", ["x"]), _client("c1", ["y", "z"])]
    assert fingerprint(a) != fingerprint(b)


def test_fingerprint_changes_when_aux_type_changes():
    """An RGB-only conversion must be visible in the fingerprint, otherwise two
    different experiments could claim the same split id."""
    a = [_client("c0", ["x"], aux="thermal")]
    b = [_client("c0", ["x"], aux="none")]
    assert fingerprint(a) != fingerprint(b)


# ---------------------------------------------------------------- runner

def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
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
