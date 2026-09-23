#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for warmup.py.

    D:\\Conda\\envs\\sdstrack\\python.exe lib/federated/test_warmup.py

A wrong schedule does not crash. It quietly trains the wrong tensors for 30
rounds and the only symptom is a slightly worse final number, which is
indistinguishable from "the method does not work". So the tests below pin down
the things that would be invisible at runtime: which groups are open in each
phase, that a phase gap is rejected rather than silently skipped, and that the
scheduler actually hands the aggregator something it can consume.
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lib.federated.param_groups import (  # noqa: E402
    AUX_SPECIFIC, FUSION, NORM, RGB_SHARED, TRAINABLE_GROUPS,
)
from lib.federated.warmup import (  # noqa: E402
    Phase, RoundPlan, Schedule, WarmupScheduler, default_phases,
)


# ---------------------------------------------------------------- schedules

def test_schedule_constant():
    s = Schedule(0.5)
    assert s.at(0.0) == 0.5 and s.at(1.0) == 0.5


def test_schedule_linear_endpoints_and_midpoint():
    s = Schedule("linear(0.0, 0.5)")
    assert abs(s.at(0.0) - 0.0) < 1e-9
    assert abs(s.at(0.5) - 0.25) < 1e-9
    assert abs(s.at(1.0) - 0.5) < 1e-9


def test_schedule_cosine_is_monotone_and_hits_endpoints():
    s = Schedule("cosine(0.0, 1.0)")
    vals = [s.at(i / 10) for i in range(11)]
    assert abs(vals[0]) < 1e-9 and abs(vals[-1] - 1.0) < 1e-9
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])), vals


def test_schedule_clamps_out_of_range_progress():
    s = Schedule("linear(0.0, 1.0)")
    assert s.at(-5) == 0.0 and s.at(99) == 1.0


def test_schedule_rejects_garbage():
    for bad in ("linear(1)", "exp(0,1)", "nonsense"):
        try:
            Schedule(bad)
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"expected failure for {bad!r}")


# ---------------------------------------------------------------- phase

def test_phase_rejects_unknown_group():
    """A typo'd group name must fail at construction, not silently train
    nothing for the whole phase."""
    try:
        Phase("bad", 0, 10, trainable_groups=("adapter_thermal",))
    except ValueError as e:
        assert "unknown group" in str(e)
        return
    raise AssertionError("expected ValueError for an unknown group")


def test_phase_rejects_empty_range():
    try:
        Phase("bad", 5, 5, trainable_groups=(RGB_SHARED,))
    except ValueError:
        return
    raise AssertionError("expected ValueError for end_round <= start_round")


def test_phase_progress_endpoints():
    p = Phase("p", 10, 20, trainable_groups=(RGB_SHARED,))
    assert p.progress(10) == 0.0
    assert abs(p.progress(19) - 1.0) < 1e-9


def test_single_round_phase_does_not_divide_by_zero():
    p = Phase("p", 3, 4, trainable_groups=(RGB_SHARED,))
    assert p.progress(3) == 0.0


# ---------------------------------------------------------------- scheduler

def test_default_phases_are_contiguous_and_cover_every_round():
    sch = WarmupScheduler.default(120)
    covered = [sch.phase_at(r).name for r in range(120)]
    assert len(covered) == 120
    assert covered[0] == "rgb_common"
    assert covered[-1] == "full"


def test_gap_between_phases_is_rejected():
    """The guard that matters: a gap would leave rounds unscheduled."""
    try:
        WarmupScheduler([
            Phase("a", 0, 10, trainable_groups=(RGB_SHARED,)),
            Phase("b", 12, 20, trainable_groups=tuple(TRAINABLE_GROUPS)),
        ])
    except ValueError as e:
        assert "contiguous" in str(e)
        return
    raise AssertionError("expected ValueError for a phase gap")


def test_phases_must_start_at_zero():
    try:
        WarmupScheduler([Phase("a", 3, 10, trainable_groups=(RGB_SHARED,))])
    except ValueError as e:
        assert "start at round 0" in str(e)
        return
    raise AssertionError("expected ValueError when the first phase skips round 0")


def test_phase1_opens_only_the_rgb_half():
    """The core claim of Phase 1: nothing modality-specific may move."""
    sch = WarmupScheduler.default(120)
    plan = sch.resolve(0)
    assert plan.phase_name == "rgb_common"
    assert AUX_SPECIFIC not in plan.trainable_groups
    assert FUSION not in plan.trainable_groups
    assert RGB_SHARED in plan.trainable_groups and NORM in plan.trainable_groups
    assert plan.modality_aware is False
    assert plan.loss_weights == {"giou": 2.0, "l1": 5.0, "focal": 1.0}


def test_phase2_opens_aux_and_turns_on_modality_awareness():
    sch = WarmupScheduler.default(120)
    r = sch.phases[1].start_round
    plan = sch.resolve(r)
    assert plan.phase_name == "adapter_warmup"
    assert AUX_SPECIFIC in plan.trainable_groups
    assert FUSION not in plan.trainable_groups, "fusion belongs to phase 3"
    assert plan.modality_aware is True


def test_no_unimplemented_aux_loss_weight_is_scheduled():
    sch = WarmupScheduler.default(120)
    for p in sch.phases:
        assert set(p.loss_weights) == {"giou", "l1", "focal"}


def test_final_phase_opens_everything():
    sch = WarmupScheduler.default(120)
    plan = sch.resolve(119)
    assert set(plan.trainable_groups) == set(TRAINABLE_GROUPS)
    assert plan.frozen_groups == ()


def test_frozen_groups_is_the_complement():
    sch = WarmupScheduler.default(120)
    for r in (0, 30, 60, 119):
        plan = sch.resolve(r)
        assert set(plan.trainable_groups) | set(plan.frozen_groups) == set(TRAINABLE_GROUPS)
        assert not (set(plan.trainable_groups) & set(plan.frozen_groups))


def test_beyond_last_round_stays_in_final_phase():
    sch = WarmupScheduler.default(10)
    assert sch.resolve(999).phase_name == sch.phases[-1].name


# ---------------------------------------------------------------- ablation

def test_no_warmup_opens_everything_from_round_zero():
    """The 'Ours minus progressive warm-up' ablation row."""
    sch = WarmupScheduler.no_warmup(120)
    for r in (0, 50, 119):
        plan = sch.resolve(r)
        assert set(plan.trainable_groups) == set(TRAINABLE_GROUPS)
        assert plan.phase_name == "no_warmup"


def test_no_warmup_differs_from_default_at_round_zero():
    """If these agreed, the ablation would measure nothing."""
    a = WarmupScheduler.default(120).resolve(0)
    b = WarmupScheduler.no_warmup(120).resolve(0)
    assert set(a.trainable_groups) != set(b.trainable_groups)


# ---------------------------------------------------------------- config

def test_from_config_parses_spec_8_1_shape():
    conf = {"phases": [
        {"name": "rgb_common", "rounds": [0, 20],
         "trainable": ["rgb_shared", "norm"], "modality_aware": False,
         "loss_weights": {"giou": 2.0, "l1": 5.0, "focal": 1.0}},
        {"name": "adapter_warmup", "rounds": "[20, 50)",
         "trainable": ["rgb_shared", "norm", "aux_specific"],
         "loss_weights": {"giou": 2.0, "l1": 5.0, "focal": 1.0}},
        {"name": "full", "rounds": [50, 120], "trainable": list(TRAINABLE_GROUPS),
         "loss_weights": {"giou": 2.0, "l1": 5.0, "focal": 1.0}, "downweight_missing_aux": False,
         "missing_client_weight": 0.5},
    ]}
    sch = WarmupScheduler.from_config(conf)
    assert sch.num_rounds == 120
    assert sch.resolve(0).loss_weights == {"giou": 2.0, "l1": 5.0, "focal": 1.0}
    assert sch.resolve(49).loss_weights == {"giou": 2.0, "l1": 5.0, "focal": 1.0}
    last = sch.resolve(119)
    assert last.downweight_missing_aux is False
    assert last.missing_client_weight == 0.5


def test_from_config_none_falls_back_to_default():
    sch = WarmupScheduler.from_config(None, num_rounds=60)
    assert sch.num_rounds == 60
    assert sch.resolve(0).phase_name == "rgb_common"


def test_config_typo_in_group_name_fails_loudly():
    conf = {"phases": [{"name": "x", "rounds": [0, 10],
                        "trainable": ["rgb_share"]}]}      # missing the 'd'
    try:
        WarmupScheduler.from_config(conf)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a mistyped group name")


# ---------------------------------------------------------------- interop

def test_plan_is_consumable_by_the_aggregator():
    """Aggregator reads phase.trainable_groups / downweight_missing_aux /
    missing_client_weight via getattr; a RoundPlan must satisfy that."""
    plan = WarmupScheduler.default(120).resolve(119)
    for attr in ("trainable_groups", "downweight_missing_aux", "missing_client_weight"):
        assert hasattr(plan, attr), attr

    from lib.federated import Aggregator
    pg = {"backbone.prompt_blocks.0.conv0_0.weight": RGB_SHARED,
          "backbone.patch_embed_prompt.proj.weight": AUX_SPECIFIC}
    agg = Aggregator(pg)
    phase1 = WarmupScheduler.default(120).resolve(0)
    open_keys = agg._trainable_keys(phase1)
    assert "backbone.prompt_blocks.0.conv0_0.weight" in open_keys
    assert "backbone.patch_embed_prompt.proj.weight" not in open_keys, \
        "phase 1 must not let the aggregator touch aux tensors"


def test_round_plan_serialises_everything_needed_for_logging():
    """spec 16: the log must show phase, trainable groups, loss weights and
    aggregation policy for every round."""
    d = WarmupScheduler.default(120).resolve(25).as_dict()
    for k in ("round", "phase", "trainable", "frozen", "modality_aware", "loss_weights"):
        assert k in d, k
    assert isinstance(d["loss_weights"], dict) and d["loss_weights"]


def test_describe_lists_every_phase():
    text = WarmupScheduler.default(120).describe()
    for p in default_phases(120):
        assert p.name in text


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
