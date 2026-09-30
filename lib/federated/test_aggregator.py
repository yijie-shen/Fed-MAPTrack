#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for param_groups.py and aggregator.py.

Runs standalone (no pytest needed):
    D:\\Conda\\envs\\sdstrack\\python.exe lib/federated/test_aggregator.py

Each test checks one rule from aggregator.py's docstring. Where a rule exists
to prevent a specific silent failure, there is also a NEGATIVE test showing the
wrong behaviour would be detected -- a test that cannot fail proves nothing.
"""

from __future__ import annotations

import os
import sys
import traceback
from collections import OrderedDict

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lib.federated import (  # noqa: E402
    AUX_SPECIFIC,
    FROZEN,
    FUSION,
    NO_AUX,
    NORM,
    RGB_SHARED,
    Aggregator,
    ClientUpdate,
    GlobalState,
    UnknownPromptParam,
    build_param_groups,
    group_of,
    make_initial_state,
    trainable_keys,
)

# ---------------------------------------------------------------- fixtures

# Realistic key names, taken verbatim from a built ViPT model.
KEYS = [
    "backbone.patch_embed.proj.weight",                    # frozen
    "backbone.blocks.0.attn.qkv.weight",                   # frozen
    "box_head.conv1_ctr.0.weight",                         # frozen
    "backbone.patch_embed_prompt.proj.weight",             # aux_specific
    "backbone.patch_embed_prompt.proj.bias",               # aux_specific
    "backbone.prompt_blocks.0.conv0_1.weight",             # aux_specific
    "backbone.prompt_blocks.0.conv0_0.weight",             # rgb_shared
    "backbone.prompt_blocks.0.fovea.smooth",               # rgb_shared
    "backbone.prompt_blocks.0.conv1x1.weight",             # fusion
    "backbone.prompt_norms.0.weight",                      # norm
]

AUX_KEY = "backbone.patch_embed_prompt.proj.weight"
RGB_KEY = "backbone.prompt_blocks.0.conv0_0.weight"


def _pg():
    return build_param_groups(KEYS)


def _state(fill_aux, fill_rgb):
    """A client state where the aux tensor is `fill_aux` and rgb is `fill_rgb`."""
    return {
        AUX_KEY: torch.full((2, 2), float(fill_aux)),
        RGB_KEY: torch.full((2, 2), float(fill_rgb)),
    }


def _initial(aux_types=()):
    model_state = {k: torch.zeros(2, 2) for k in KEYS}
    return make_initial_state(model_state, _pg(), aux_types=aux_types)


# ---------------------------------------------------------------- tests

def test_grouping_assigns_every_key():
    pg = _pg()
    assert pg["backbone.patch_embed.proj.weight"] == FROZEN
    assert pg["box_head.conv1_ctr.0.weight"] == FROZEN
    assert pg["backbone.patch_embed_prompt.proj.weight"] == AUX_SPECIFIC
    assert pg["backbone.prompt_blocks.0.conv0_1.weight"] == AUX_SPECIFIC
    assert pg["backbone.prompt_blocks.0.conv0_0.weight"] == RGB_SHARED
    assert pg["backbone.prompt_blocks.0.fovea.smooth"] == RGB_SHARED
    assert pg["backbone.prompt_blocks.0.conv1x1.weight"] == FUSION
    assert pg["backbone.prompt_norms.0.weight"] == NORM


def test_trainable_set_matches_vipt_freeze_rule():
    """ViPT freezes anything without 'prompt' (base_functions.py:179-181).
    Our grouping must agree exactly, or we would upload the wrong tensors."""
    pg = _pg()
    ours = set(trainable_keys(pg))
    vipt = {k for k in KEYS if "prompt" in k}
    assert ours == vipt, (ours ^ vipt)


def test_ddp_module_prefix_is_handled():
    assert group_of("module.backbone.prompt_blocks.3.conv0_1.weight") == AUX_SPECIFIC


def test_unknown_prompt_param_is_loud_not_silent():
    """A new trainable tensor with no rule must raise, not be dropped."""
    try:
        group_of("backbone.prompt_brandnew.weight")
    except UnknownPromptParam:
        return
    raise AssertionError("expected UnknownPromptParam, got silent fallthrough")


def test_R1_shared_is_sample_weighted_mean():
    """rgb_shared averages over ALL providers, weighted by num_samples."""
    agg = Aggregator(_pg())
    updates = [
        ClientUpdate("t1", num_samples=300, aux_type="thermal", state=_state(1, 1)),
        ClientUpdate("d1", num_samples=100, aux_type="depth", state=_state(9, 5)),
    ]
    new, _ = agg.aggregate(_initial(), updates)
    # (300*1 + 100*5) / 400 = 2.0
    assert torch.allclose(new.shared[RGB_KEY], torch.full((2, 2), 2.0)), new.shared[RGB_KEY]


def test_R2_aux_only_averages_same_modality():
    """The thermal bank must not see the depth client's tensor at all."""
    agg = Aggregator(_pg())
    updates = [
        ClientUpdate("t1", num_samples=100, aux_type="thermal", state=_state(1, 0)),
        ClientUpdate("t2", num_samples=300, aux_type="thermal", state=_state(5, 0)),
        ClientUpdate("d1", num_samples=999, aux_type="depth", state=_state(100, 0)),
    ]
    new, _ = agg.aggregate(_initial(), updates)
    # thermal: (100*1 + 300*5)/400 = 4.0 ; depth's 100 must not leak in
    assert torch.allclose(new.per_modality["thermal"][AUX_KEY], torch.full((2, 2), 4.0))
    assert torch.allclose(new.per_modality["depth"][AUX_KEY], torch.full((2, 2), 100.0))


def test_R2_negative_fedavg_would_contaminate():
    """Same inputs under plain FedAvg give a contaminated value -- proof that
    the previous test is actually measuring modality isolation."""
    agg = Aggregator(_pg(), modality_aware=False)
    updates = [
        ClientUpdate("t1", num_samples=100, aux_type="thermal", state=_state(1, 0)),
        ClientUpdate("t2", num_samples=300, aux_type="thermal", state=_state(5, 0)),
        ClientUpdate("d1", num_samples=999, aux_type="depth", state=_state(100, 0)),
    ]
    new, _ = agg.aggregate(_initial(), updates)
    mixed = new.shared[AUX_KEY][0, 0].item()
    assert not new.per_modality, "modality_aware=False must not create banks"
    assert abs(mixed - 4.0) > 1.0, f"expected contamination, got {mixed}"


def test_R3_rgb_only_client_feeds_shared_but_no_aux_bank():
    agg = Aggregator(_pg())
    updates = [
        ClientUpdate("t1", num_samples=100, aux_type="thermal", state=_state(2, 2)),
        ClientUpdate("r1", num_samples=100, aux_type="none", state=_state(50, 6)),
    ]
    new, _ = agg.aggregate(_initial(), updates)
    # shared sees both: (100*2 + 100*6)/200 = 4.0
    assert torch.allclose(new.shared[RGB_KEY], torch.full((2, 2), 4.0))
    # thermal bank sees only t1
    assert torch.allclose(new.per_modality["thermal"][AUX_KEY], torch.full((2, 2), 2.0))
    assert "none" not in new.per_modality


def test_R4_no_contributor_keeps_previous_not_zero():
    """The rule this project cares most about: missing is skip, not zero."""
    agg = Aggregator(_pg())
    prev = _initial(aux_types=("thermal",))
    prev.per_modality["thermal"][AUX_KEY] = torch.full((2, 2), 7.0)

    # Only a depth client reports this round -- nothing touches the thermal bank.
    updates = [ClientUpdate("d1", num_samples=100, aux_type="depth", state=_state(3, 3))]
    new, stats = agg.aggregate(prev, updates)

    assert torch.allclose(new.per_modality["thermal"][AUX_KEY], torch.full((2, 2), 7.0)), \
        "thermal bank must be untouched, not zeroed or diluted"
    kept = [s for s in stats.per_key if s.bank == "thermal" and s.kept_previous]
    assert kept, "kept_previous must be recorded for auditability"


def test_R4_param_mask_false_is_not_a_zero_contribution():
    """A masked key must be skipped, not averaged in as whatever is in state."""
    agg = Aggregator(_pg())
    prev = _initial(aux_types=("thermal",))
    prev.per_modality["thermal"][AUX_KEY] = torch.full((2, 2), 7.0)

    upd = ClientUpdate(
        "t1", num_samples=100, aux_type="thermal",
        state={AUX_KEY: torch.zeros(2, 2), RGB_KEY: torch.full((2, 2), 1.0)},
        param_mask={AUX_KEY: False, RGB_KEY: True},
    )
    new, stats = agg.aggregate(prev, [upd])
    assert torch.allclose(new.per_modality["thermal"][AUX_KEY], torch.full((2, 2), 7.0)), \
        "masked key must not overwrite the bank with zeros"
    assert torch.allclose(new.shared[RGB_KEY], torch.full((2, 2), 1.0))
    reasons = [r for s in stats.per_key for r in s.skipped if "param_mask" in r]
    assert reasons, "skip reason must be recorded"


def test_R5_reliability_and_missing_client_weight():
    pg = _pg()
    base = Aggregator(pg)
    u = ClientUpdate("x", num_samples=100, aux_type="thermal", state=_state(1, 1),
                     reliability=5.0)
    assert base.weight_of(u) == 100.0            # reliability off by default

    rel = Aggregator(pg, use_reliability_weight=True)
    assert rel.weight_of(u) == 150.0             # clamped 5.0 -> 1.5

    low = ClientUpdate("y", num_samples=100, aux_type="thermal", state=_state(1, 1),
                       reliability=0.0)
    assert rel.weight_of(low) == 20.0            # clamped 0.0 -> 0.2

    dw = Aggregator(pg, missing_client_weight=0.5)
    rgb_only = ClientUpdate("r", num_samples=100, aux_type="none", state=_state(1, 1))
    assert dw.weight_of(rgb_only) == 50.0


def test_R6_degrades_to_plain_fedavg():
    """modality_aware=False must be exactly a sample-weighted mean over all
    clients for every trainable key -- the isolation path for debugging."""
    agg = Aggregator(_pg(), modality_aware=False)
    updates = [
        ClientUpdate("a", num_samples=100, aux_type="thermal", state=_state(2, 2)),
        ClientUpdate("b", num_samples=300, aux_type="depth", state=_state(6, 6)),
    ]
    new, _ = agg.aggregate(_initial(), updates)
    expect = (100 * 2 + 300 * 6) / 400.0         # 5.0
    assert torch.allclose(new.shared[AUX_KEY], torch.full((2, 2), expect))
    assert torch.allclose(new.shared[RGB_KEY], torch.full((2, 2), expect))
    assert new.per_modality == {}


def test_R6_fedavg_result_actually_reaches_the_clients():
    """Regression for a bug that silently broke the whole FedAvg baseline
    (found 2026-09-20, after 20 rounds of a pilot had already been run on it).

    `view_for()` lays per_modality ON TOP of shared. With modality_aware=False
    every key is reduced into shared -- so if the banks are left untouched they
    keep overriding the pooled result, and what the clients actually receive is
    the UNTRAINED initialisation, round after round.

    The old R6 test missed it because its fixture called _initial() with no
    aux_types, so there were no banks to override anything. The production path
    -- server.py -> make_initial_state(..., aux_types=(...)) -- always has them.
    THAT is the lesson: a fixture that does not construct the object the way
    production does can pass forever while production is broken.

    So this test asserts on what leaves the server, not on internals.
    """
    agg = Aggregator(_pg(), modality_aware=False)
    state = _initial(aux_types=("thermal", "depth", "event"))
    assert state.per_modality, "fixture must have banks, or it re-creates the bug"
    before = _clone_view(state, "thermal")

    updates = [
        ClientUpdate("a", num_samples=100, aux_type="thermal", state=_state(2, 2)),
        ClientUpdate("b", num_samples=300, aux_type="depth", state=_state(6, 6)),
    ]
    new, _ = agg.aggregate(state, updates)
    expect = (100 * 2 + 300 * 6) / 400.0          # 5.0

    for aux in ("thermal", "depth", "event"):
        got = new.view_for(aux)[AUX_KEY]
        assert torch.allclose(got, torch.full((2, 2), expect)), \
            f"{aux}: dispatched {got.flatten()[0].item()}, expected {expect}"
    assert not torch.allclose(new.view_for("thermal")[AUX_KEY], before[AUX_KEY]), \
        "the dispatched adapter never moved off its initial value"


def test_R6_every_modality_is_dispatched_the_same_adapter():
    """FedAvg means one adapter for everyone. If two modalities are handed
    different aux tensors, R2/R3 did not actually collapse into R1."""
    agg = Aggregator(_pg(), modality_aware=False)
    state = _initial(aux_types=("thermal", "depth", "event"))
    updates = [
        ClientUpdate("a", num_samples=100, aux_type="thermal", state=_state(2, 2)),
        ClientUpdate("b", num_samples=300, aux_type="depth", state=_state(6, 6)),
    ]
    new, _ = agg.aggregate(state, updates)
    ref = new.view_for("thermal")[AUX_KEY]
    for aux in ("depth", "event"):
        assert torch.equal(new.view_for(aux)[AUX_KEY], ref), aux


def test_fedper_shared_reduction_matches_fedavg_when_private_aux_is_removed():
    """FedPer hands aux_specific back to the server privately.  The ordinary
    aggregator must therefore see exactly the same shared-provider inputs as
    FedAvg, without ever receiving an auxiliary tensor to average."""
    state = _initial(aux_types=("thermal", "depth"))
    updates = [
        ClientUpdate("t", 100, "thermal", _state(2, 1)),
        ClientUpdate("d", 300, "depth", _state(6, 9)),
    ]
    fedavg, _ = Aggregator(_pg(), modality_aware=False).aggregate(state, updates)
    stripped = [ClientUpdate(u.client_id, u.num_samples, u.aux_type,
                             {RGB_KEY: u.state[RGB_KEY]}) for u in updates]
    fedper, stats = Aggregator(_pg(), modality_aware=False).aggregate(state, stripped)
    assert torch.equal(fedper.shared[RGB_KEY], fedavg.shared[RGB_KEY])
    aux_stats = [s for s in stats.per_key if s.group == AUX_SPECIFIC]
    assert aux_stats and all(s.n_contributors == 0 and s.kept_previous
                             for s in aux_stats)


def _alpha_setup():
    """Two thermal clients and one depth client, values picked so every blend
    below is an exact decimal.

        thermal own  = (100*2 + 300*6) / 400 = 5.0
        pooled (all) = (100*2 + 300*6 + 400*1) / 800 = 3.0
    """
    state = _initial(aux_types=("thermal", "depth"))
    updates = [
        ClientUpdate("t1", num_samples=100, aux_type="thermal", state=_state(2, 2)),
        ClientUpdate("t2", num_samples=300, aux_type="thermal", state=_state(6, 6)),
        ClientUpdate("d1", num_samples=400, aux_type="depth", state=_state(1, 1)),
    ]
    return state, updates


def test_alpha_1_is_bit_identical_to_the_old_behaviour():
    """Every number this project has produced so far was alpha = 1. If the new
    code path shifts it even slightly, all of those results silently stop being
    comparable to anything measured from here on."""
    state, updates = _alpha_setup()
    old, _ = Aggregator(_pg(), modality_aware=True).aggregate(state, updates)
    new, _ = Aggregator(_pg(), modality_aware=True,
                        modality_alpha=1.0).aggregate(state, updates)
    for aux in ("thermal", "depth"):
        for k in old.per_modality[aux]:
            assert torch.equal(old.per_modality[aux][k], new.per_modality[aux][k]), \
                (aux, k)
    for k in old.shared:
        assert torch.equal(old.shared[k], new.shared[k]), k


def test_alpha_0_reproduces_the_pooled_fedavg_adapter():
    """The other endpoint: no modality identity left, every bank holds the
    tensor plain FedAvg would have produced."""
    state, updates = _alpha_setup()
    blended, _ = Aggregator(_pg(), modality_aware=True,
                            modality_alpha=0.0).aggregate(state, updates)
    pooled, _ = Aggregator(_pg(), modality_aware=False).aggregate(state, updates)
    want = pooled.shared[AUX_KEY]
    assert torch.allclose(want, torch.full((2, 2), 3.0)), want
    for aux in ("thermal", "depth"):
        assert torch.allclose(blended.per_modality[aux][AUX_KEY], want), aux


def test_alpha_blends_exactly_as_documented():
    """alpha * own + (1 - alpha) * pooled, on the arithmetic, not by eyeball."""
    state, updates = _alpha_setup()
    for a, expect in ((1.0, 5.0), (0.75, 4.5), (0.5, 4.0), (0.25, 3.5), (0.0, 3.0)):
        new, _ = Aggregator(_pg(), modality_aware=True,
                            modality_alpha=a).aggregate(state, updates)
        got = new.per_modality["thermal"][AUX_KEY]
        assert torch.allclose(got, torch.full((2, 2), expect)), \
            "alpha=%s: got %.4f, expected %.4f" % (a, got.flatten()[0], expect)


def test_alpha_leaves_non_modality_tensors_alone():
    """Partial sharing is about aux_specific only. rgb_shared/fusion/norm are
    pooled over everyone at every alpha -- blending them would be a silent
    second change riding along with the first."""
    state, updates = _alpha_setup()
    ref, _ = Aggregator(_pg(), modality_aware=True,
                        modality_alpha=1.0).aggregate(state, updates)
    for a in (0.0, 0.5):
        new, _ = Aggregator(_pg(), modality_aware=True,
                            modality_alpha=a).aggregate(state, updates)
        assert torch.equal(new.shared[RGB_KEY], ref.shared[RGB_KEY]), a


def test_alpha_does_not_touch_a_bank_with_no_contributor():
    """R4 must survive partial sharing. A bank that simply had no data this
    round keeps its previous value -- folding the pooled tensor into it would
    overwrite an adapter that merely sat out, which is the failure R4 exists to
    prevent."""
    state = _initial(aux_types=("thermal", "event"))
    state.per_modality["event"][AUX_KEY] = torch.full((2, 2), 42.0)
    updates = [
        ClientUpdate("t1", num_samples=100, aux_type="thermal", state=_state(2, 2)),
    ]
    new, _ = Aggregator(_pg(), modality_aware=True,
                        modality_alpha=0.0).aggregate(state, updates)
    assert torch.allclose(new.per_modality["event"][AUX_KEY],
                          torch.full((2, 2), 42.0)), \
        new.per_modality["event"][AUX_KEY]


def test_alpha_out_of_range_is_rejected():
    for bad in (-0.1, 1.5):
        try:
            Aggregator(_pg(), modality_alpha=bad)
        except ValueError:
            continue
        raise AssertionError("modality_alpha=%r was accepted" % bad)


def test_alpha_ignores_rgb_only_clients_in_the_pool():
    """An RGB-only client masks its aux tensors False. It must stay out of the
    pooled term too, or partial sharing would quietly feed untrained values
    into every modality bank."""
    state = _initial(aux_types=("thermal",))
    updates = [
        ClientUpdate("t1", num_samples=100, aux_type="thermal", state=_state(2, 2)),
        ClientUpdate("rgb", num_samples=900, aux_type=NO_AUX, state=_state(99, 99),
                     param_mask={AUX_KEY: False, RGB_KEY: True}),
    ]
    new, _ = Aggregator(_pg(), modality_aware=True,
                        modality_alpha=0.0).aggregate(state, updates)
    got = new.per_modality["thermal"][AUX_KEY]
    assert torch.allclose(got, torch.full((2, 2), 2.0)), \
        "the RGB-only client's 99 leaked into the pool: %.4f" % got.flatten()[0]


def test_modality_aware_still_dispatches_separate_adapters():
    """The fix above must not quietly turn modality-aware runs into FedAvg."""
    agg = Aggregator(_pg(), modality_aware=True)
    state = _initial(aux_types=("thermal", "depth"))
    updates = [
        ClientUpdate("a", num_samples=100, aux_type="thermal", state=_state(2, 2)),
        ClientUpdate("b", num_samples=300, aux_type="depth", state=_state(6, 6)),
    ]
    new, _ = agg.aggregate(state, updates)
    th = new.view_for("thermal")[AUX_KEY]
    dp = new.view_for("depth")[AUX_KEY]
    assert torch.allclose(th, torch.full((2, 2), 2.0)), th
    assert torch.allclose(dp, torch.full((2, 2), 6.0)), dp
    assert not torch.equal(th, dp), "banks collapsed -- modality awareness lost"


def _clone_view(state, aux):
    return {k: v.clone() for k, v in state.view_for(aux).items()}


def test_new_modality_starts_from_init_not_zeros():
    """A sensor appearing for the first time must inherit the pretrained
    initialisation; starting it at zero throws away the OSTrack prior."""
    init = _initial()
    init.shared[AUX_KEY] = torch.full((2, 2), 3.0)      # pretend pretrained value
    agg = Aggregator(_pg())
    # event client reports only the RGB key this round
    upd = ClientUpdate("e1", num_samples=100, aux_type="event",
                       state={RGB_KEY: torch.full((2, 2), 1.0)})
    new, _ = agg.aggregate(init, [upd])
    assert torch.allclose(new.per_modality["event"][AUX_KEY], torch.full((2, 2), 3.0))


def test_view_for_dispatches_shared_plus_own_bank():
    st = _initial(aux_types=("thermal", "depth"))
    st.shared[RGB_KEY] = torch.full((2, 2), 1.0)
    st.per_modality["thermal"][AUX_KEY] = torch.full((2, 2), 11.0)
    st.per_modality["depth"][AUX_KEY] = torch.full((2, 2), 22.0)

    v = st.view_for("thermal")
    assert torch.allclose(v[RGB_KEY], torch.full((2, 2), 1.0))
    assert torch.allclose(v[AUX_KEY], torch.full((2, 2), 11.0))
    assert torch.allclose(st.view_for("depth")[AUX_KEY], torch.full((2, 2), 22.0))


def test_flat_dict_roundtrip():
    st = _initial(aux_types=("thermal", "event"))
    st.per_modality["thermal"][AUX_KEY] = torch.full((2, 2), 4.0)
    back = GlobalState.from_flat_dict(st.to_flat_dict())
    assert set(back.shared) == set(st.shared)
    assert set(back.per_modality) == {"thermal", "event"}
    assert torch.allclose(back.per_modality["thermal"][AUX_KEY], torch.full((2, 2), 4.0))


def test_flat_dict_roundtrip_preserves_initial_aux_separately():
    st = _initial(aux_types=("thermal",))
    back = GlobalState.from_flat_dict(st.to_flat_dict())
    assert set(back.initial_aux) == set(st.initial_aux)
    assert "__initial_aux__" not in back.per_modality
    assert all(torch.equal(back.initial_aux[k], st.initial_aux[k])
               for k in st.initial_aux)


def test_input_state_is_not_mutated():
    """aggregate() must be pure; the caller may still need round N-1."""
    prev = _initial(aux_types=("thermal",))
    prev.per_modality["thermal"][AUX_KEY] = torch.full((2, 2), 7.0)
    snapshot = prev.per_modality["thermal"][AUX_KEY].clone()

    agg = Aggregator(_pg())
    agg.aggregate(prev, [ClientUpdate("t1", num_samples=10, aux_type="thermal",
                                      state=_state(99, 99))])
    assert torch.allclose(prev.per_modality["thermal"][AUX_KEY], snapshot)


def test_phase_restricts_open_groups():
    """spec Phase 1 trains only the RGB path; aux banks must stay put."""
    class Phase:
        trainable_groups = (RGB_SHARED,)

    prev = _initial(aux_types=("thermal",))
    prev.per_modality["thermal"][AUX_KEY] = torch.full((2, 2), 7.0)
    agg = Aggregator(_pg())
    new, _ = agg.aggregate(
        prev,
        [ClientUpdate("t1", num_samples=100, aux_type="thermal", state=_state(99, 3))],
        phase=Phase(),
    )
    assert torch.allclose(new.shared[RGB_KEY], torch.full((2, 2), 3.0))
    assert torch.allclose(new.per_modality["thermal"][AUX_KEY], torch.full((2, 2), 7.0)), \
        "Phase 1 must not touch the aux bank"


def test_stats_record_contributors_and_skips():
    agg = Aggregator(_pg())
    updates = [
        ClientUpdate("t1", num_samples=100, aux_type="thermal", state=_state(1, 1)),
        ClientUpdate("d1", num_samples=100, aux_type="depth", state=_state(2, 2)),
    ]
    _, stats = agg.aggregate(_initial(), updates, round_idx=7)
    assert stats.round_idx == 7
    th = [s for s in stats.per_key if s.bank == "thermal" and s.key == AUX_KEY][0]
    assert th.contributors == ["t1"]
    assert any("aux_type=depth" in r for r in th.skipped), th.skipped
    summary = stats.group_summary()
    assert RGB_SHARED in summary and summary[RGB_SHARED]["tensors"] >= 1


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
