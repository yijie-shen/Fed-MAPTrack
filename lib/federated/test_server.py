#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for server.py.

    D:\\Conda\\envs\\sdstrack\\python.exe lib/federated/test_server.py

No GPU, no real model, no data -- runs in well under a second.

Why a stub is the RIGHT thing here, not a shortcut:
`group_of()` buckets any key without "prompt" as FROZEN, so a stub state_dict
using ViPT's real key names produces the real grouping. And a stub client that
returns a KNOWN constant per parameter lets the aggregation result be computed
by hand -- which is the only way to test that the server's weighted mean is
arithmetically right, rather than merely that it runs. `smoke_client.py` and the
server smoke run cover the real model and real data.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback
from collections import OrderedDict

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from lib.federated.aggregator import (ClientUpdate, GlobalState, NO_AUX,
                                      Aggregator)  # noqa: E402
from lib.federated.param_groups import (  # noqa: E402
    AUX_SPECIFIC, FROZEN, FUSION, NORM, RGB_SHARED, build_param_groups,
)
from lib.federated.server import VARIANTS, FederatedServer, _jsonable  # noqa: E402
from lib.federated.warmup import Phase

# Real ViPT key names, one per group, plus two frozen ones.
KEYS = OrderedDict([
    ("backbone.blocks.0.attn.qkv.weight", FROZEN),
    ("box_head.conv1.0.weight", FROZEN),
    ("backbone.patch_embed_prompt.proj.weight", AUX_SPECIFIC),
    ("backbone.prompt_blocks.0.conv0_1.weight", AUX_SPECIFIC),
    ("backbone.prompt_blocks.0.conv0_0.weight", RGB_SHARED),
    ("backbone.prompt_blocks.0.fovea.weight", RGB_SHARED),
    ("backbone.prompt_blocks.0.conv1x1.weight", FUSION),
    ("backbone.prompt_norms.0.weight", NORM),
])


class StubNet:
    """Only `state_dict()` is ever called on the net by the server."""

    def __init__(self):
        self._sd = OrderedDict((k, torch.full((2,), -7.0)) for k in KEYS)

    def state_dict(self):
        return OrderedDict((k, v.clone()) for k, v in self._sd.items())


class StubClient:
    """Uploads `value` for every open key, so aggregation is hand-checkable."""

    def __init__(self, client_id, aux_type, num_samples, value, dataset_name="lasher"):
        self.client_id = client_id
        self.aux_type = aux_type
        self.num_samples = num_samples
        self.value = float(value)
        self.dataset_name = dataset_name
        self.calls = []

    def local_train(self, net, dispatched, plan, round_idx, local_steps,
                    device="cpu"):
        self.calls.append((round_idx, plan.phase_name, local_steps,
                           len(dispatched)))
        groups = build_param_groups(net.state_dict().keys())
        open_groups = set(plan.trainable_groups)
        state = OrderedDict(
            (k, torch.full((2,), self.value))
            for k in dispatched if groups[k] in open_groups)
        has_aux = self.aux_type != NO_AUX
        mask = OrderedDict(
            (k, has_aux or groups[k] != AUX_SPECIFIC) for k in state)
        update = ClientUpdate(client_id=self.client_id,
                              num_samples=self.num_samples,
                              aux_type=self.aux_type, state=state,
                              param_mask=mask)
        stats = OrderedDict(client_id=self.client_id, aux_type=self.aux_type,
                            round=round_idx)
        stats["mean_Loss/total"] = 1.0
        stats["mean_IoU"] = 0.5
        stats["aux_present_frac"] = 1.0 if has_aux else 0.0
        return update, stats


def _server(variant="modality", clients=None, **kw):
    """A server on stubs. num_samples are chosen so the weighted means below
    come out as exact decimals."""
    clients = clients if clients is not None else [
        StubClient("rgbt_a", "thermal", 100, 1.0),
        StubClient("rgbt_b", "thermal", 300, 5.0),
        StubClient("rgb_only", NO_AUX, 200, 9.0),
        StubClient("rgbd_a", "depth", 400, 2.0),
    ]
    kw.setdefault("num_rounds", 4)
    kw.setdefault("local_steps", 3)
    kw.setdefault("verbose", False)
    kw.setdefault("device", "cpu")
    kw.setdefault("out_dir", tempfile.mkdtemp(prefix="fedtest_"))
    return FederatedServer(StubNet(), clients, cfg=None, settings=None,
                           variant=variant, **kw), clients


# --------------------------------------------------------------- the variants

def test_variant_table_matches_the_documented_baselines():
    assert VARIANTS["local"] == (False, False, False)
    assert VARIANTS["fedavg"] == (False, False, True)
    assert VARIANTS["fedavg_warm"] == (False, True, True)
    assert VARIANTS["modality"] == (True, False, True)
    assert VARIANTS["ours"] == (True, True, True)


def test_unknown_variant_is_rejected():
    try:
        _server(variant="nope")
    except ValueError:
        return
    raise AssertionError("an unknown variant was accepted")


def test_bucket_groups_are_rejected_outside_modality_variant():
    try:
        _server("fedavg", bucket_groups=("aux_specific", "fusion"))
    except ValueError:
        return
    raise AssertionError("fedavg accepted modality-only bucket groups")


def test_bucketed_non_aux_groups_use_dataset_bucket_for_rgb_only_clients():
    clients = [
        StubClient("rgbt_sensor", "thermal", 100, 1),
        StubClient("rgbt_rgb", NO_AUX, 300, 5, "lasher"),
        StubClient("rgbd_sensor", "depth", 100, 9, "depthtrack"),
        StubClient("rgbd_rgb", NO_AUX, 100, 13, "depthtrack"),
    ]
    s, _ = _server("modality", clients=clients,
                   bucket_groups=("aux_specific", "fusion", "norm"))
    s.run_round(0)
    fusion = "backbone.prompt_blocks.0.conv1x1.weight"
    norm = "backbone.prompt_norms.0.weight"
    # thermal (1*100 + 5*300)/400 = 4; depth (9 + 13)/2 = 11
    for key in (fusion, norm):
        assert torch.equal(s.state.per_modality["thermal"][key], torch.full((2,), 4.0))
        assert torch.equal(s.state.per_modality["depth"][key], torch.full((2,), 11.0))
    rgb = "backbone.prompt_blocks.0.conv0_0.weight"
    # (1*100 + 5*300 + 9*100 + 13*100) / 600
    assert torch.allclose(s.state.shared[rgb], torch.full((2,), 3800.0 / 600.0))


def test_default_bucket_groups_matches_explicit_legacy_default():
    clients = [StubClient("t", "thermal", 100, 1),
               StubClient("d", "depth", 300, 5)]
    a, _ = _server("modality", clients=clients)
    b, _ = _server("modality", clients=clients, bucket_groups=("aux_specific",))
    a.run_round(0)
    b.run_round(0)
    for aux in a.state.per_modality:
        for key in a.state.per_modality[aux]:
            assert torch.equal(a.state.per_modality[aux][key], b.state.per_modality[aux][key])
    for key in a.state.shared:
        assert torch.equal(a.state.shared[key], b.state.shared[key])


def test_bucketed_save_load_then_continue_is_bit_identical():
    clients = [StubClient("t", "thermal", 100, 1),
               StubClient("r", NO_AUX, 300, 5, "lasher"),
               StubClient("d", "depth", 200, 9, "depthtrack")]
    groups = ("aux_specific", "fusion", "norm")
    uninterrupted, _ = _server("modality", clients=clients, bucket_groups=groups)
    uninterrupted.run_round(0)
    uninterrupted.run_round(1)
    paused, _ = _server("modality", clients=clients, bucket_groups=groups)
    paused.run_round(0)
    checkpoint = paused.save_state(1)
    resumed, _ = _server("modality", clients=clients, bucket_groups=groups)
    assert resumed.load_state(checkpoint) == 1
    resumed.run_round(1)
    assert uninterrupted.state.to_flat_dict().keys() == resumed.state.to_flat_dict().keys()
    for key, value in uninterrupted.state.to_flat_dict().items():
        assert torch.equal(value, resumed.state.to_flat_dict()[key]), key


def test_only_ours_uses_warmup():
    for variant, expect in (("ours", True), ("modality", False),
                            ("fedavg", False), ("local", False)):
        s, _ = _server(variant, num_rounds=120)
        phases = {s.scheduler.resolve(r).phase_name for r in range(120)}
        many = len(phases) > 1
        assert many == expect, "%s: phases=%s" % (variant, phases)


def test_modality_aware_reaches_the_aggregator():
    for variant, expect in (("ours", True), ("modality", True), ("fedavg", False)):
        s, _ = _server(variant)
        assert s.aggregator.modality_aware is expect, variant


def test_the_phase_can_turn_modality_awareness_off_for_a_round():
    """Regression for a real latent bug (2026-09-18).

    Aggregator reads only its constructor flag; RoundPlan.modality_aware was
    logged and consumed by nobody, so the log could claim a strategy the
    aggregator was not following. `ours` round 0 is rgb_common, whose plan says
    modality_aware=False -- the aggregator must agree by the time it runs.
    """
    s, _ = _server("ours", num_rounds=120)
    assert s.scheduler.resolve(0).modality_aware is False, "phase table changed"
    s.run_round(0)
    assert s.aggregator.modality_aware is False, \
        "the phase said modality_aware=False and the aggregator ignored it"
    r = json.loads(open(s.log_path, encoding="utf-8").read().splitlines()[0])
    assert r["modality_aware"] is False, r["modality_aware"]


def test_fedavg_warm_logs_what_it_did_not_what_the_phase_wanted():
    """`fedavg_warm` is the cell where the phase and the variant disagree in
    every round after rgb_common: the warm-up table says modality_aware=True,
    the variant says False, and False wins. The log has to say False.

    Regression for the 2026-09-21 catch -- same shape as the 2026-09-18 bug
    (a value written to the log sitting next to a different value that actually
    drives behaviour), found when this variant was added.
    """
    s, _ = _server("fedavg_warm", num_rounds=120)
    plan = s.scheduler.resolve(30)                 # adapter_warmup
    assert plan.modality_aware is True, "phase table changed"
    s.scheduler.resolve = lambda _r, _p=plan: _p
    s.run_round(30)
    assert s.aggregator.modality_aware is False
    r = json.loads(open(s.log_path, encoding="utf-8").read().splitlines()[0])
    assert r["modality_aware"] is False, r["modality_aware"]
    assert r["phase_modality_aware"] is True, "the phase's wish should still be recorded"


def test_local_logs_modality_aware_false():
    """`local` aggregates nothing, so no aggregation strategy was applied."""
    s, _ = _server("local")
    s.run_round(0)
    r = json.loads(open(s.log_path, encoding="utf-8").read().splitlines()[0])
    assert r["modality_aware"] is False, r["modality_aware"]


def test_modality_alpha_is_recorded_in_every_round():
    for a in (1.0, 0.5):
        s, _ = _server("modality", modality_alpha=a)
        s.run_round(0)
        r = json.loads(open(s.log_path, encoding="utf-8").read().splitlines()[0])
        assert r["modality_alpha"] == a, (a, r["modality_alpha"])
        assert s.aggregator.modality_alpha == a


def test_the_variant_is_the_ceiling_not_the_phase():
    """fedavg must stay pooled even if a phase asks for modality awareness."""
    s, _ = _server("fedavg", num_rounds=120)
    plan = s.scheduler.resolve(0)
    object.__setattr__(plan, "modality_aware", True)   # a phase asking for it
    s.scheduler.resolve = lambda _r, _p=plan: _p
    s.run_round(0)
    assert s.aggregator.modality_aware is False, \
        "fedavg was talked into modality-aware aggregation by a phase"


def test_only_local_skips_aggregation():
    for variant in VARIANTS:
        s, _ = _server(variant)
        assert s.do_aggregate == (variant != "local"), variant
        assert (s.local_states is not None) == (variant == "local"), variant


# ------------------------------------------------------------------ selection

def test_select_is_deterministic():
    s, _ = _server(clients_per_round=2)
    a = s.select(3, s.scheduler.resolve(3))
    b = s.select(3, s.scheduler.resolve(3))
    assert a == b, (a, b)
    assert len(a) == 2


def test_select_changes_with_the_round():
    s, _ = _server(clients_per_round=2)
    picks = {tuple(s.select(r, s.scheduler.resolve(r))) for r in range(12)}
    assert len(picks) > 1, "same clients every round -- the round is not seeded in"


def test_select_accepts_a_fraction():
    s, _ = _server(clients_per_round=0.5)
    assert len(s.select(0, s.scheduler.resolve(0))) == 2


def test_select_returns_everyone_sorted_when_k_is_all():
    s, clients = _server(clients_per_round=len(_server()[1]))
    got = s.select(0, s.scheduler.resolve(0))
    assert got == list(s.clients), got


def test_select_never_exceeds_the_client_count():
    s, _ = _server(clients_per_round=99)
    assert len(s.select(0, s.scheduler.resolve(0))) == len(s.clients)


# ---------------------------------------------------------------- aggregation

def test_aggregation_equals_a_hand_computed_weighted_mean():
    """The arithmetic, not just that it runs.

    shared keys  -> every client contributes:
        (100*1 + 300*5 + 200*9 + 400*2) / 1000 = 4.2
    thermal bank -> only the two thermal clients:
        (100*1 + 300*5) / 400 = 4.0
    depth bank   -> only the depth client: 2.0
    The RGB-only client's 9.0 must NOT appear in any bank.
    """
    s, _ = _server("modality")
    s.run_round(0)

    for key, group in KEYS.items():
        if group in (FROZEN, AUX_SPECIFIC):
            continue
        got = s.state.shared[key][0].item()
        assert abs(got - 4.2) < 1e-6, "%s: %.6f != 4.2" % (key, got)

    aux_keys = [k for k, g in KEYS.items() if g == AUX_SPECIFIC]
    for key in aux_keys:
        t = s.state.per_modality["thermal"][key][0].item()
        d = s.state.per_modality["depth"][key][0].item()
        assert abs(t - 4.0) < 1e-6, "thermal %s: %.6f != 4.0" % (key, t)
        assert abs(d - 2.0) < 1e-6, "depth %s: %.6f != 2.0" % (key, d)
        assert abs(t - 9.0) > 1e-6 and abs(d - 9.0) > 1e-6, \
            "the RGB-only client's value leaked into a modality bank"


def test_fedavg_pools_aux_specific_across_modalities():
    """The degradation path: without modality awareness the aux tensors are one
    average over everyone who uploaded them -- thermal + depth together."""
    s, _ = _server("fedavg")
    s.run_round(0)
    key = next(k for k, g in KEYS.items() if g == AUX_SPECIFIC)
    # thermal(100*1 + 300*5) + depth(400*2) over 800; the RGB-only client is
    # masked False on aux keys, so it must still be excluded.
    want = (100 * 1.0 + 300 * 5.0 + 400 * 2.0) / 800.0
    got = s.state.shared[key][0].item()
    assert abs(got - want) < 1e-6, "%.6f != %.6f" % (got, want)


def test_bank_with_no_contributor_keeps_its_previous_value():
    """R4: 'event' has no client here, so its bank must survive untouched --
    not be zeroed, and not be filled from another modality."""
    s, clients = _server("modality", clients=[
        StubClient("rgbt_a", "thermal", 100, 1.0),
        StubClient("rgbe_a", "event", 100, 3.0),
    ])
    before = {k: v.clone() for k, v in s.state.per_modality["event"].items()}
    s.clients.pop("rgbe_a")          # event client drops out of this round
    s.run_round(0)
    for k, v in before.items():
        assert torch.equal(s.state.per_modality["event"][k], v), \
            "event bank changed with no contributor: %s" % k


def test_rgb_only_client_alone_writes_no_aux_specific():
    s, _ = _server("modality", clients=[StubClient("rgb", NO_AUX, 100, 9.0)])
    before = {aux: {k: v.clone() for k, v in bank.items()}
              for aux, bank in s.state.per_modality.items()}
    s.run_round(0)
    for aux, bank in before.items():
        for k, v in bank.items():
            assert torch.equal(s.state.per_modality[aux][k], v), \
                "RGB-only client wrote into %s/%s" % (aux, k)


# ----------------------------------------------------------- local (no merge)

def test_local_variant_keeps_one_state_per_client():
    s, _ = _server("local")
    assert set(s.local_states) == set(s.clients)


def test_local_variant_does_not_share_between_clients():
    s, _ = _server("local", clients=[
        StubClient("a", "thermal", 100, 1.0),
        StubClient("b", "thermal", 100, 5.0),
    ])
    s.run_round(0)
    key = next(k for k, g in KEYS.items() if g == RGB_SHARED)
    a = s.local_states["a"].shared[key][0].item()
    b = s.local_states["b"].shared[key][0].item()
    assert abs(a - 1.0) < 1e-6 and abs(b - 5.0) < 1e-6, (a, b)


def test_local_variant_leaves_the_global_state_alone():
    s, _ = _server("local")
    before = {k: v.clone() for k, v in s.state.shared.items()}
    s.run_round(0)
    for k, v in before.items():
        assert torch.equal(s.state.shared[k], v), k


def test_local_save_state_preserves_every_private_client_state():
    """Regression for the local-export loss (2026-09-20).

    Checking only `state` would be vacuous: local intentionally leaves that
    initial global object unchanged.  The persisted blob must retain each
    client's actual post-training bank instead.
    """
    s, _ = _server("local")
    s.run_round(0)
    blob = torch.load(s.save_state(1), map_location="cpu", weights_only=False)
    assert set(blob["local_states"]) == set(s.clients)
    key = next(k for k, g in KEYS.items() if g == RGB_SHARED)
    for cid, state in s.local_states.items():
        restored = GlobalState.from_flat_dict(blob["local_states"][cid])
        assert torch.equal(restored.shared[key], state.shared[key]), cid


def test_local_export_uses_the_trained_private_client_view():
    """The checkpoint must be the client model, not untouched self.state."""
    s, _ = _server("local", clients=[
        StubClient("thermal_one", "thermal", 100, 1.0),
        StubClient("thermal_two", "thermal", 100, 5.0),
    ])
    s.run_round(0)
    path = s.export_checkpoint(1, client_id="thermal_two")[0]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for key, group in KEYS.items():
        if group != FROZEN:
            assert torch.equal(ck["net"][key], torch.full((2,), 5.0)), key
    assert ck["fed"]["client_id"] == "thermal_two"
    assert ck["fed"]["aux_type"] == "thermal"


# ------------------------------------------------------------------- logging

def test_round_log_records_what_spec16_requires():
    s, _ = _server("ours")
    s.run_round(0)
    lines = open(s.log_path, encoding="utf-8").read().splitlines()
    assert len(lines) == 1, lines
    r = json.loads(lines[0])
    for field in ("round", "phase", "trainable", "modality_aware",
                  "loss_weights", "variant", "seed", "selected",
                  "local_steps", "clients", "aggregation"):
        assert field in r, "missing %r" % field
    assert isinstance(r["trainable"], list) and r["trainable"]
    assert r["clients"] and "mean_IoU" in r["clients"][0]


def test_log_appends_one_line_per_round():
    s, _ = _server("modality", num_rounds=3)
    s.run(save_every=0)
    lines = open(s.log_path, encoding="utf-8").read().splitlines()
    assert len(lines) == 3, len(lines)
    assert [json.loads(x)["round"] for x in lines] == [0, 1, 2]


def test_clients_receive_the_round_and_step_budget():
    s, clients = _server("modality", num_rounds=2, local_steps=7)
    s.run(save_every=0)
    for c in s.clients.values():
        rounds = [call[0] for call in c.calls]
        steps = {call[2] for call in c.calls}
        assert rounds == [0, 1], (c.client_id, rounds)
        assert steps == {7}, (c.client_id, steps)


# ---------------------------------------------------------------- checkpoints

def test_export_writes_one_file_per_modality():
    """Its own client list: the shared fixture has no event client, because its
    num_samples are chosen to make the weighted means above exact."""
    s, _ = _server("modality", clients=[
        StubClient("rgbt_a", "thermal", 100, 1.0),
        StubClient("rgbd_a", "depth", 100, 2.0),
        StubClient("rgbe_a", "event", 100, 3.0),
    ])
    s.run_round(0)
    paths = s.export_checkpoint(1)
    assert len(paths) == len(s.aux_types) == 3, paths
    names = sorted(os.path.basename(p) for p in paths)
    assert names == ["ViPTrack_depth_ep0001.pth.tar",
                     "ViPTrack_event_ep0001.pth.tar",
                     "ViPTrack_thermal_ep0001.pth.tar"], names


def test_default_fixture_has_the_modalities_the_arithmetic_assumes():
    """Guards the fixture itself: if someone adds a client here, the hand-
    computed means above silently start testing different numbers."""
    s, _ = _server("modality")
    assert s.aux_types == ("depth", "thermal"), s.aux_types
    assert sum(c.num_samples for c in s.clients.values()) == 1000


def test_export_uses_the_official_filename_for_a_single_modality():
    """lib/test/parameter/vipt.py looks for exactly ViPTrack_ep%04d.pth.tar."""
    s, _ = _server("modality")
    path = s.export_checkpoint(12, aux_type="thermal")[0]
    assert os.path.basename(path) == "ViPTrack_ep0012.pth.tar", path


def test_exported_state_dict_has_exactly_the_model_keys():
    """A missing or extra key makes the official load_state_dict(strict=True)
    fail, which is how a federated run would become unevaluable."""
    s, _ = _server("modality")
    s.run_round(0)
    want = set(StubNet().state_dict())
    for path in s.export_checkpoint(1):
        got = set(torch.load(path, map_location="cpu", weights_only=False)["net"])
        assert got == want, (sorted(got - want)[:3], sorted(want - got)[:3])


def test_export_leaves_frozen_tensors_untouched():
    s, _ = _server("modality")
    s.run_round(0)
    frozen = [k for k, g in KEYS.items() if g == FROZEN]
    ck = torch.load(s.export_checkpoint(1)[0], map_location="cpu",
                    weights_only=False)["net"]
    for k in frozen:
        assert torch.equal(ck[k], s.frozen_reference[k]), \
            "%s moved; the backbone must never travel" % k


def test_exported_modalities_differ_in_aux_specific_only():
    """The point of per-modality banks: same shared weights, different adapters."""
    s, _ = _server("modality")
    s.run_round(0)
    paths = {os.path.basename(p).split("_")[1]: p
             for p in s.export_checkpoint(1)}
    th = torch.load(paths["thermal"], map_location="cpu", weights_only=False)["net"]
    dp = torch.load(paths["depth"], map_location="cpu", weights_only=False)["net"]
    for k, g in KEYS.items():
        same = torch.equal(th[k], dp[k])
        if g == AUX_SPECIFIC:
            assert not same, "%s identical across modalities" % k
        else:
            assert same, "%s differs across modalities but is %s" % (k, g)


def test_export_carries_provenance():
    s, _ = _server("ours")
    ck = torch.load(s.export_checkpoint(5, aux_type="depth")[0],
                    map_location="cpu", weights_only=False)
    assert ck["epoch"] == 5
    assert ck["fed"] == {"variant": "ours", "aux_type": "depth",
                         "round": 5, "seed": 42}, ck["fed"]


def test_save_state_round_trips_the_banks():
    s, _ = _server("modality")
    s.run_round(0)
    blob = torch.load(s.save_state(1), map_location="cpu", weights_only=False)
    back = GlobalState.from_flat_dict(blob["state"])
    assert set(back.per_modality) == set(s.state.per_modality)
    for aux, bank in s.state.per_modality.items():
        for k, v in bank.items():
            assert torch.equal(back.per_modality[aux][k], v), (aux, k)
    for k, v in s.state.shared.items():
        assert torch.equal(back.shared[k], v), k


# ----------------------------------------------------------------- utilities

def test_jsonable_survives_tensors_sets_and_nesting():
    out = _jsonable({"a": {1: torch.zeros(2)}, "b": {"x", "y"}, "c": [1, 2.5, None]})
    json.dumps(out)                       # must not raise
    assert out["c"] == [1, 2.5, None]
    assert sorted(out["b"]) == ["x", "y"]


def test_rgb_only_weight_half_matches_legacy_phase_weight():
    clients = [StubClient("rgb", NO_AUX, 200, 9.0),
               StubClient("thermal", "thermal", 100, 1.0)]
    s, _ = _server("fedavg", clients=clients, rgb_only_weight=0.5)
    plan = s.scheduler.resolve(0)
    updates = [c.local_train(s.net, s.state.view_for(c.aux_type), plan, 0, 3)[0]
               for c in clients]
    legacy = Phase("old", 0, 1, trainable_groups=plan.trainable_groups,
                   downweight_missing_aux=True, missing_client_weight=0.5)
    a, _ = Aggregator(s.param_groups, modality_aware=False).aggregate(
        s.state, updates, phase=legacy)
    b, _ = Aggregator(s.param_groups, modality_aware=False,
                      missing_client_weight=0.5).aggregate(s.state, updates,
                                                           phase=plan)
    for k in a.shared:
        assert torch.equal(a.shared[k], b.shared[k]), k


def test_log_loss_weights_are_the_objective_weights():
    from types import SimpleNamespace
    from lib.federated.client import _build_objective
    c = SimpleNamespace(TRAIN=SimpleNamespace(GIOU_WEIGHT=3.0, L1_WEIGHT=7.0))
    _, weights = _build_objective(c)
    s, _ = _server("fedavg")
    s.cfg = c
    rec = s.run_round(0)
    assert rec["loss_weights"] == {"giou": weights["giou"],
                                    "l1": weights["l1"], "focal": weights["focal"]}


def test_alpha_zero_system_endpoint_with_missing_modality_and_export():
    class MissingThermal(StubClient):
        def local_train(self, net, dispatched, plan, round_idx, local_steps,
                        device="cpu"):
            update, stats = super().local_train(net, dispatched, plan, round_idx,
                                                local_steps, device)
            if round_idx == 1 and self.aux_type == "thermal":
                for k in list(update.param_mask):
                    if "prompt" in k and ("patch_embed_prompt" in k or "conv0_1" in k):
                        update.param_mask[k] = False
            return update, stats

    def clients():
        return [MissingThermal("rgb", NO_AUX, 100, 9.0),
                MissingThermal("thermal", "thermal", 100, 1.0),
                MissingThermal("depth", "depth", 100, 3.0)]
    mod, _ = _server("modality", clients=clients(), modality_alpha=0.0,
                      num_rounds=2)
    fed, _ = _server("fedavg", clients=clients(), num_rounds=2)
    mod.run_round(0); fed.run_round(0)
    for k in mod.state.view_for("thermal"):
        assert torch.equal(mod.state.view_for("thermal")[k],
                           fed.state.view_for("thermal")[k]), k
    mod.run_round(1); fed.run_round(1)
    thermal_key = next(iter(mod.state.per_modality["thermal"]))
    assert not torch.equal(mod.state.per_modality["thermal"][thermal_key],
                           fed.state.shared[thermal_key])
    paths = mod.export_checkpoint(2)
    assert paths and all(os.path.isfile(p) for p in paths)


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    tmp_root = tempfile.gettempdir()
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            print("  [OK]   %s" % name)
            passed += 1
        except Exception:
            print("  [FAIL] %s" % name)
            traceback.print_exc()
            failed.append(name)
    for d in os.listdir(tmp_root):
        if d.startswith("fedtest_"):
            shutil.rmtree(os.path.join(tmp_root, d), ignore_errors=True)
    print("\n%d/%d passed" % (passed, len(tests)))
    if failed:
        print("failed: " + ", ".join(failed))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
