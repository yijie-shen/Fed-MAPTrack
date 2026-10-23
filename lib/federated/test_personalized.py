#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the FedMSplit-P and FED-PRIME baselines (added 2026-10-16).

    D:\\Conda\\envs\\sdstrack\\python.exe lib/federated/test_personalized.py

Four layers, cheapest first:
  1. FedMSplit-P aggregation arithmetic (hand-computed)
  2. FED-PRIME inter-prompt alignment
  3. the FED-PRIME token pools inside the REAL ViPT backbone (CPU, tiny batch)
  4. server wiring on stub clients: dispatch, privacy of the intra pool,
     degeneration to FedAvg on the shared groups, export, save/restore
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
import traceback
from collections import OrderedDict

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from lib.federated.aggregator import ClientUpdate, NO_AUX  # noqa: E402
from lib.federated.param_groups import (  # noqa: E402
    AUX_SPECIFIC, FUSION, NORM, PRIME, RGB_SHARED, TRAINABLE_GROUPS,
    build_param_groups)
from lib.federated.personalized import (  # noqa: E402
    align_global_prompts, fedmsplit_p_aggregate)
from lib.federated.server import FederatedServer  # noqa: E402

FAST = os.environ.get("PERSONALIZED_FAST") == "1"


def _close(a, b, tol=1e-6):
    return float((a - b).abs().max()) <= tol


# ======================================================== 1. FedMSplit-P maths

def _vec(*xs):
    return torch.tensor(xs, dtype=torch.float32)


def test_fedmsplit_two_clients_one_block_matches_the_formula():
    states = OrderedDict(a={"k": _vec(1., 0.)}, b={"k": _vec(0., 2.)})
    blocks = OrderedDict(blk=["k"])
    prov = {"a": {"blk": True}, "b": {"blk": True}}
    personal, gavg, _ = fedmsplit_p_aggregate(states, blocks, prov,
                                              {"a": 1.0, "b": 3.0})
    # Gram = [[1, 0], [0, 4]] / sqrt(2)
    s = math.sqrt(2.0)
    A = torch.softmax(torch.tensor([[1 / s, 0.0], [0.0, 4 / s]]), dim=1)
    # single block => c_kl = A[k, l] and w_kl = A[k, l]^2
    for i, k in enumerate("ab"):
        w = A[i] ** 2
        w = w / w.sum()
        expect = w[0] * states["a"]["k"] + w[1] * states["b"]["k"]
        assert _close(personal[k]["k"], expect, 1e-6), (k, personal[k]["k"], expect)
    # global average is data-volume weighted
    assert _close(gavg["k"], (1 * _vec(1., 0.) + 3 * _vec(0., 2.)) / 4)


def test_fedmsplit_identical_clients_stay_identical():
    x = torch.randn(7)
    states = OrderedDict((c, {"k": x.clone()}) for c in "abc")
    prov = {c: {"blk": True} for c in "abc"}
    personal, gavg, _ = fedmsplit_p_aggregate(
        states, OrderedDict(blk=["k"]), prov, {"a": 1., "b": 5., "c": 2.})
    for c in "abc":
        assert _close(personal[c]["k"], x)
    assert _close(gavg["k"], x)


def test_fedmsplit_weights_favour_the_similar_client():
    # a and b nearly equal, c far away: a must lean toward b, not c
    states = OrderedDict(a={"k": _vec(5., 5.)}, b={"k": _vec(5., 5.1)},
                         c={"k": _vec(-5., -5.)})
    prov = {c: {"blk": True} for c in "abc"}
    personal, _, info = fedmsplit_p_aggregate(
        states, OrderedDict(blk=["k"]), prov, {c: 1.0 for c in "abc"})
    assert personal["a"]["k"][0] > 4.9, personal["a"]["k"]
    assert info["blk"]["providers"] == 3


def test_fedmsplit_client_without_a_block_is_left_out_of_it():
    states = OrderedDict(
        a={"x": _vec(1., 1.), "y": _vec(2., 2.)},
        b={"x": _vec(3., 3.), "y": _vec(9., 9.)},     # b has no valid y
        c={"x": _vec(5., 5.), "y": _vec(4., 4.)})
    blocks = OrderedDict(bx=["x"], by=["y"])
    prov = {"a": {"bx": True, "by": True}, "b": {"bx": True, "by": False},
            "c": {"bx": True, "by": True}}
    personal, gavg, info = fedmsplit_p_aggregate(
        states, blocks, prov, {"a": 1., "b": 1., "c": 1.})
    assert "y" not in personal["b"], "b must not receive a block it does not provide"
    assert info["by"]["providers"] == 2 and info["bx"]["providers"] == 3
    # b's y value (9) must not leak into anyone's y or the global y
    assert float(personal["a"]["y"].max()) <= 4.0 + 1e-6
    assert _close(gavg["y"], _vec(3., 3.))


def test_fedmsplit_rows_are_convex_combinations():
    torch.manual_seed(0)
    states = OrderedDict((c, {"k": torch.randn(50)}) for c in "abcd")
    prov = {c: {"blk": True} for c in "abcd"}
    personal, _, info = fedmsplit_p_aggregate(
        states, OrderedDict(blk=["k"]), prov, {c: 1.0 for c in "abcd"})
    lo = torch.stack([s["k"] for s in states.values()]).min(0)[0]
    hi = torch.stack([s["k"] for s in states.values()]).max(0)[0]
    for c in "abcd":
        assert bool((personal[c]["k"] >= lo - 1e-6).all()
                    and (personal[c]["k"] <= hi + 1e-6).all())
    assert 0.0 < info["blk"]["mean_self_weight"] <= 1.0


# ======================================================= 2. FED-PRIME alignment

def _prompts(tau=6, dim=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(tau, dim, generator=g)


def test_align_recovers_a_permutation_and_averages():
    base = _prompts()
    rows = OrderedDict()
    perms = {"a": [2, 0, 5, 1, 4, 3], "b": [5, 4, 3, 2, 1, 0], "c": [0, 1, 2, 3, 4, 5]}
    g = torch.Generator().manual_seed(1)
    for c, p in perms.items():
        rows[c] = base[p] + 0.01 * torch.randn(base.shape, generator=g)
    new, info = align_global_prompts(rows, base.clone(), outer_loops=3, seed=0)
    assert new.shape == base.shape
    assert info["empty_centres"] == 0
    # every centre sits next to the prompt it was matched with (noise only)
    assert float((new - base).abs().max()) < 0.05, float((new - base).abs().max())


def test_align_leaves_unmatched_centres_untouched():
    base = _prompts(tau=8)
    rows = OrderedDict(a=base[[0, 1, 2]] + 0.01, b=base[[1, 2, 0]] - 0.01)
    new, info = align_global_prompts(rows, base.clone(), outer_loops=2, seed=3)
    assert info["empty_centres"] == 5
    assert torch.equal(new[3:], base[3:]), "centres nobody matched must keep their value"


def test_align_single_client_is_prior_plus_row_mean():
    base = _prompts(tau=4)
    rows = OrderedDict(a=base + 1.0)
    new, _ = align_global_prompts(rows, base.clone(), outer_loops=2, seed=0)
    assert _close(new, base + 0.5, 1e-5), "centre = mean of {previous, matched row}"


def test_align_is_deterministic_for_a_seed():
    base = _prompts()
    rows = OrderedDict(a=base + 0.1, b=base[[3, 2, 1, 0, 5, 4]] - 0.1)
    n1, _ = align_global_prompts(rows, base.clone(), seed=7)
    n2, _ = align_global_prompts(rows, base.clone(), seed=7)
    assert torch.equal(n1, n2)


def test_align_rejects_more_rows_than_centres():
    base = _prompts(tau=3)
    try:
        align_global_prompts(OrderedDict(a=_prompts(tau=4)), base)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_align_with_no_clients_returns_the_previous_pool():
    base = _prompts()
    new, info = align_global_prompts(OrderedDict(), base.clone())
    assert torch.equal(new, base) and info["clients"] == 0


# ================================================ 3. token pools in the real net

_NET = {}


def _build(pool):
    from lib.config.vipt.config import cfg, update_config_from_file
    from lib.models.vipt.ostrack_prompt import build_viptrack
    update_config_from_file(os.path.join(REPO, "experiments/vipt/deep_rgbt.yaml"))
    cfg.MODEL.PRETRAIN_FILE = ""
    cfg.MODEL.BACKBONE.PRIME_POOL_SIZE = pool
    cfg.MODEL.BACKBONE.PRIME_TOP_K = 5
    torch.manual_seed(0)
    return build_viptrack(cfg, training=True)


def _net(pool):
    if pool not in _NET:
        _NET[pool] = _build(pool)
    return _NET[pool]


def _inputs(b=2):
    torch.manual_seed(1)
    return torch.randn(b, 6, 128, 128), torch.randn(b, 6, 256, 256)


def test_pools_add_exactly_three_prime_tensors():
    off, on = _net(0), _net(20)
    extra = set(on.state_dict()) - set(off.state_dict())
    assert set(off.state_dict()) - set(on.state_dict()) == set()
    assert extra == {"backbone.prime_local_prompt", "backbone.prime_global_prompt",
                     "backbone.prime_query_prompt_proj.weight"}, extra
    groups = build_param_groups(on.state_dict().keys())          # strict
    assert {k for k in extra} == {k for k, g in groups.items() if g == PRIME}
    n = sum(on.state_dict()[k].numel() for k in extra)
    assert n == 2 * 20 * 768 + 768 * 768, n
    assert "prime_local_prompt" in " ".join(extra) and PRIME in TRAINABLE_GROUPS


def test_off_creates_no_pool_and_no_prefix():
    off = _net(0)
    assert off.backbone.prime_pool_size == 0
    assert not any("prime" in k for k in off.state_dict())


def test_forward_with_pools_keeps_the_head_input_shape():
    if FAST:
        return
    off, on = _net(0), _net(20)
    z, x = _inputs()
    off.train(); on.train()
    o_off = off(z, x)
    o_on = on(z, x)
    assert o_off["pred_boxes"].shape == o_on["pred_boxes"].shape
    assert o_off["backbone_feat"].shape == o_on["backbone_feat"].shape, \
        (o_off["backbone_feat"].shape, o_on["backbone_feat"].shape)
    assert torch.isfinite(o_on["pred_boxes"]).all()


def test_gradients_reach_both_pools_and_the_query_projection():
    if FAST:
        return
    net = _net(20)
    net.train()
    net.zero_grad()
    z, x = _inputs()
    out = net(z, x)
    reg = net.backbone.prime_reg
    loss = out["pred_boxes"].sum() * 1e-3 - 0.01 * reg
    loss.backward()
    bb = net.backbone
    assert bb.prime_local_prompt.grad is not None and bb.prime_local_prompt.grad.abs().sum() > 0
    assert bb.prime_global_prompt.grad is not None and bb.prime_global_prompt.grad.abs().sum() > 0
    # the query is detached from the task loss; only the relevance term trains it
    assert bb.prime_query_prompt_proj.weight.grad is not None
    assert bb.prime_query_prompt_proj.weight.grad.abs().sum() > 0


def test_prefix_changes_the_prediction_and_usage_is_tracked():
    if FAST:
        return
    net = _net(20)
    net.train()
    z, x = _inputs()
    net.backbone.reset_prime_usage()
    assert not net.backbone.prime_global_used.any()
    a = net(z, x)["pred_boxes"].detach().clone()
    used = net.backbone.prime_global_used.clone()
    assert used.sum() == 5, "exactly top-k rows of the inter pool are used per batch"
    with torch.no_grad():
        net.backbone.prime_local_prompt.add_(1.0)
    b = net(z, x)["pred_boxes"].detach()
    with torch.no_grad():
        net.backbone.prime_local_prompt.sub_(1.0)
    assert not torch.allclose(a, b), "the prompt tokens must influence the output"


def test_selection_takes_the_rows_most_similar_to_the_query():
    bb = _net(20).backbone
    dim = bb.prime_local_prompt.shape[1]
    with torch.no_grad():
        bb.prime_query_prompt_proj.weight.copy_(torch.eye(dim))
        pool = torch.zeros(20, dim)
        for i in range(20):
            pool[i, i] = 1.0                      # row i = unit vector e_i
        bb.prime_local_prompt.copy_(pool)
        bb.prime_global_prompt.copy_(pool)
    v = torch.zeros(1, 4, dim)
    for i, val in zip((3, 7, 11, 13, 17), (5., 4., 3., 2., 1.)):
        v[0, 0, i] = val                          # max over tokens keeps these
    prefix, reg = bb._prime_prefix(v)
    assert prefix.shape == (1, 10, dim)
    chosen = prefix[0, :5].abs().argmax(dim=1).tolist()
    assert sorted(chosen) == [3, 7, 11, 13, 17], chosen
    q = v[0, 0] / v[0, 0].norm()
    expect = 2 * float(q[[3, 7, 11, 13, 17]].sum())      # local + global pool
    assert abs(float(reg) - expect) < 1e-5, (float(reg), expect)


# ================================================ 4. server wiring on stub clients

BASE_KEYS = OrderedDict([
    ("backbone.blocks.0.attn.qkv.weight", "frozen"),
    ("backbone.patch_embed_prompt.proj.weight", AUX_SPECIFIC),
    ("backbone.prompt_blocks.0.conv0_1.weight", AUX_SPECIFIC),
    ("backbone.prompt_blocks.0.conv0_0.weight", RGB_SHARED),
    ("backbone.prompt_blocks.0.conv1x1.weight", FUSION),
    ("backbone.prompt_norms.0.weight", NORM),
])
PRIME_KEYS = OrderedDict([
    ("backbone.prime_local_prompt", PRIME),
    ("backbone.prime_global_prompt", PRIME),
    ("backbone.prime_query_prompt_proj.weight", PRIME),
])
POOL = 4


class StubNet:
    def __init__(self, with_prime):
        self.keys = list(BASE_KEYS) + (list(PRIME_KEYS) if with_prime else [])
        g = torch.Generator().manual_seed(0)
        self._sd = OrderedDict()
        for k in self.keys:
            shape = (POOL, 3) if "prime_" in k and "proj" not in k else (3, 3) if "proj" in k else (6,)
            self._sd[k] = torch.randn(shape, generator=g)

    def state_dict(self):
        return OrderedDict((k, v.clone()) for k, v in self._sd.items())


class StubClient:
    """Uploads its own constant, so every aggregate is hand-checkable."""

    def __init__(self, cid, aux, n, value):
        self.client_id, self.aux_type, self.num_samples = cid, aux, n
        # the merged server asks every client for its dataset bucket (bucketgroups)
        self.dataset_name = {'thermal': 'lasher', 'depth': 'depthtrack',
                             'event': 'visevent'}.get(aux, 'lasher')
        self.value = float(value)
        self.seen = []                  # (round, dispatched, prox_anchor)

    def local_train(self, net, dispatched, plan, round_idx, local_steps,
                    device="cpu", prox_anchor=None):
        self.seen.append((round_idx, OrderedDict((k, v.clone()) for k, v in dispatched.items()),
                          None if prox_anchor is None else
                          OrderedDict((k, v.clone()) for k, v in prox_anchor.items())))
        groups = build_param_groups(net.state_dict().keys())
        open_groups = set(plan.trainable_groups)
        state = OrderedDict()
        for k, v in dispatched.items():
            if groups[k] in open_groups:
                base = torch.full_like(v, self.value)
                if groups[k] == PRIME and "prime_global" in k:
                    base = v + self.value                        # rows move a little
                state[k] = base
        has_aux = self.aux_type != NO_AUX
        mask = OrderedDict((k, has_aux or groups[k] != AUX_SPECIFIC) for k in state)
        update = ClientUpdate(self.client_id, self.num_samples, self.aux_type,
                              state, mask)
        if any("prime_global" in k for k in state):
            used = torch.zeros(POOL, dtype=torch.bool)
            used[[0, 1]] = True
            update.extras["prime_global_used"] = used
        stats = OrderedDict(client_id=self.client_id, aux_type=self.aux_type,
                            round=round_idx)
        stats["mean_Loss/total"] = 1.0
        stats["mean_IoU"] = 0.5
        stats["aux_present_frac"] = 1.0 if has_aux else 0.0
        return update, stats


def _clients():
    return [StubClient("rgbt_a", "thermal", 100, 1.0),
            StubClient("rgbt_b", "thermal", 300, 5.0),
            StubClient("rgb_only", NO_AUX, 200, 9.0),
            StubClient("rgbd_a", "depth", 400, 2.0)]


def _server(variant, with_prime=False, clients=None, **kw):
    kw.setdefault("num_rounds", 3)
    kw.setdefault("local_steps", 2)
    kw.setdefault("verbose", False)
    kw.setdefault("device", "cpu")
    kw.setdefault("out_dir", tempfile.mkdtemp(prefix="fedtest_"))
    clients = clients if clients is not None else _clients()
    return FederatedServer(StubNet(with_prime), clients, cfg=None, settings=None,
                           variant=variant, **kw), clients


# ---- FedMSplit-P ------------------------------------------------------------

def test_fedmsplit_p_defaults_are_the_official_ones():
    s, _ = _server("fedmsplit_p")
    assert s.prox_mu == 0.01 and s.do_aggregate and not s.modality_aware


def test_fedmsplit_p_keeps_one_model_per_client_and_no_global_state_is_dispatched():
    s, cl = _server("fedmsplit_p")
    s.run_round(0)
    s.run_round(1)
    # round 1 start points are the personalised aggregates: all differ from each
    # other for a shared block of two clients with different uploads
    a = cl[0].seen[1][1]["backbone.prompt_blocks.0.conv0_0.weight"]
    b = cl[1].seen[1][1]["backbone.prompt_blocks.0.conv0_0.weight"]
    assert not torch.equal(a, b), "personalised models must differ between clients"
    assert float(a.min()) >= 1.0 - 1e-5 and float(a.max()) <= 9.0 + 1e-5


def test_fedmsplit_p_rgb_only_client_always_gets_the_initial_aux_tensors():
    s, cl = _server("fedmsplit_p")
    init = s.state.initial_aux
    for r in range(3):
        s.run_round(r)
    for _, dispatched, _ in cl[2].seen:
        for k, v in init.items():
            assert torch.equal(dispatched[k], v), k


def test_fedmsplit_p_aux_block_ignores_the_rgb_only_client():
    s, cl = _server("fedmsplit_p")
    s.run_round(0)
    s.run_round(1)
    k = "backbone.patch_embed_prompt.proj.weight"
    for c in (cl[0], cl[1], cl[3]):                  # sensor clients
        v = c.seen[1][1][k]
        assert float(v.max()) <= 5.0 + 1e-5, "the RGB-only client's 9.0 must not enter aux"


def test_fedmsplit_p_prox_anchor_is_the_data_weighted_global_average():
    s, cl = _server("fedmsplit_p")
    s.run_round(0)
    s.run_round(1)
    anchor = cl[0].seen[1][2]
    k = "backbone.prompt_blocks.0.conv0_0.weight"
    expect = (100 * 1.0 + 300 * 5.0 + 200 * 9.0 + 400 * 2.0) / 1000.0
    assert _close(anchor[k], torch.full_like(anchor[k], expect), 1e-5), anchor[k]
    assert cl[0].seen[0][2] is not None


def test_fedmsplit_p_single_client_keeps_its_own_upload():
    s, cl = _server("fedmsplit_p", clients=[StubClient("rgbt_a", "thermal", 10, 3.0)])
    s.run_round(0)
    s.run_round(1)
    for k, v in cl[0].seen[1][1].items():
        if "blocks.0.attn" in k:
            continue
        assert _close(v, torch.full_like(v, 3.0), 1e-6), k


def test_fedmsplit_p_export_is_one_checkpoint_per_sensor_client_in_model_format():
    s, cl = _server("fedmsplit_p")
    s.run_round(0)
    paths = s.export_checkpoint(1)
    names = sorted(os.path.basename(p) for p in paths)
    assert names == ["ViPTrack_rgbd_a_ep0001.pth.tar", "ViPTrack_rgbt_a_ep0001.pth.tar",
                     "ViPTrack_rgbt_b_ep0001.pth.tar"], names
    ref = StubNet(False).state_dict()
    for p in paths:
        blob = torch.load(p, map_location="cpu", weights_only=False)
        assert list(blob["net"]) == list(ref), "same keys, same order as the model"
        assert blob["fed"]["variant"] == "fedmsplit_p" and "client_id" in blob["fed"]
        assert torch.equal(blob["net"]["backbone.blocks.0.attn.qkv.weight"],
                           ref["backbone.blocks.0.attn.qkv.weight"])


def test_fedmsplit_p_save_and_restore_round_trip():
    s, _ = _server("fedmsplit_p")
    s.run_round(0)
    path = s.save_state(1)
    t, _ = _server("fedmsplit_p")
    assert t.load_state(path) == 1
    for cid in s.fmp_states:
        for k in s.fmp_states[cid]:
            assert torch.equal(s.fmp_states[cid][k], t.fmp_states[cid][k])
    for k in s.fmp_anchor:
        assert torch.equal(s.fmp_anchor[k], t.fmp_anchor[k])
    s.run_round(1)
    t.run_round(1)
    for cid in s.fmp_states:
        for k in s.fmp_states[cid]:
            assert torch.equal(s.fmp_states[cid][k], t.fmp_states[cid][k]), (cid, k)


def test_fedmsplit_p_partial_participation_leaves_others_untouched():
    s, cl = _server("fedmsplit_p", clients_per_round=2)
    before = {c: OrderedDict((k, v.clone()) for k, v in st.items())
              for c, st in s.fmp_states.items()}
    rec = s.run_round(0)
    for cid in s.clients:
        if cid not in rec["selected"]:
            for k, v in s.fmp_states[cid].items():
                assert torch.equal(v, before[cid][k])


# ---- FED-PRIME --------------------------------------------------------------

def test_fedprime_refuses_a_net_without_the_pools():
    try:
        _server("fedprime", with_prime=False)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_fedprime_shared_groups_equal_fedavg():
    p, _ = _server("fedprime", with_prime=True)
    a, _ = _server("fedavg", with_prime=False)
    for r in range(2):
        p.run_round(r)
        a.run_round(r)
    for k, g in p.param_groups.items():
        if g in (AUX_SPECIFIC, RGB_SHARED, FUSION, NORM):
            assert torch.equal(p.state.shared[k], a.state.shared[k]), k
            for aux, bank in a.state.per_modality.items():
                if k in bank:
                    assert torch.equal(p.state.per_modality[aux][k], bank[k]), (aux, k)


def test_fedprime_intra_pool_and_query_stay_on_the_client():
    s, cl = _server("fedprime", with_prime=True)
    for r in range(3):
        s.run_round(r)
    lk, qk = "backbone.prime_local_prompt", "backbone.prime_query_prompt_proj.weight"
    for c in cl:
        # round r+1 starts from exactly what the client uploaded in round r
        for r in (0, 1):
            nxt = c.seen[r + 1][1]
            assert _close(nxt[lk], torch.full_like(nxt[lk], c.value), 1e-6), (c.client_id, r)
            assert _close(nxt[qk], torch.full_like(nxt[qk], c.value), 1e-6)
    assert lk not in s.state.shared and qk not in s.state.shared


def test_fedprime_inter_pool_is_aligned_and_identical_for_everyone():
    s, cl = _server("fedprime", with_prime=True)
    g = "backbone.prime_global_prompt"
    first = s.state.shared[g].clone()
    s.run_round(0)
    assert not torch.equal(first, s.state.shared[g]), "the inter pool must move"
    s.run_round(1)
    pools = [c.seen[1][1][g] for c in cl]
    for p_ in pools[1:]:
        assert torch.equal(pools[0], p_), "every client receives the same inter pool"
    assert pools[0].shape == first.shape and torch.isfinite(pools[0]).all()
    assert s.pers_info["clients"] == 4 and s.pers_info["rows"] == 8


def test_fedprime_export_is_per_sensor_client_with_that_clients_pool():
    s, cl = _server("fedprime", with_prime=True)
    s.run_round(0)
    paths = s.export_checkpoint(1)
    assert len(paths) == 3
    lk = "backbone.prime_local_prompt"
    ref = StubNet(True).state_dict()
    for p in paths:
        blob = torch.load(p, map_location="cpu", weights_only=False)
        assert list(blob["net"]) == list(ref)
        cid = blob["fed"]["client_id"]
        own = next(c for c in cl if c.client_id == cid).value
        assert _close(blob["net"][lk], torch.full_like(blob["net"][lk], own), 1e-6)


def test_fedprime_save_and_restore_round_trip():
    s, _ = _server("fedprime", with_prime=True)
    s.run_round(0)
    path = s.save_state(1)
    t, _ = _server("fedprime", with_prime=True)
    assert t.load_state(path) == 1
    s.run_round(1)
    t.run_round(1)
    for k in s.state.shared:
        assert torch.equal(s.state.shared[k], t.state.shared[k]), k
    for cid in s.prime_local:
        for k in s.prime_local[cid]:
            assert torch.equal(s.prime_local[cid][k], t.prime_local[cid][k])


def test_run_config_records_the_fixed_hyperparameters():
    import json
    s, _ = _server("fedprime", with_prime=True)
    cfgd = json.load(open(os.path.join(s.out_dir, "run_config.json")))
    h = cfgd["personalized"]
    assert h["prime_sim_scalar"] == 0.01 and h["prime_outer_loops"] == 5
    assert h["prime_pool_size"] == 20 and h["prime_top_k"] == 5
    s2, _ = _server("fedavg")
    assert "personalized" not in json.load(open(os.path.join(s2.out_dir, "run_config.json")))


def test_existing_variants_never_carry_personalised_state():
    for v in ("local", "fedavg", "fedavg_warm", "modality", "ours", "fedprox"):
        s, _ = _server(v, num_rounds=4)
        assert s.fmp_states is None and s.prime_local is None and s.pers_info is None, v


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
