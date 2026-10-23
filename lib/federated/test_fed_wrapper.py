#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for fed_wrapper.py.

    D:\\Conda\\envs\\sdstrack\\python.exe lib/federated/test_fed_wrapper.py

Builds the REAL ViPT model on CPU. The central acceptance criterion from spec
section 16 is "forward, loss and logging must not fail when template_aux/search_aux is None",
and that is only meaningful against the real backbone -- a stub would prove
nothing about whether patch_embed_prompt survives a zero-filled input.

Batch size 1 keeps a CPU forward to a few seconds. Set FED_WRAPPER_FAST=1 to
skip the forward tests and only run the cheap mask/state-dict ones.
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from lib.config.vipt.config import cfg, update_config_from_file  # noqa: E402
from lib.federated import (  # noqa: E402
    AUX_SPECIFIC,
    FedViPTWrapper,
    build_param_groups,
    is_modality_sensitive,
)
from lib.models.vipt.ostrack_prompt import build_viptrack  # noqa: E402

FAST = os.environ.get("FED_WRAPPER_FAST") == "1"
_CACHE = {}


def get_wrapper():
    if "w" not in _CACHE:
        update_config_from_file(os.path.join(REPO, "experiments/vipt/deep_rgbt.yaml"))
        cfg.MODEL.PRETRAIN_FILE = ""          # no weights needed for shape tests
        net = build_viptrack(cfg, training=True).eval()
        _CACHE["w"] = FedViPTWrapper(net)
    return _CACHE["w"]


def _inputs(b=1):
    return torch.randn(b, 3, 128, 128), torch.randn(b, 3, 256, 256)


# ---------------------------------------------------------------- masks

def test_modality_mask_shape_matches_spec_3_1():
    m = FedViPTWrapper.build_modality_mask("thermal", True)
    assert m == {"rgb": 1, "thermal": 1, "depth": 0, "event": 0}, m
    m = FedViPTWrapper.build_modality_mask("none", False)
    assert m == {"rgb": 1, "thermal": 0, "depth": 0, "event": 0}, m
    # aux_type set but the tensors were absent -> still all zeros
    m = FedViPTWrapper.build_modality_mask("depth", False)
    assert m == {"rgb": 1, "thermal": 0, "depth": 0, "event": 0}, m


def test_explicit_modality_mask_changes_tensor_seen_by_network():
    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.seen = []
        def forward(self, template, search, **kwargs):
            self.seen.append((template.detach().clone(), search.detach().clone()))
            return {"pred_boxes": torch.zeros(template.shape[0], 4),
                    "score_map": torch.zeros(1), "size_map": torch.zeros(1),
                    "offset_map": torch.zeros(1)}
    net = Recorder(); w = FedViPTWrapper(net)
    rgb = torch.ones(1, 3, 2, 2); aux = torch.full_like(rgb, 7)
    w(rgb, rgb, aux, aux, aux_type="thermal",
      modality_mask={"rgb": 1, "thermal": 0, "depth": 0, "event": 0})
    assert torch.equal(net.seen[0][0][:, 3:], torch.zeros_like(aux))
def test_param_mask_blocks_aux_keys_for_rgb_only_client():
    """design note 16: an RGB-only client never writes any auxiliary adapter."""
    w = get_wrapper()
    mask = w.param_mask_for("none", aux_present=False)
    aux_keys = w.aux_specific_keys()
    assert aux_keys, "expected some AUX_SPECIFIC keys"
    assert all(mask[k] is False for k in aux_keys), "aux keys must be masked out"
    shared = [k for k, v in mask.items() if k not in aux_keys]
    assert shared and all(mask[k] for k in shared), "shared keys must stay usable"


def test_param_mask_allows_everything_for_a_client_with_aux():
    w = get_wrapper()
    mask = w.param_mask_for("thermal", aux_present=True)
    assert all(mask.values())


def test_param_mask_covers_exactly_the_trainable_keys():
    w = get_wrapper()
    mask = w.param_mask_for("thermal")
    assert set(mask) == set(w.trainable_state().keys())


# ---------------------------------------------------------------- fed state

def test_trainable_state_excludes_the_frozen_backbone():
    """The whole point: 0.84 M travels, 92.5 M does not."""
    w = get_wrapper()
    st = w.trainable_state()
    assert all("prompt" in k for k in st), \
        [k for k in st if "prompt" not in k][:3]
    n = sum(v.numel() for v in st.values())
    assert 0.8e6 < n < 0.9e6, f"{n} params -- expected ~0.84 M"


def test_load_trainable_state_writes_and_reports():
    w = get_wrapper()
    st = w.trainable_state()
    key = next(iter(st))
    new = {key: torch.full_like(st[key], 0.123)}
    written = w.load_trainable_state(new)
    assert written == [key]
    assert torch.allclose(w.trainable_state()[key], torch.full_like(st[key], 0.123))
    w.load_trainable_state({key: st[key]})            # restore


def test_load_trainable_state_refuses_frozen_keys():
    """A dispatch that tried to overwrite the backbone must raise, not
    silently succeed -- that would turn 0.84 M of traffic into 93 M and
    quietly destroy the pretrained prior."""
    w = get_wrapper()
    frozen_key = next(k for k, g in w.param_groups.items()
                      if g == "frozen" and k in w.net.state_dict())
    try:
        w.load_trainable_state({frozen_key: torch.zeros_like(w.net.state_dict()[frozen_key])})
    except ValueError:
        return
    raise AssertionError("expected ValueError when writing a frozen tensor")


def test_load_trainable_state_rejects_unknown_key():
    w = get_wrapper()
    try:
        w.load_trainable_state({"not.a.real.key": torch.zeros(1)})
    except KeyError:
        return
    raise AssertionError("expected KeyError for an unknown key")


# ---------------------------------------------------------------- forward

def test_forward_with_aux_runs():
    if FAST:
        return
    w = get_wrapper()
    z, x = _inputs()
    za, xa = torch.randn(1, 3, 128, 128), torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = w(z, x, za, xa, aux_type="thermal")
    assert out["pred_boxes"].shape[-1] == 4, out["pred_boxes"].shape
    assert out["aux_present"] is True
    assert out["modality_mask"]["thermal"] == 1
    assert out["aux_fill"] is None


def test_forward_without_aux_does_not_raise():
    """THE spec 16 acceptance criterion."""
    if FAST:
        return
    w = get_wrapper()
    z, x = _inputs()
    with torch.no_grad():
        out = w(z, x, None, None, aux_type="none")
    assert torch.isfinite(out["pred_boxes"]).all(), "missing aux produced NaN/Inf"
    assert out["aux_present"] is False
    assert out["aux_fill"] == "zeros"
    assert out["modality_mask"] == {"rgb": 1, "thermal": 0, "depth": 0, "event": 0}


def test_forward_accepts_packed_six_channel_input():
    """The existing LTRLoader hands over 6-channel tensors; the wrapper must
    take them unchanged so it can drop into the current data pipeline."""
    if FAST:
        return
    w = get_wrapper()
    z6 = torch.randn(1, 6, 128, 128)
    x6 = torch.randn(1, 6, 256, 256)
    with torch.no_grad():
        out = w(z6, x6, aux_type="depth")
    assert out["aux_present"] is True
    assert out["modality_mask"]["depth"] == 1


def test_packed_and_split_inputs_agree():
    """Feeding [rgb|aux] packed must equal feeding them separately."""
    if FAST:
        return
    w = get_wrapper()
    torch.manual_seed(0)
    rgb_z, rgb_x = _inputs()
    aux_z, aux_x = torch.randn(1, 3, 128, 128), torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        a = w(rgb_z, rgb_x, aux_z, aux_x, aux_type="thermal")["pred_boxes"]
        b = w(torch.cat([rgb_z, aux_z], 1), torch.cat([rgb_x, aux_x], 1),
              aux_type="thermal")["pred_boxes"]
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max().item()


def test_aux_fill_rgb_differs_from_zeros():
    """The two fill policies must actually produce different behaviour --
    otherwise recording which one was used would be meaningless."""
    if FAST:
        return
    update_config_from_file(os.path.join(REPO, "experiments/vipt/deep_rgbt.yaml"))
    cfg.MODEL.PRETRAIN_FILE = ""
    net = build_viptrack(cfg, training=True).eval()
    wz = FedViPTWrapper(net, aux_fill="zeros")
    wr = FedViPTWrapper(net, aux_fill="rgb")
    torch.manual_seed(1)
    z, x = _inputs()
    with torch.no_grad():
        a = wz(z, x, None, None, aux_type="none")["pred_boxes"]
        b = wr(z, x, None, None, aux_type="none")["pred_boxes"]
    assert not torch.allclose(a, b, atol=1e-6), \
        "zeros and rgb fill gave identical output; the policy has no effect"


def test_adapter_stats_are_collected_without_touching_the_model():
    if FAST:
        return
    w = get_wrapper()
    z, x = _inputs()
    before = len(w._hooks)
    with torch.no_grad():
        out = w(z, x, None, None, aux_type="none", return_aux_stats=True)
    assert out["adapter_stats"], "expected per-layer prompt norms"
    assert "prompt_norm_mean" in out["adapter_stats"]
    assert len(w._hooks) == before, "hooks must be removed after the forward"


def test_bad_channel_count_is_rejected():
    w = get_wrapper()
    try:
        w._split_if_packed(torch.randn(1, 4, 8, 8))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a 4-channel tensor")


# ---------------------------------------------------------------- runner

def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    if FAST:
        print("FED_WRAPPER_FAST=1 -> forward tests skipped\n")
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
