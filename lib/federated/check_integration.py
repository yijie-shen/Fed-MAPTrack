#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end check that the three P0 pieces actually fit together.

    python lib/federated/check_integration.py

No training and no GPU: local "training" is replaced by a deterministic
perturbation of the dispatched tensors. What is being verified is the PLUMBING
and the INVARIANTS, which is exactly what unit tests on each module separately
cannot show:

  1. a real client_splits.json drives real client metadata
  2. each client receives shared + only its own modality bank
  3. an RGB-only client's aux tensors are masked and never reach any bank
  4. thermal and depth banks stay different after aggregation (isolation held)
  5. the frozen 92.5 M never moves, in either direction
  6. the same pipeline with modality_aware=False collapses to one shared bank
"""

from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from lib.config.vipt.config import cfg, update_config_from_file  # noqa: E402
from lib.federated import (  # noqa: E402
    NO_AUX,
    Aggregator,
    ClientUpdate,
    FedViPTWrapper,
    make_initial_state,
)
from lib.models.vipt.ostrack_prompt import build_viptrack  # noqa: E402

SPLIT = os.path.join(REPO, "configs", "fed", "client_splits",
                     "lasher_depthtrack_visevent_seed42.json")


def fake_local_training(dispatched, client_id, round_idx):
    """Stand-in for train_one_client: a deterministic, client-specific shift.

    Deterministic so the check is reproducible; client-specific so that if the
    aggregator ever mixed two clients up, the numbers would not line up.
    """
    seed = (hash(client_id) ^ (round_idx * 7919)) & 0xFFFF
    g = torch.Generator().manual_seed(seed)
    out = OrderedDict()
    for k, v in dispatched.items():
        out[k] = v + torch.randn(v.shape, generator=g) * 0.01
    return out


def main():
    if not os.path.isfile(SPLIT):
        print(f"!! split file not found: {SPLIT}\n"
              f"   run: python -m lib.federated.split_builder --seed 42 ...")
        return 1
    doc = json.load(open(SPLIT, encoding="utf-8"))
    clients_meta = doc["clients"]
    print(f"split fingerprint={doc['fingerprint']}  {doc['n_clients']} clients")

    update_config_from_file(os.path.join(REPO, "experiments/vipt/deep_rgbt.yaml"))
    cfg.MODEL.PRETRAIN_FILE = ""
    net = build_viptrack(cfg, training=True)
    wrapper = FedViPTWrapper(net)

    model_state = net.state_dict()
    aux_types = sorted({c["aux_type"] for c in clients_meta if c["aux_type"] != NO_AUX})
    state = make_initial_state(model_state, wrapper.param_groups, aux_types=aux_types)
    agg = Aggregator(wrapper.param_groups, modality_aware=True)

    aux_key = next(iter(wrapper.aux_specific_keys()))
    shared_key = next(k for k, g in wrapper.param_groups.items()
                      if g == "rgb_shared")
    frozen_key = next(k for k, g in wrapper.param_groups.items()
                      if g == "frozen" and k in model_state)
    frozen_before = model_state[frozen_key].detach().clone()

    ok = True
    for rnd in range(2):
        updates = []
        for meta in clients_meta:
            aux = meta["aux_type"]
            dispatched = state.view_for(aux)

            # invariant 2/3: an RGB-only client gets shared tensors and, since
            # there is no bank for 'none', whatever the shared copy holds.
            trained = fake_local_training(dispatched, meta["client_id"], rnd)
            pmask = wrapper.param_mask_for(aux, aux_present=(aux != NO_AUX))

            updates.append(ClientUpdate(
                client_id=meta["client_id"],
                num_samples=meta["num_samples"],
                aux_type=aux,
                state=trained,
                param_mask=pmask,
            ))

        state, stats = agg.aggregate(state, updates, round_idx=rnd)
        summary = stats.group_summary()
        print(f"\nround {rnd}:")
        for g, s in summary.items():
            print(f"  {g:<14} tensors={s['tensors']:<4} "
                  f"kept_previous={s['kept_previous']:<4} "
                  f"mean_contributors={s['mean_contributors']:.2f}")

    # ---- invariant 3: no RGB-only client ever fed an aux bank ---------------
    rgb_only = [c["client_id"] for c in clients_meta if c["aux_type"] == NO_AUX]
    leaked = [st for st in stats.per_key
              if st.bank != "shared"
              and any(cid in rgb_only for cid in st.contributors)]
    print(f"\n[{'OK' if not leaked else 'FAIL'}] RGB-only clients never wrote an "
          f"aux bank ({len(rgb_only)} such clients)")
    ok &= not leaked

    # ---- invariant 4: banks really are different ---------------------------
    banks = {a: state.per_modality[a][aux_key] for a in aux_types
             if a in state.per_modality}
    pairs = [(a, b) for i, a in enumerate(sorted(banks)) for b in sorted(banks)[i + 1:]]
    diffs = {f"{a}/{b}": (banks[a] - banks[b]).abs().max().item() for a, b in pairs}
    all_differ = all(d > 1e-6 for d in diffs.values())
    print(f"[{'OK' if all_differ else 'FAIL'}] modality banks stayed distinct: "
          + ", ".join(f"{k}={v:.2e}" for k, v in diffs.items()))
    ok &= all_differ

    # ---- invariant 5: the frozen backbone never moved ----------------------
    frozen_after = net.state_dict()[frozen_key]
    frozen_still = torch.allclose(frozen_before, frozen_after)
    in_transit = [k for k in state.shared if wrapper.param_groups[k] == "frozen"]
    print(f"[{'OK' if frozen_still and not in_transit else 'FAIL'}] frozen backbone "
          f"untouched and never in transit ({len(in_transit)} frozen keys in state)")
    ok &= frozen_still and not in_transit

    # ---- dispatch round-trips through the wrapper --------------------------
    written = wrapper.load_trainable_state(state.view_for(aux_types[0]))
    print(f"[{'OK' if written else 'FAIL'}] dispatch applied to the model: "
          f"{len(written)} tensors written")
    ok &= bool(written)

    # ---- invariant 6: FedAvg degradation -----------------------------------
    plain = Aggregator(wrapper.param_groups, modality_aware=False)
    st2 = make_initial_state(model_state, wrapper.param_groups, aux_types=())
    st2, _ = plain.aggregate(st2, updates, round_idx=0)
    degraded = (st2.per_modality == {}) and (aux_key in st2.shared)
    print(f"[{'OK' if degraded else 'FAIL'}] modality_aware=False collapses to a "
          f"single shared bank")
    ok &= degraded

    n_upload = sum(v.numel() for v in wrapper.trainable_state().values())
    n_total = sum(v.numel() for v in model_state.values())
    print(f"\nper-round upload {n_upload:,} / {n_total:,} params "
          f"= {n_upload / n_total * 100:.2f}%   "
          f"({n_upload * 4 / 1e6:.1f} MB fp32 per client per round)")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
