#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Smoke test for client.py: one client, a few real local steps.

    D:\\Conda\\envs\\sdstrack\\python.exe lib/federated/smoke_client.py

Real model, real data, real loss. The point is to prove the whole local path
runs and that the upload has the right SHAPE -- not to reach any accuracy.

Checks, in order of what would hurt most if wrong:
  1. the dispatched state is applied and only trainable keys are written
  2. the phase's frozen groups really receive no gradient
  3. an RGB-only client uploads aux_specific tensors marked False
  4. an RGB-only client's aux_specific tensors do not MOVE
  5. the loss is finite and the parameters actually changed
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import OrderedDict

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from lib.config.vipt.config import cfg, update_config_from_file  # noqa: E402
from lib.federated.aggregator import make_initial_state  # noqa: E402
from lib.federated.client import FederatedClient  # noqa: E402
from lib.federated.client_data import load_split  # noqa: E402
from lib.federated.param_groups import (  # noqa: E402
    AUX_SPECIFIC, TRAINABLE_GROUPS, build_param_groups,
)
from lib.federated.warmup import WarmupScheduler  # noqa: E402
from lib.models.vipt.ostrack_prompt import build_viptrack  # noqa: E402
from lib.train.base_functions import update_settings  # noqa: E402

SPLIT = os.path.join(REPO, "configs", "fed", "client_splits",
                     "lasher_depthtrack_visevent_seed42.json")


class _Settings:
    """The handful of attributes the official processing/actor read off
    `settings`. Built here instead of importing the trainer's admin stack, so
    the smoke test does not need the whole training entry point."""

    def __init__(self):
        self.local_rank = -1
        self.device = "cuda"
        self.num_template = 1
        self.num_search = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", default="rgbd_003",
                    help="client id; rgbd_003 has 1 sequence so it builds fast")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--round", type=int, default=25,
                    help="round index; 25 lands in adapter_warmup by default")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    update_config_from_file(os.path.join(REPO, "experiments/vipt/repro_rgbt.yaml"))
    settings = _Settings()
    update_settings(settings, cfg)
    # cfg is an EasyDict here (lib/config/vipt/config.py), not a yacs CfgNode --
    # there is no freeze/defrost, assignment is direct.
    cfg.TRAIN.NUM_WORKER = args.workers

    print("building ViPT ...")
    t0 = time.time()
    net = build_viptrack(cfg, training=True).to(args.device)
    print("  %.1fs" % (time.time() - t0))

    param_groups = build_param_groups(net.state_dict().keys())
    _, by_id = load_split(SPLIT)
    client_cfg = by_id[args.client]

    state = make_initial_state(net.state_dict(), param_groups,
                               aux_types=("thermal", "depth", "event"))
    plan = WarmupScheduler.default(num_rounds=120).resolve(args.round)
    print(plan.one_line())

    client = FederatedClient(client_cfg, cfg, settings, base_seed=42, verbose=True)
    print(client)

    dispatched = state.view_for(client.aux_type)
    before = OrderedDict((k, v.detach().clone())
                         for k, v in net.state_dict().items()
                         if param_groups.get(k) in TRAINABLE_GROUPS)

    t0 = time.time()
    update, stats = client.local_train(net, dispatched, plan, args.round,
                                       local_steps=args.steps, device=args.device)
    dt = time.time() - t0

    print("")
    for k, v in stats.items():
        print("  %-22s %s" % (k, ("%.5f" % v) if isinstance(v, float) else v))
    print("  %-22s %.1fs (%.2f s/step)" % ("wall", dt, dt / args.steps))

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print("  [%s] %s%s" % ("OK " if cond else "FAIL", name,
                               "" if cond else "  <- " + detail))
        ok = ok and cond

    print("")
    open_groups = set(plan.trainable_groups)

    uploaded_groups = {param_groups[k] for k in update.state}
    check("upload carries only this phase's groups",
          uploaded_groups <= open_groups,
          "extra: %s" % (uploaded_groups - open_groups))

    frozen_moved = [k for k, v in net.state_dict().items()
                    if param_groups.get(k) in TRAINABLE_GROUPS
                    and param_groups[k] not in open_groups
                    and not torch.equal(v, before[k])]
    check("groups frozen this phase did not move", not frozen_moved,
          "moved: %s" % frozen_moved[:3])

    moved = [k for k in update.state if not torch.equal(update.state[k].cpu(), before[k].cpu())]
    check("open parameters actually changed", len(moved) > 0,
          "nothing moved -- lr, requires_grad or the optimizer is wrong")

    aux_keys = [k for k in update.state if param_groups[k] == AUX_SPECIFIC]
    if client.aux_type == "none":
        check("RGB-only client marks aux_specific False",
              aux_keys and all(not update.param_mask[k] for k in aux_keys),
              "aux keys present: %d" % len(aux_keys))
        aux_moved = [k for k in aux_keys
                     if not torch.equal(update.state[k].cpu(), before[k].cpu())]
        check("RGB-only client's aux_specific did not move", not aux_moved,
              "moved: %s" % aux_moved[:3])
    else:
        check("sensor client marks aux_specific True",
              all(update.param_mask[k] for k in aux_keys) if aux_keys else True)

    losses = [v for k, v in stats.items() if k.startswith("mean_Loss")]
    check("loss is finite", all(v == v and abs(v) < 1e4 for v in losses),
          "losses: %s" % losses)

    print("")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
