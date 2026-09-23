"""The federated server: round loop, aggregation, logging, checkpoint export.

    D:\\Conda\\envs\\sdstrack\\python.exe lib/federated/server.py --variant ours --rounds 4 --dry-run

What this file is responsible for
---------------------------------
- deciding which clients train in a round, reproducibly
- dispatching each client the view of the global state its modality needs
- handing the updates to the Aggregator with THIS round's phase
- writing a log line per round that fully explains that round (spec 16)
- exporting a checkpoint the OFFICIAL evaluation script can load unchanged

The four baselines are one flag, not four call sites
----------------------------------------------------
`--variant` sets every switch that distinguishes them. Configuring a baseline by
hand across several flags is exactly how an ablation ends up comparing something
other than what the table claims, and that error is invisible afterwards.

  local     no aggregation at all; each client keeps its own parameters
  fedavg    sample-weighted average over the union of uploaded keys, no warm-up
  modality  per-modality banks for aux_specific tensors, no warm-up
  ours      per-modality banks + progressive warm-up

`local` is the only one that is not a variation of the aggregator: it keeps a
per-client state and never merges, which is the floor the others must beat.

Exported checkpoints
--------------------
`export_checkpoint` writes `{'net': <full state_dict>, ...}` -- the frozen
backbone plus the aggregated prompt tensors -- under the filename the official
`lib/test/parameter/vipt.py` already looks for. So a federated run is evaluated
by exactly the pipeline that produced the reproduction numbers, with no second
code path to keep honest. A modality-aware run exports ONE file per modality,
because there is no single set of aux_specific weights to export.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import OrderedDict

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lib.federated.aggregator import (  # noqa: E402
    NO_AUX, Aggregator, GlobalState, make_initial_state,
)
from lib.federated.client import FederatedClient, seed_everything  # noqa: E402
from lib.federated.client_data import load_split  # noqa: E402
from lib.federated.param_groups import build_param_groups  # noqa: E402
from lib.federated.warmup import WarmupScheduler  # noqa: E402

DEFAULT_SPLIT = os.path.join(
    _REPO, "configs", "fed", "client_splits",
    "lasher_depthtrack_visevent_seed42_v2.json")

#: variant -> (modality_aware, use_warmup, aggregate)
#
# 2026-09-21: `fedavg_warm` added. The two mechanisms form a 2x2 grid and the
# (no modality awareness, warm-up) cell was missing, so the warm-up effect could
# only ever be measured ON TOP of modality awareness -- never on its own. That
# gap mattered: at 20 rounds `ours - modality` was +1.29 AUC (t=2.35) and it
# looked like warm-up was a contribution; at 120 rounds the same difference is
# -0.14 (t=-0.27). Without this cell there is no way to tell whether warm-up
# does anything at all, and a reviewer would ask on sight.
VARIANTS = OrderedDict(
    local=(False, False, False),
    fedavg=(False, False, True),
    fedavg_warm=(False, True, True),
    modality=(True, False, True),
    ours=(True, True, True),
    fedprox=(False, False, True),
)


class FederatedServer:
    def __init__(self, net, clients, cfg, settings, variant="ours",
                 num_rounds=120, clients_per_round=None, local_steps=50,
                 seed=42, out_dir=None, device="cuda", verbose=True,
                 modality_alpha=1.0, rgb_only_weight=1.0, prox_mu=None):
        if variant not in VARIANTS:
            raise ValueError("variant must be one of %s" % list(VARIANTS))
        self.net = net
        self.clients = OrderedDict((c.client_id, c) for c in clients)
        self.cfg = cfg
        self.settings = settings
        self.variant = variant
        self.modality_aware, self.use_warmup, self.do_aggregate = VARIANTS[variant]
        self.num_rounds = int(num_rounds)
        self.clients_per_round = clients_per_round
        self.local_steps = int(local_steps)
        self.seed = int(seed)
        self.device = device
        self.verbose = verbose

        self.out_dir = out_dir or os.path.join(
            _REPO, "output", "fed", "%s_seed%d" % (variant, seed))
        os.makedirs(self.out_dir, exist_ok=True)

        self.param_groups = build_param_groups(net.state_dict().keys())
        self.frozen_reference = OrderedDict(
            (k, v.detach().clone()) for k, v in net.state_dict().items())

        aux_types = sorted({c.aux_type for c in clients if c.aux_type != NO_AUX})
        self.aux_types = tuple(aux_types)
        self.state = make_initial_state(net.state_dict(), self.param_groups,
                                        aux_types=self.aux_types)
        # `local` keeps a separate state per client and never merges them.
        self.local_states = (
            {cid: make_initial_state(net.state_dict(), self.param_groups,
                                     aux_types=self.aux_types)
             for cid in self.clients} if variant == "local" else None)

        self.scheduler = (WarmupScheduler.default(num_rounds) if self.use_warmup
                          else WarmupScheduler.no_warmup(
                              num_rounds, modality_aware=self.modality_aware))
        self.modality_alpha = float(modality_alpha)
        self.rgb_only_weight = float(rgb_only_weight)
        self.prox_mu = (0.01 if prox_mu is None else float(prox_mu)) if variant == "fedprox" else 0.0
        if variant != "fedprox" and prox_mu not in (None, 0, 0.0):
            raise ValueError("prox_mu is only valid with fedprox")
        if self.rgb_only_weight <= 0:
            raise ValueError("rgb_only_weight must be positive")
        self.aggregator = Aggregator(self.param_groups,
                                     modality_aware=self.modality_aware,
                                     modality_alpha=self.modality_alpha,
                                     missing_client_weight=self.rgb_only_weight)
        self.log_path = os.path.join(self.out_dir, "round_metrics.jsonl")

    # ------------------------------------------------------------- selection

    def select(self, round_idx, plan):
        """Which clients train this round. Seeded by round so a run replays."""
        ids = list(self.clients)
        k = self.clients_per_round or plan.clients_per_round or len(ids)
        if isinstance(k, float) and k <= 1.0:
            k = max(1, int(round(k * len(ids))))
        k = min(int(k), len(ids))
        if k == len(ids):
            return ids
        rng = random.Random(self.seed * 7919 + round_idx)
        return sorted(rng.sample(ids, k))

    # ----------------------------------------------------------------- round

    def run_round(self, round_idx):
        plan = self.scheduler.resolve(round_idx)
        chosen = self.select(round_idx, plan)
        if self.verbose:
            print(plan.one_line())
            print("  clients: %s" % ", ".join(chosen))

        updates, client_stats = [], []
        t0 = time.time()
        for cid in chosen:
            client = self.clients[cid]
            src = (self.local_states[cid] if self.local_states is not None
                   else self.state)
            dispatched = src.view_for(client.aux_type)
            update, stats = client.local_train(
                self.net, dispatched, plan, round_idx,
                local_steps=self.local_steps, device=self.device)
            updates.append(update)
            client_stats.append(stats)

            if self.local_states is not None:
                # no aggregation: fold this client's own result straight back
                # into its private state, so next round it resumes from itself.
                for k_, v in update.state.items():
                    if k_ in src.shared:
                        src.shared[k_] = v.detach().clone()
                    bank = src.per_modality.get(client.aux_type)
                    if bank is not None and k_ in bank:
                        bank[k_] = v.detach().clone()
            if self.verbose:
                print("    %-11s loss %.4f  IoU %.4f  aux %.2f"
                      % (cid, stats.get("mean_Loss/total", float("nan")),
                         stats.get("mean_IoU", float("nan")),
                         stats.get("aux_present_frac", 0.0)))

        agg_summary = None
        if self.do_aggregate:
            # Honour the phase's own decision, not just the variant's.
            #
            # 2026-09-18: Aggregator reads only its constructor flag, while
            # RoundPlan.modality_aware was written to the log and consumed by
            # nobody -- so the log could claim a strategy the aggregator was not
            # following. It happens to be harmless with the default phase table
            # (the one phase with modality_aware=False, rgb_common, also keeps
            # aux_specific out of trainable_groups, so no modality-sensitive key
            # is ever open there), but that is a coincidence of that table, not
            # a guarantee. A phase that deliberately shares adapters early
            # would have been silently ignored. `and` keeps the variant as the
            # ceiling: fedavg stays pooled no matter what a phase asks for.
            self.aggregator.modality_aware = (
                self.modality_aware and plan.modality_aware)
            self.state, agg_stats = self.aggregator.aggregate(
                self.state, updates, round_idx=round_idx, phase=plan)
            agg_summary = agg_stats.group_summary()

        record = OrderedDict(plan.as_dict())
        # The scheduler only selects parameter groups. Loss weights come from
        # the same objective factory used by clients, so the log cannot drift.
        train_cfg = getattr(self.cfg, "TRAIN", None)
        record["loss_weights"] = {
            "giou": float(getattr(train_cfg, "GIOU_WEIGHT", 2.0)),
            "l1": float(getattr(train_cfg, "L1_WEIGHT", 5.0)),
            "focal": 1.0,
        }
        # 2026-09-21: the log must carry what the aggregator DID, not what the
        # phase asked for. `plan.as_dict()` reports the phase's wish, and the
        # variant is the ceiling -- for `fedavg_warm` the two are opposite in
        # every phase after rgb_common, so an unpatched record would claim
        # modality-aware aggregation on a run that pooled everything. Same
        # failure as the 2026-09-18 one, in a new place: a value written to the
        # log next to a separate value that actually drives behaviour.
        record["modality_aware"] = (self.aggregator.modality_aware
                                    if self.do_aggregate else False)
        record["phase_modality_aware"] = plan.modality_aware
        record["modality_alpha"] = self.modality_alpha
        record["rgb_only_weight"] = self.rgb_only_weight
        record["prox_mu"] = self.prox_mu
        record["variant"] = self.variant
        record["seed"] = self.seed
        record["selected"] = chosen
        record["local_steps"] = self.local_steps
        record["seconds"] = round(time.time() - t0, 2)
        record["clients"] = client_stats
        if agg_summary is not None:
            record["aggregation"] = _jsonable(agg_summary)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record

    def run(self, start=0, save_every=10):
        if self.verbose:
            print("variant=%s  modality_aware=%s  warmup=%s  aggregate=%s"
                  % (self.variant, self.modality_aware, self.use_warmup,
                     self.do_aggregate))
            print("clients=%d  rounds=%d  local_steps=%d  out=%s"
                  % (len(self.clients), self.num_rounds, self.local_steps,
                     self.out_dir))
        for r in range(start, self.num_rounds):
            self.run_round(r)
            if save_every and ((r + 1) % save_every == 0 or r + 1 == self.num_rounds):
                self.save_state(r + 1)
        return self.state

    # ------------------------------------------------------------ checkpoint

    def save_state(self, round_idx):
        """The federated state itself -- small (about 3.4 MB), resumable."""
        path = os.path.join(self.out_dir, "fed_state_r%04d.pt" % round_idx)
        blob = {"round": round_idx, "variant": self.variant,
                "seed": self.seed,
                "state": self.state.to_flat_dict()}
        if self.local_states is not None:
            # 2026-09-20: `local` never updates self.state, so persisting only
            # it silently discarded every client's trained private model.
            blob["local_states"] = {
                cid: state.to_flat_dict()
                for cid, state in self.local_states.items()
            }
        torch.save(blob, path)
        return path

    def export_checkpoint(self, round_idx, aux_type=None, out_dir=None,
                          client_id=None):
        """Write a checkpoint the OFFICIAL evaluation script loads unchanged.

        Frozen backbone + this modality's aggregated prompt tensors, saved as
        {'net': state_dict} under ViPTrack_ep%04d.pth.tar. Returns the paths.
        """
        out_dir = out_dir or os.path.join(self.out_dir, "checkpoints")
        os.makedirs(out_dir, exist_ok=True)
        if self.local_states is not None:
            if client_id is not None:
                if client_id not in self.local_states:
                    raise KeyError("unknown local client: %s" % client_id)
                client_ids = [client_id]
            elif aux_type is not None:
                client_ids = [cid for cid, client in self.clients.items()
                              if client.aux_type == aux_type]
                if not client_ids:
                    raise KeyError("no local client has aux_type: %s" % aux_type)
            else:
                client_ids = list(self.clients)
            targets = [(cid, self.clients[cid].aux_type,
                        self.local_states[cid]) for cid in client_ids]
        else:
            aux_types = ([aux_type] if aux_type else
                         (list(self.aux_types) or [NO_AUX]))
            targets = [(None, aux, self.state) for aux in aux_types]

        written = []
        for cid, aux, source in targets:
            sd = OrderedDict((k, v.detach().clone())
                             for k, v in self.frozen_reference.items())
            view = source.view_for(aux)
            missing = [k for k in view if k not in sd]
            if missing:
                raise KeyError("aggregated keys absent from the model: %s"
                               % missing[:5])
            for k, v in view.items():
                sd[k] = v.detach().clone().to(sd[k].dtype)
            name = ("ViPTrack_%s_ep%04d.pth.tar" % (cid, round_idx)
                    if cid is not None else
                    ("ViPTrack_ep%04d.pth.tar" % round_idx if len(targets) == 1
                     else "ViPTrack_%s_ep%04d.pth.tar" % (aux, round_idx)))
            path = os.path.join(out_dir, name)
            provenance = {"variant": self.variant, "aux_type": aux,
                          "round": round_idx, "seed": self.seed}
            if cid is not None:
                provenance["client_id"] = cid
            torch.save({"net": sd, "epoch": round_idx,
                        "fed": provenance}, path)
            written.append(path)
        return written


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


# ------------------------------------------------------------------- driver

class _Settings:
    def __init__(self, device):
        self.local_rank = -1
        self.device = device
        self.num_template = 1
        self.num_search = 1


def build_everything(args):
    from lib.config.vipt.config import cfg, update_config_from_file
    from lib.models.vipt.ostrack_prompt import build_viptrack
    from lib.train.base_functions import update_settings

    update_config_from_file(os.path.join(_REPO, "experiments/vipt/%s.yaml"
                                         % args.config))
    settings = _Settings(args.device)
    update_settings(settings, cfg)
    cfg.TRAIN.NUM_WORKER = args.workers

    seed_everything(args.seed)
    net = build_viptrack(cfg, training=True).to(args.device)

    _, by_id = load_split(args.split)
    wanted = args.clients.split(",") if args.clients else list(by_id)
    prox_mu = args.prox_mu if args.prox_mu is not None else (0.01 if args.variant == "fedprox" else 0.0)
    clients = [FederatedClient(by_id[cid], cfg, settings, base_seed=args.seed,
                               prox_mu=prox_mu,
                               verbose=False)
               for cid in wanted]
    return net, clients, cfg, settings


def main(argv=None):
    p = argparse.ArgumentParser(description="Federated ViPT server")
    p.add_argument("--variant", default="ours", choices=list(VARIANTS))
    p.add_argument("--modality-alpha", type=float, default=1.0,
                   help="partial sharing for modality-sensitive tensors: "
                        "1.0 = full isolation (default, what every result "
                        "before 2026-09-21 used), 0.0 = the pooled FedAvg "
                        "adapter, in between = a blend. No effect without "
                        "modality awareness.")
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--rgb-only-weight", type=float, default=1.0)
    p.add_argument("--prox-mu", type=float, default=None)
    p.add_argument("--config", default="repro_rgbt")
    p.add_argument("--rounds", type=int, default=120)
    p.add_argument("--local-steps", type=int, default=50)
    p.add_argument("--clients-per-round", type=float, default=None,
                   help="count, or a fraction <= 1.0; default: the phase's own")
    p.add_argument("--clients", default="", help="comma-separated subset, for tests")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--export-at-end", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="print the round plan and the selection, train nothing")
    args = p.parse_args(argv)
    if args.prox_mu is not None and args.variant != "fedprox":
        raise ValueError("--prox-mu is only valid with --variant fedprox")

    if args.dry_run:
        _, by_id = load_split(args.split)
        ids = args.clients.split(",") if args.clients else list(by_id)
        mod_aware, use_warmup, _ = VARIANTS[args.variant]
        sched = (WarmupScheduler.default(args.rounds) if use_warmup
                 else WarmupScheduler.no_warmup(args.rounds, modality_aware=mod_aware))
        print("variant=%s  clients=%d" % (args.variant, len(ids)))
        for r in range(args.rounds):
            print(sched.resolve(r).one_line())
        return 0

    net, clients, cfg, settings = build_everything(args)
    server = FederatedServer(
        net, clients, cfg, settings, variant=args.variant,
        modality_alpha=args.modality_alpha,
        rgb_only_weight=args.rgb_only_weight,
        prox_mu=args.prox_mu,
        num_rounds=args.rounds, clients_per_round=args.clients_per_round,
        local_steps=args.local_steps, seed=args.seed,
        out_dir=args.out_dir or None, device=args.device)
    server.run(save_every=args.save_every)
    if args.export_at_end:
        for path in server.export_checkpoint(args.rounds):
            print("exported %s" % path)
    print("log -> %s" % server.log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
