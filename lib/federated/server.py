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
import subprocess
from datetime import datetime, timezone
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
from lib.federated.param_groups import (  # noqa: E402
    AUX_SPECIFIC, FUSION, NORM, PRIME, RGB_SHARED, build_param_groups)
from lib.federated.personalized import (  # noqa: E402
    FEDMSPLIT_PROX_LAMBDA, PRIME_OUTER_LOOPS, PRIME_POOL_SIZE, PRIME_SIM_SCALAR,
    PRIME_TOP_K, align_global_prompts, fedmsplit_p_aggregate)
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
    # 2026-10-16: the two baselines of Phung et al. (ICCV'25). Both are
    # personalised: one model per client, exported per sensor client. Their
    # aggregation lives in _aggregate_fedmsplit_p / _after_aggregate_fedprime.
    fedmsplit_p=(False, False, True),
    fedprime=(False, False, True),
)

#: the variants that keep a private model per client
NEW_PERSONALIZED = ("fedmsplit_p", "fedprime")
#: parameter groups that FedMSplit-P treats as blocks
FEDMSPLIT_BLOCKS = (AUX_SPECIFIC, RGB_SHARED, FUSION, NORM)


class FederatedServer:
    def __init__(self, net, clients, cfg, settings, variant="ours",
                 num_rounds=120, clients_per_round=None, local_steps=50,
                 seed=42, out_dir=None, device="cuda", verbose=True,
                 modality_alpha=1.0, rgb_only_weight=1.0, prox_mu=None,
                 split_path=None, prime_sim_scalar=None, prime_outer_loops=None):
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
        if os.path.exists(os.path.join(self.out_dir, "round_metrics.jsonl")) or \
           any(n.startswith("fed_state_") for n in os.listdir(self.out_dir)):
            raise RuntimeError("refusing to reuse non-empty federated output directory")
        self.run_id = "%s-%d-%d" % (variant, seed, int(time.time() * 1000))

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

        # ---- FedMSplit-P / FED-PRIME private state (2026-10-16) -------------
        self.fmp_states = None       # fedmsplit_p: {cid: {key: tensor}}
        self.fmp_anchor = None       # fedmsplit_p: data-weighted mean of uploads
        self.prime_local = None      # fedprime: {cid: {key: tensor}} intra pool + query
        self.prime_global_key = None
        self.pers_info = None
        if variant == "fedmsplit_p":
            self.fmp_states = {
                cid: OrderedDict((k, v.detach().clone())
                                 for k, v in self.state.shared.items())
                for cid in self.clients}
            self.fmp_anchor = OrderedDict((k, v.detach().clone())
                                          for k, v in self.state.shared.items())
            # nothing in `self.state` is dispatched or exported for this variant
        if variant == "fedprime":
            backbone = getattr(net, "backbone", None)
            if backbone is not None and not getattr(backbone, "prime_pool_size", 0):
                raise ValueError("fedprime needs a model built with "
                                 "MODEL.BACKBONE.PRIME_POOL_SIZE > 0")
            prime_keys = [k for k, g in self.param_groups.items() if g == PRIME]
            gkeys = [k for k in prime_keys if k.endswith("prime_global_prompt")]
            if len(gkeys) != 1 or len(prime_keys) != 3:
                raise ValueError("expected 3 FED-PRIME tensors (local pool, "
                                 "global pool, query projection), got %s" % prime_keys)
            self.prime_global_key = gkeys[0]
            local_keys = [k for k in prime_keys if k != self.prime_global_key]
            # the intra pool and the query projection never leave the client
            self.prime_local = {
                cid: OrderedDict((k, self.state.shared[k].detach().clone())
                                 for k in local_keys)
                for cid in self.clients}
            for k in local_keys:
                del self.state.shared[k]
        self.prime_sim_scalar = float(PRIME_SIM_SCALAR if prime_sim_scalar is None
                                      else prime_sim_scalar)
        self.prime_outer_loops = int(PRIME_OUTER_LOOPS if prime_outer_loops is None
                                     else prime_outer_loops)

        self.scheduler = (WarmupScheduler.default(num_rounds) if self.use_warmup
                          else WarmupScheduler.no_warmup(
                              num_rounds, modality_aware=self.modality_aware))
        self.modality_alpha = float(modality_alpha)
        self.rgb_only_weight = float(rgb_only_weight)
        # FedMSplit-P's proximal weight is the official default (0.01), which is
        # also the project's FedProx mu; it pulls toward the global average.
        if variant == "fedmsplit_p" and prox_mu is None:
            prox_mu = FEDMSPLIT_PROX_LAMBDA
        self.prox_mu = (0.01 if prox_mu is None else float(prox_mu)) if variant in ("fedprox", "fedmsplit_p") else 0.0
        if variant not in ("fedprox", "fedmsplit_p") and prox_mu not in (None, 0, 0.0):
            raise ValueError("prox_mu is only valid with fedprox / fedmsplit_p")
        if self.rgb_only_weight <= 0:
            raise ValueError("rgb_only_weight must be positive")
        self.aggregator = Aggregator(self.param_groups,
                                     modality_aware=self.modality_aware,
                                     modality_alpha=self.modality_alpha,
                                     missing_client_weight=self.rgb_only_weight)
        self.log_path = os.path.join(self.out_dir, "round_metrics.jsonl")
        self.param_stats_path = os.path.join(self.out_dir, "param_group_stats.jsonl")
        try:
            git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               cwd=_REPO, text=True).strip()
        except Exception:
            git_hash = "unknown"
        train_cfg = getattr(cfg, "TRAIN", None)
        actual_cfg = {k: getattr(train_cfg, k, None) for k in
                      ("BATCH_SIZE", "ACCUM_ITER", "LR", "WEIGHT_DECAY", "FIX_BN", "GRAD_CLIP_NORM")}
        actual_cfg["ACCUM_ITER"] = getattr(settings, "accum_iter", actual_cfg["ACCUM_ITER"])
        try:
            git_clean = not bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=_REPO, text=True).strip())
        except Exception:
            git_clean = False
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        capability = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        split_doc = {}
        if split_path and os.path.isfile(split_path):
            with open(split_path, encoding="utf-8") as sf:
                split_doc = json.load(sf)
        with open(os.path.join(self.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump({"run_id": self.run_id, "variant": variant, "seed": seed,
                       "local_steps": self.local_steps, "modality_alpha": self.modality_alpha,
                       "rgb_only_weight": self.rgb_only_weight, "prox_mu": self.prox_mu,
                       "git_hash": git_hash, "git_clean": git_clean,
                       "python": sys.version, "torch": torch.__version__,
                       "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                       "gpu": gpu, "compute_capability": capability,
                       "cfg_train": actual_cfg, "split_path": split_path,
                       "fingerprint": split_doc.get("fingerprint"),
                       "weights_fingerprint": split_doc.get("weights_fingerprint"),
                       "started_at": datetime.now(timezone.utc).isoformat(),
                       "phase_table": self.scheduler.describe(),
                       **({"personalized": {
                           "prime_sim_scalar": self.prime_sim_scalar,
                           "prime_outer_loops": self.prime_outer_loops,
                           "prime_pool_size": PRIME_POOL_SIZE,
                           "prime_top_k": PRIME_TOP_K,
                           "fedmsplit_prox_lambda": self.prox_mu}}
                          if variant in NEW_PERSONALIZED else {})},
                      f, ensure_ascii=False, indent=2)

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
            dispatched = self._dispatch(cid, client, src)
            extra = ({"prox_anchor": self.fmp_anchor}
                     if self.fmp_states is not None else {})
            update, stats = client.local_train(
                self.net, dispatched, plan, round_idx,
                local_steps=self.local_steps, device=self.device, **extra)
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
            if self.fmp_states is not None:
                agg_summary = self._aggregate_fedmsplit_p(updates)
                self._last_agg_stats = None
            else:
                aggregation_updates = updates
                if self.prime_local is not None:
                    # the pools are NOT FedAvg'd: strip them from the reduction
                    aggregation_updates = [self._without_prime(u) for u in updates]
                self.state, agg_stats = self.aggregator.aggregate(
                    self.state, aggregation_updates, round_idx=round_idx, phase=plan)
                agg_summary = agg_stats.group_summary()
                self._last_agg_stats = agg_stats
                if self.prime_local is not None:
                    self._after_aggregate_fedprime(updates, round_idx)

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
            if self._last_agg_stats is not None:
                with open(self.param_stats_path, "a", encoding="utf-8") as f:
                    for st in self._last_agg_stats.per_key:
                        f.write(json.dumps(_jsonable(st.__dict__), ensure_ascii=False) + "\n")
        if self.pers_info is not None:
            record["personalized"] = _jsonable(self.pers_info)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record

    # ------------------------------------------- FedMSplit-P / FED-PRIME

    def _dispatch(self, client_id, client, source):
        """What client `client_id` starts its local steps from."""
        if self.fmp_states is not None:
            out = OrderedDict(self.fmp_states[client_id])
            if client.aux_type == NO_AUX:
                # project-wide missing-modality contract: an RGB-only client
                # always sees the pretrained auxiliary tensors
                out.update(self.state.initial_aux)
            return out
        if self.prime_local is not None:
            out = source.view_for(client.aux_type)
            out.update(self.prime_local[client_id])
            return out
        return source.view_for(client.aux_type)

    def _without_prime(self, update):
        state = OrderedDict((k, v) for k, v in update.state.items()
                            if self.param_groups.get(k) != PRIME)
        mask = None if update.param_mask is None else OrderedDict(
            (k, v) for k, v in update.param_mask.items() if k in state)
        return type(update)(update.client_id, update.num_samples, update.aux_type,
                            state, mask, update.reliability, dict(update.extras))

    def _aggregate_fedmsplit_p(self, updates):
        """Similarity-weighted personalised update (see personalized.py)."""
        states = OrderedDict((u.client_id, u.state) for u in updates)
        blocks = OrderedDict(
            (g, [k for k in next(iter(states.values()))
                 if self.param_groups.get(k) == g]) for g in FEDMSPLIT_BLOCKS)
        provides = {u.client_id: {g: (g != AUX_SPECIFIC or u.aux_type != NO_AUX)
                                  for g in FEDMSPLIT_BLOCKS} for u in updates}
        weights = {u.client_id: float(u.num_samples) for u in updates}
        personal, gavg, info = fedmsplit_p_aggregate(states, blocks, provides, weights)
        for cid, tensors in personal.items():
            for k, v in tensors.items():
                self.fmp_states[cid][k] = v.detach().clone()
        for k, v in gavg.items():
            self.fmp_anchor[k] = v.detach().clone()
        self.pers_info = info
        return OrderedDict((b, dict(v)) for b, v in info.items())

    def _after_aggregate_fedprime(self, updates, round_idx):
        """Keep each client's intra pool / query; align the inter pool."""
        gkey = self.prime_global_key
        previous = self.state.shared[gkey]
        rows = OrderedDict()
        for u in updates:
            for k in self.prime_local[u.client_id]:
                self.prime_local[u.client_id][k] = u.state[k].detach().clone()
            used = u.extras["prime_global_used"].bool()
            if bool(used.any()):
                rows[u.client_id] = u.state[gkey].detach().cpu()[used]
        new_pool, info = align_global_prompts(
            rows, previous.detach().cpu(), outer_loops=self.prime_outer_loops,
            seed=(self.seed * 1_000_003 + round_idx) & 0x7FFFFFFF)
        self.state.shared[gkey] = new_pool.to(previous.dtype).to(previous.device)
        self.pers_info = OrderedDict(info)
        return info

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
                "run_id": self.run_id,
                "state": self.state.to_flat_dict()}
        if self.local_states is not None:
            # 2026-09-20: `local` never updates self.state, so persisting only
            # it silently discarded every client's trained private model.
            blob["local_states"] = {
                cid: state.to_flat_dict()
                for cid, state in self.local_states.items()
            }
        if self.fmp_states is not None:
            blob["fmp_states"] = self.fmp_states
            blob["fmp_anchor"] = self.fmp_anchor
        if self.prime_local is not None:
            blob["prime_local"] = self.prime_local
        torch.save(blob, path)
        return path

    def load_state(self, path):
        """Restore a state written by :meth:`save_state` for the same variant."""
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("variant") != self.variant:
            raise ValueError("checkpoint variant %r does not match %r" %
                             (blob.get("variant"), self.variant))
        self.state = GlobalState.from_flat_dict(blob["state"])
        if self.local_states is not None:
            self.local_states = {cid: GlobalState.from_flat_dict(s)
                                 for cid, s in blob["local_states"].items()}
        if self.fmp_states is not None:
            self.fmp_states = blob["fmp_states"]
            self.fmp_anchor = blob["fmp_anchor"]
        if self.prime_local is not None:
            self.prime_local = blob["prime_local"]
        return int(blob["round"])

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
        elif self.fmp_states is not None or self.prime_local is not None:
            # personalised: one checkpoint per SENSOR client (the evaluation
            # protocol averages the three sensor clients' scores)
            if client_id is not None:
                if client_id not in self.clients:
                    raise KeyError("unknown client: %s" % client_id)
                client_ids = [client_id]
            else:
                client_ids = [cid for cid, c in self.clients.items()
                              if c.aux_type != NO_AUX
                              and (aux_type is None or c.aux_type == aux_type)]
            targets = [(cid, self.clients[cid].aux_type, self.state)
                       for cid in client_ids]
        else:
            aux_types = ([aux_type] if aux_type else
                         (list(self.aux_types) or [NO_AUX]))
            targets = [(None, aux, self.state) for aux in aux_types]

        written = []
        for cid, aux, source in targets:
            sd = OrderedDict((k, v.detach().clone())
                             for k, v in self.frozen_reference.items())
            view = (self._dispatch(cid, self.clients[cid], source)
                    if (self.fmp_states is not None or self.prime_local is not None)
                    else source.view_for(aux))
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

    if args.variant == "fedprime":
        cfg.MODEL.BACKBONE.PRIME_POOL_SIZE = PRIME_POOL_SIZE
        cfg.MODEL.BACKBONE.PRIME_TOP_K = PRIME_TOP_K

    seed_everything(args.seed)
    net = build_viptrack(cfg, training=True).to(args.device)

    _, by_id = load_split(args.split)
    wanted = args.clients.split(",") if args.clients else list(by_id)
    prox_mu = args.prox_mu if args.prox_mu is not None else (
        0.01 if args.variant in ("fedprox", "fedmsplit_p") else 0.0)
    prime_sim = (args.prime_sim_scalar if args.prime_sim_scalar is not None
                 else PRIME_SIM_SCALAR) if args.variant == "fedprime" else 0.0
    clients = [FederatedClient(by_id[cid], cfg, settings, base_seed=args.seed,
                               prox_mu=prox_mu, prime_sim_scalar=prime_sim,
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
    p.add_argument("--prime-sim-scalar", type=float, default=None,
                   help="fedprime only: weight of the prompt-relevance term "
                        "(default 0.01, the official default)")
    p.add_argument("--prime-outer-loops", type=int, default=None,
                   help="fedprime only: server alignment loops (default 5)")
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
    if args.prox_mu is not None and args.variant not in ("fedprox", "fedmsplit_p"):
        raise ValueError("--prox-mu is only valid with --variant fedprox / fedmsplit_p")
    if args.variant != "fedprime" and (args.prime_sim_scalar is not None
                                       or args.prime_outer_loops is not None):
        raise ValueError("--prime-* options are only valid with --variant fedprime")

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
        prime_sim_scalar=args.prime_sim_scalar,
        prime_outer_loops=args.prime_outer_loops,
        split_path=args.split,
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
