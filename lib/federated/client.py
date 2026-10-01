"""One federated client: local training on its own sequences, then an upload.

Design decisions, and why
-------------------------
1. LOCAL TRAINING USES THE OFFICIAL `ViPTActor`, UNCHANGED.
   The loss (GIoU + L1 + focal) and the forward path are exactly the ones the
   reproduction ran. If a federated variant scores differently, that difference
   must come from the federated layer, not from a re-implemented loss. Anything
   else makes the ablation unattributable.

2. MODALITY MASKING IS DONE ON THE DATA, NOT IN THE MODEL.
   ViPT packs the pair as 6 channels: `x[:, :3]` RGB, `x[:, 3:]` auxiliary. A
   client with no auxiliary sensor -- or a sample that happens to be missing it
   -- gets `x[:, 3:]` zeroed. That is precisely what
   `FedViPTWrapper(aux_fill="zeros")` means, applied where it costs nothing.
   So the wrapper here is used only for parameter plumbing
   (`load_trainable_state` / `trainable_state` / `param_mask_for`).

3. THE OPTIMIZER IS REBUILT EVERY ROUND BY DEFAULT.
   Carrying Adam moments across rounds means a client's update depends on how
   many rounds it happened to be sampled in, which is a confound we would then
   have to control for. `optimizer_state="persist"` keeps them for the ablation
   that asks the question directly.

4. SEEDING IS PER (ROUND, CLIENT).
   A single global seed makes every client draw the same batches; seeding per
   client alone makes round 2 repeat round 1. The derived seed keeps a whole run
   reproducible while still giving each client-round its own sample stream.

5. WHAT GOES UP IS ONLY WHAT THIS PHASE TRAINED.
   `ClientUpdate.state` carries the phase's trainable groups only, and
   `param_mask` marks an RGB-only client's aux_specific tensors False. The
   aggregator needs "did not train this" and "trained it to zero" to be
   different things (rule R4), and this is where that distinction is created.
"""

from __future__ import annotations

import os
import random
import sys
from collections import OrderedDict

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import lib.train.data.transforms as tfm  # noqa: E402
from lib.train.actors import ViPTActor  # noqa: E402
from lib.train.data import LTRLoader, opencv_loader, processing, sampler  # noqa: E402
from lib.federated.aggregator import NO_AUX, ClientUpdate  # noqa: E402
from lib.federated.client_data import build_client_dataset  # noqa: E402
from lib.federated.fed_wrapper import FedViPTWrapper  # noqa: E402
from lib.federated.param_groups import TRAINABLE_GROUPS, is_modality_sensitive  # noqa: E402

RGB_CHANNELS = 3


def derive_seed(base_seed: int, round_idx: int, client_id: str) -> int:
    """Stable per (run, round, client). Uses the id's characters so that adding
    a client does not shift every other client's stream."""
    h = 0
    for ch in client_id:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return (base_seed * 1_000_003 + round_idx * 9_176 + h) & 0x7FFFFFFF


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mask_aux_(batch_images: torch.Tensor, keep: torch.Tensor):
    """Zero the auxiliary half for the samples where `keep` is False, in place.

    `batch_images` is (N_frames, batch, 6, H, W) as LTRLoader stacks it
    (stack_dim=1). `keep` is a bool tensor of length `batch`.
    """
    if batch_images.shape[2] != 2 * RGB_CHANNELS:
        raise ValueError(
            "expected 6 channels, got %d; ViPT packs RGB in x[:, :3] and the "
            "auxiliary modality in x[:, 3:]" % batch_images.shape[2])
    drop = ~keep
    if drop.any():
        batch_images[:, drop, RGB_CHANNELS:] = 0.0


class FederatedClient:
    """Holds one client's data and runs its local steps.

    The model is NOT owned here: the server passes the shared network in, so a
    single-process simulation keeps one copy of the 92.5 M frozen backbone
    instead of one per client. Only the 0.84 M trainable tensors are swapped in
    and out per round, which is the whole point of prompt tuning.
    """

    def __init__(self, client_cfg, cfg, settings, base_seed: int = 42,
                 optimizer_state: str = "reset", verbose: bool = False,
                 prox_mu: float = 0.0):
        if optimizer_state not in ("reset", "persist"):
            raise ValueError("optimizer_state must be 'reset' or 'persist'")
        self.cfg = cfg
        self.settings = settings
        self.base_seed = int(base_seed)
        self.optimizer_state = optimizer_state
        self.verbose = verbose
        self.prox_mu = float(prox_mu)

        self.client_id = client_cfg["client_id"]
        self.dataset_name = client_cfg["dataset"]
        self.aux_type = client_cfg.get("aux_type", NO_AUX)
        self.num_samples = int(client_cfg["num_samples"])
        self.missing_prob = float(client_cfg.get("missing_prob", 0.0))
        self.sequence_ids = list(client_cfg["sequence_ids"])

        self._dataset = None
        self._opt_state = None

    # ------------------------------------------------------------------ data

    @property
    def dataset(self):
        """Built on first use: constructing 12 datasets up front costs minutes
        and most rounds touch only a few clients."""
        if self._dataset is None:
            self._dataset = build_client_dataset(
                {"client_id": self.client_id, "dataset": self.dataset_name,
                 "sequence_ids": self.sequence_ids,
                 "num_samples": self.num_samples},
                verbose=self.verbose)
        return self._dataset

    def _build_loader(self, local_steps: int, seed: int):
        """A loader that yields exactly `local_steps` batches.

        Mirrors lib/train/base_functions.py:build_dataloaders so the augmentation
        and the sampling mode match the reproduction run; the only differences
        are the dataset (this client's sequences) and samples_per_epoch (the
        local budget instead of 60000).
        """
        cfg, settings = self.cfg, self.settings

        transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                        tfm.RandomHorizontalFlip(probability=0.5))
        transform_train = tfm.Transform(
            tfm.ToTensorAndJitter(0.2),
            tfm.RandomHorizontalFlip_Norm(probability=0.5),
            tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

        data_processing = processing.ViPTProcessing(
            search_area_factor=settings.search_area_factor,
            output_sz=settings.output_sz,
            center_jitter_factor=settings.center_jitter_factor,
            scale_jitter_factor=settings.scale_jitter_factor,
            mode='sequence',
            transform=transform_train,
            joint_transform=transform_joint,
            settings=settings)

        batch_size = cfg.TRAIN.BATCH_SIZE
        ds = sampler.TrackingSampler(
            datasets=[self.dataset],
            p_datasets=[1],
            samples_per_epoch=local_steps * batch_size,
            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
            num_search_frames=settings.num_search,
            num_template_frames=settings.num_template,
            processing=data_processing,
            frame_sample_mode=getattr(cfg.DATA, "SAMPLER_MODE", "causal"),
            train_cls=getattr(cfg.TRAIN, "TRAIN_CLS", False))

        # worker_init keeps the sample stream tied to the derived seed even with
        # NUM_WORKER > 0, where each worker would otherwise reseed from entropy.
        def _init_worker(worker_id):
            seed_everything(seed + worker_id + 1)

        return LTRLoader('train', ds, training=True, batch_size=batch_size,
                         shuffle=True, num_workers=cfg.TRAIN.NUM_WORKER,
                         drop_last=True, stack_dim=1,
                         worker_init_fn=_init_worker)

    # --------------------------------------------------------------- masking

    def sample_keep_mask(self, batch_size: int, rng: random.Random):
        """Per-sample availability of the auxiliary modality.

        An RGB-only client never has it. A sensor client has it except on the
        `missing_prob` fraction of samples, which is how spec 3.1's
        sample-level missingness is realised.
        """
        if self.aux_type == NO_AUX:
            return torch.zeros(batch_size, dtype=torch.bool)
        if self.missing_prob <= 0:
            return torch.ones(batch_size, dtype=torch.bool)
        return torch.tensor([rng.random() >= self.missing_prob
                             for _ in range(batch_size)], dtype=torch.bool)

    # -------------------------------------------------------------- training

    def local_train(self, net, dispatched_state, plan, round_idx, local_steps,
                    device="cuda", prox_mu=None, prox_anchor=None,
                    seed_offset=0):
        """Run this client's local steps and return what it uploads.

        `net` is the shared network; `dispatched_state` is
        `GlobalState.view_for(self.aux_type)`. Returns (ClientUpdate, stats).
        """
        # A Ditto personal pass must not perturb FedAvg's reproducible stream.
        # The default (zero) deliberately preserves every existing variant.
        seed = derive_seed(self.base_seed, round_idx, self.client_id) + int(seed_offset)
        seed_everything(seed)
        rng = random.Random(seed ^ 0x5F3759DF)

        wrapper = FedViPTWrapper(net, aux_fill="zeros")

        # 1. apply what the server sent -- raises if it ever contains a frozen key
        written = wrapper.load_trainable_state(dispatched_state, strict_groups=True)

        # 2. open only this phase's groups
        open_groups = set(plan.trainable_groups)
        n_open = 0
        for name, p in net.named_parameters():
            g = wrapper.param_groups.get(name)
            if g in TRAINABLE_GROUPS and g in open_groups:
                if self.aux_type == NO_AUX and g == "aux_specific":
                    p.requires_grad_(False)
                    continue
                p.requires_grad_(True)
                n_open += 1
            else:
                p.requires_grad_(False)
        if n_open == 0:
            return self._empty_update(wrapper, plan, round_idx)

        # 3. optimizer over the open tensors only
        trainable = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.cfg.TRAIN.LR,
                                      weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        if self.optimizer_state == "persist" and self._opt_state is not None:
            try:
                optimizer.load_state_dict(self._opt_state)
            except ValueError:
                # the open parameter set changed with the phase -- moments from
                # the previous shape cannot be reused; start clean and say so.
                if self.verbose:
                    print("  [%s] optimizer state dropped: parameter set changed"
                          % self.client_id)

        objective, loss_weight = _build_objective(self.cfg)
        actor = ViPTActor(net=net, objective=objective, loss_weight=loss_weight,
                          settings=self.settings, cfg=self.cfg)
        net.train()
        if getattr(self.settings, "fix_bn", False):
            actor.fix_bns()

        loader = self._build_loader(local_steps, seed)
        accum = max(1, int(getattr(self.settings, "accum_iter", 1)))

        totals = OrderedDict()
        n_batches = 0
        n_aux_present = 0
        n_samples_seen = 0
        optimizer.zero_grad()
        optimizer_steps = 0

        for i, data in enumerate(loader, 1):
            data = data.to(device)
            # spec 4.2's epoch-dependent candidate elimination: rounds stand in
            # for epochs so the schedule still ramps over a federated run.
            data['epoch'] = round_idx + 1
            data['settings'] = self.settings

            bs = data['template_images'].shape[1]
            keep = self.sample_keep_mask(bs, rng)
            mask_aux_(data['template_images'], keep)
            mask_aux_(data['search_images'], keep)
            n_aux_present += int(keep.sum())
            n_samples_seen += bs

            loss, status = actor(data)
            prox = torch.zeros((), device=loss.device)
            active_prox_mu = self.prox_mu if prox_mu is None else float(prox_mu)
            anchor = dispatched_state if prox_anchor is None else prox_anchor
            if active_prox_mu:
                for name, p in net.named_parameters():
                    if p.requires_grad and name in anchor:
                        prox = prox + (p - anchor[name].to(p)).pow(2).sum()
                prox = 0.5 * active_prox_mu * prox
                loss = loss + prox
            (loss / accum).backward()
            if (i % accum == 0) or (i == local_steps):
                if self.settings.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable,
                                                   self.settings.grad_clip_norm)
                optimizer.step()
                optimizer_steps += 1
                optimizer.zero_grad()

            for k, v in status.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            totals["Loss/prox"] = totals.get("Loss/prox", 0.0) + float(prox.detach())
            n_batches += 1

        if self.optimizer_state == "persist":
            self._opt_state = optimizer.state_dict()

        # 4. build the upload: this phase's groups only
        full = wrapper.trainable_state(detach=True)
        state = OrderedDict((k, v) for k, v in full.items()
                            if wrapper.param_groups[k] in open_groups)

        aux_present = self.aux_type != NO_AUX and n_aux_present > 0
        mask = wrapper.param_mask_for(self.aux_type, aux_present=aux_present)
        param_mask = OrderedDict((k, mask[k]) for k in state if k in mask)

        update = ClientUpdate(
            client_id=self.client_id,
            num_samples=self.num_samples,
            aux_type=self.aux_type,
            state=state,
            param_mask=param_mask,
            reliability=1.0)

        stats = OrderedDict(
            client_id=self.client_id,
            dataset=self.dataset_name,
            aux_type=self.aux_type,
            round=round_idx,
            phase=plan.phase_name,
            seed=seed,
            batches=n_batches,
            dispatched_keys=len(written),
            uploaded_keys=len(state),
            masked_false=sum(1 for v in param_mask.values() if not v),
            aux_present_frac=(n_aux_present / n_samples_seen) if n_samples_seen else 0.0,
            optimizer_steps=optimizer_steps,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            accum=accum, batch_size=int(self.cfg.TRAIN.BATCH_SIZE),
        )
        for k, v in totals.items():
            stats["mean_" + k] = v / max(n_batches, 1)
        return update, stats

    def _empty_update(self, wrapper, plan, round_idx):
        """Return a no-op update for a phase with no usable RGB-only groups."""
        state = OrderedDict()
        mask = wrapper.param_mask_for(self.aux_type, aux_present=False)
        return ClientUpdate(self.client_id, self.num_samples, self.aux_type,
                            state, OrderedDict(), 1.0), OrderedDict(
                                client_id=self.client_id, round=round_idx,
                                phase=plan.phase_name, batches=0,
                                skipped="no trainable parameters for client")

    def __repr__(self):
        return ("FederatedClient(%s, %s, aux=%s, seqs=%d, samples=%d)"
                % (self.client_id, self.dataset_name, self.aux_type,
                   len(self.sequence_ids), self.num_samples))


def _build_objective(cfg):
    """The official loss set, copied from lib/train/train_script.py:71-73.

    `cls` is unused while TRAIN_CLS is False, but it is in the official dict and
    costs nothing to carry -- leaving it out would be a silent divergence the
    day someone turns TRAIN_CLS on.
    """
    from torch.nn import BCEWithLogitsLoss
    from torch.nn.functional import l1_loss

    from lib.utils.box_ops import giou_loss
    from lib.utils.focal_loss import FocalLoss

    objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': FocalLoss(),
                 'cls': BCEWithLogitsLoss()}
    loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT,
                   'focal': 1., 'cls': 1.0}
    return objective, loss_weight
