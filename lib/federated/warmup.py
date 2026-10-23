#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Progressive warm-up scheduler (spec section 8).

The second of the paper's two mechanisms. The aggregator decides WHO averages
with WHOM; this decides WHEN each thing is allowed to move at all.

-------------------------------------------------------------------------
The idea in one paragraph
-------------------------------------------------------------------------
79% of a ViPT client's upload is AUX_SPECIFIC, and those tensors mean different
physical things on a thermal client than on a depth client. Opening everything
from round 0 means the very first aggregation has to reconcile "temperature ->
token" with "metres -> token". Warm-up avoids that by ordering the openings:
first train only what every client agrees on (the RGB half of the prompt),
then each sensor's own path while keeping it inside its own modality bank,
then fusion, then everything. Nothing is ever averaged across modalities until
the part that is genuinely shared has stabilised.

-------------------------------------------------------------------------
Mapping spec's phase table onto ViPT
-------------------------------------------------------------------------
spec 8 is written against SDSTrack's names (adapter_thermal, fusion_head,
shared_backbone_high). ViPT has none of those. The mapping, justified by what
each tensor consumes (see param_groups.py):

    spec "RGB path, head"          -> RGB_SHARED + NORM
        ViPT's head is frozen and never trains, and every trainable tensor
        lives in a prompt block, so "the RGB path" is exactly conv0_0 + fovea,
        the two that only ever see x[:, :3].
    spec "adapter_{thermal,...}"   -> AUX_SPECIFIC
        patch_embed_prompt + conv0_1, the tensors that consume the aux modality.
    spec "fusion_head"             -> FUSION
        conv1x1, which projects the summed branches back to embed_dim.
    spec "shared_backbone_high"    -> nothing
        ViPT freezes the whole ViT. Unfreezing high blocks would be a method
        change, not a schedule change, so it is deliberately NOT wired here.

Phase boundaries are data, not code (design note 8: "phase boundaries must be configurable, not hard-coded").
DEFAULT_PHASES is a starting point; anything can be overridden from yaml.

-------------------------------------------------------------------------
Degradation
-------------------------------------------------------------------------
`WarmupScheduler.no_warmup(num_rounds)` gives a single always-everything-open
phase. That is the "Ours without progressive warm-up" ablation row in design note 14.1,
and it is also how you tell "the schedule is wrong" apart from "the aggregation
is wrong" when a run misbehaves.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .param_groups import AUX_SPECIFIC, FUSION, NORM, RGB_SHARED, TRAINABLE_GROUPS

ALL_GROUPS = tuple(TRAINABLE_GROUPS)

# ---------------------------------------------------------------- schedules

_LINEAR = re.compile(r"^\s*linear\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)\s*$")
_COSINE = re.compile(r"^\s*cosine\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)\s*$")


class Schedule:
    """A value that may vary across a phase.

    Accepts a plain number, or spec 8.1's string form "linear(0.0, 0.5)".
    `progress` is 0.0 at the first round of the phase and 1.0 at the last, so a
    one-round phase is well defined (progress 0.0) instead of dividing by zero.
    """

    def __init__(self, spec):
        self.spec = spec
        if isinstance(spec, (int, float)):
            self.kind, self.a, self.b = "const", float(spec), float(spec)
            return
        if not isinstance(spec, str):
            raise TypeError(f"unsupported schedule {spec!r}")
        m = _LINEAR.match(spec)
        if m:
            self.kind, self.a, self.b = "linear", float(m.group(1)), float(m.group(2))
            return
        m = _COSINE.match(spec)
        if m:
            self.kind, self.a, self.b = "cosine", float(m.group(1)), float(m.group(2))
            return
        try:
            v = float(spec)
        except ValueError:
            raise ValueError(
                f"cannot parse schedule {spec!r}; use a number, "
                f"'linear(a, b)' or 'cosine(a, b)'"
            )
        self.kind, self.a, self.b = "const", v, v

    def at(self, progress: float) -> float:
        p = min(1.0, max(0.0, float(progress)))
        if self.kind == "const":
            return self.a
        if self.kind == "linear":
            return self.a + (self.b - self.a) * p
        # cosine: slow at both ends, useful for easing a loss term in
        return self.a + (self.b - self.a) * (1.0 - math.cos(math.pi * p)) / 2.0

    def __repr__(self):
        if self.kind == "const":
            return f"{self.a:g}"
        return f"{self.kind}({self.a:g},{self.b:g})"


# ---------------------------------------------------------------- phase

@dataclass
class Phase:
    """One stage of the schedule.

    Field names match what Aggregator reads via getattr (trainable_groups,
    downweight_missing_aux, missing_client_weight), so a Phase can be handed to
    Aggregator.aggregate() directly.
    """
    name: str
    start_round: int
    end_round: int                      # exclusive
    trainable_groups: Tuple[str, ...]
    modality_aware: bool = True
    loss_weights: Dict[str, Schedule] = field(default_factory=dict)
    downweight_missing_aux: bool = False
    missing_client_weight: float = 1.0
    clients_per_round: Optional[float] = None
    note: str = ""

    def __post_init__(self):
        if self.end_round <= self.start_round:
            raise ValueError(
                f"phase {self.name!r}: end_round {self.end_round} must be greater "
                f"than start_round {self.start_round}"
            )
        bad = [g for g in self.trainable_groups if g not in ALL_GROUPS]
        if bad:
            raise ValueError(
                f"phase {self.name!r}: unknown group(s) {bad}; "
                f"valid groups are {list(ALL_GROUPS)}"
            )
        self.trainable_groups = tuple(self.trainable_groups)
        self.loss_weights = {
            k: v if isinstance(v, Schedule) else Schedule(v)
            for k, v in self.loss_weights.items()
        }

    def contains(self, round_idx: int) -> bool:
        return self.start_round <= round_idx < self.end_round

    def progress(self, round_idx: int) -> float:
        span = self.end_round - self.start_round
        return 0.0 if span <= 1 else (round_idx - self.start_round) / (span - 1)


@dataclass
class RoundPlan:
    """Everything the server needs for one round, fully resolved.

    This is what gets written to round_metrics.jsonl, so spec 16's
    "warm-up takes effect: the log shows every round's phase, trainable parameter groups, loss weights
    and aggregation strategy" is satisfied by dumping this object, not by re-deriving it.
    """
    round_idx: int
    phase_name: str
    phase_index: int
    progress: float
    trainable_groups: Tuple[str, ...]
    frozen_groups: Tuple[str, ...]
    modality_aware: bool
    loss_weights: Dict[str, float]
    downweight_missing_aux: bool
    missing_client_weight: float
    clients_per_round: Optional[float]

    def as_dict(self):
        return OrderedDict(
            round=self.round_idx,
            phase=self.phase_name,
            phase_index=self.phase_index,
            progress=round(self.progress, 4),
            trainable=list(self.trainable_groups),
            frozen=list(self.frozen_groups),
            modality_aware=self.modality_aware,
            loss_weights={k: round(v, 6) for k, v in self.loss_weights.items()},
            clients_per_round=self.clients_per_round,
        )

    def one_line(self):
        w = " ".join(f"{k}={v:.3g}" for k, v in sorted(self.loss_weights.items()))
        return (f"[round {self.round_idx:>4}] phase={self.phase_name:<16}"
                f"p={self.progress:4.2f} "
                f"train={'+'.join(self.trainable_groups):<40}"
                f"mod_aware={int(self.modality_aware)} {w}")


# ---------------------------------------------------------------- defaults

def default_phases(num_rounds: int = 120) -> List[Phase]:
    """spec section 8's table, scaled to `num_rounds` and mapped onto ViPT.

    The fractions (1/6, 1/4, 1/4, rest) follow spec 8.1's example
    (20 / 30 / ... out of 120) rather than being invented here.

    Four phases need at least four rounds. Below that the `max(prev + 1, ...)`
    guards below push a boundary past `num_rounds` and Phase's own validation
    rejects it -- correct, but the message then talks about internal round
    arithmetic instead of what the caller actually did wrong. Say it here.
    Short debugging runs want `WarmupScheduler.no_warmup(n)`: one phase, any
    n >= 1.
    """
    if num_rounds < 4:
        raise ValueError(
            "default_phases needs num_rounds >= 4 (one round per phase), got "
            f"{num_rounds}. For a short run use "
            f"WarmupScheduler.no_warmup({num_rounds}), or --variant modality."
        )
    r1 = max(1, round(num_rounds * 1 / 6))            # 0   .. 20   of 120
    r2 = max(r1 + 1, round(num_rounds * 5 / 12))      # 20  .. 50
    r3 = max(r2 + 1, round(num_rounds * 2 / 3))       # 50  .. 80
    return [
        Phase(
            name="rgb_common", start_round=0, end_round=r1,
            trainable_groups=(RGB_SHARED, NORM),
            modality_aware=False,           # nothing modality-specific is open
            note="shared parameters first train; this is parameter-group warm-up, not RGB-only input",
        ),
        Phase(
            name="adapter_warmup", start_round=r1, end_round=r2,
            trainable_groups=(RGB_SHARED, NORM, AUX_SPECIFIC),
            modality_aware=True,            # aux tensors stay in their own bank
            note="each sensor learns its own mapping, never averaged across modalities",
        ),
        Phase(
            name="fusion_calib", start_round=r2, end_round=r3,
            trainable_groups=(RGB_SHARED, NORM, AUX_SPECIFIC, FUSION),
            modality_aware=True,
            note="open the fusion projection and start pulling distributions together",
        ),
        Phase(
            name="full", start_round=r3, end_round=num_rounds,
            trainable_groups=ALL_GROUPS,
            modality_aware=True,
            note="main method; no implicit RGB-only client downweight",
        ),
    ]


# ---------------------------------------------------------------- scheduler

class WarmupScheduler:
    def __init__(self, phases: Sequence[Phase]):
        if not phases:
            raise ValueError("need at least one phase")
        self.phases = sorted(phases, key=lambda p: p.start_round)
        self._validate()

    def _validate(self):
        """Contiguous, non-overlapping, starting at 0.

        A gap would leave some round with no phase at all, and the server would
        silently fall back to whatever phase_at returns -- the kind of thing
        that shows up only as a strange dip in a curve 40 rounds later.
        """
        if self.phases[0].start_round != 0:
            raise ValueError(
                f"first phase must start at round 0, got {self.phases[0].start_round}"
            )
        for a, b in zip(self.phases, self.phases[1:]):
            if b.start_round != a.end_round:
                raise ValueError(
                    f"phases must be contiguous: {a.name!r} ends at {a.end_round} "
                    f"but {b.name!r} starts at {b.start_round}"
                )

    # ------------------------------------------------ construction

    @classmethod
    def default(cls, num_rounds: int = 120):
        return cls(default_phases(num_rounds))

    @classmethod
    def no_warmup(cls, num_rounds: int = 120, modality_aware: bool = True):
        """Single phase with everything open -- the ablation baseline."""
        return cls([Phase(
            name="no_warmup", start_round=0, end_round=num_rounds,
            trainable_groups=ALL_GROUPS, modality_aware=modality_aware,
            note="ablation: progressive warm-up disabled",
        )])

    @classmethod
    def from_config(cls, conf, num_rounds: Optional[int] = None):
        """Parse a phase table, for inspection only; the server uses code phases.

        Consistency/calibration/auxiliary losses are not implemented. A config
        that requests any non-zero value for them therefore fails loudly.

        warmup:
          phases:
            - name: rgb_common
              rounds: [0, 20]          # or "[0, 20)" as a string
              trainable: [rgb_shared, norm]
              modality_aware: false
              aux_loss_weight: 0.0
        """
        if conf is None:
            return cls.default(num_rounds or 120)
        raw = conf.get("phases") if isinstance(conf, dict) else conf
        if not raw:
            return cls.default(num_rounds or 120)

        phases = []
        for i, p in enumerate(raw):
            lo, hi = _parse_rounds(p.get("rounds"), i, raw, num_rounds)
            weights = dict(p.get("loss_weights") or {})
            # spec 8.1 uses the flat key aux_loss_weight; accept both.
            if "aux_loss_weight" in p:
                weights["aux"] = p["aux_loss_weight"]
            for name in ("aux", "consistency", "calib"):
                value = weights.get(name, 0.0)
                if isinstance(value, str) or float(value) != 0.0:
                    raise ValueError("unimplemented loss %r requested" % name)
            weights.setdefault("giou", 2.0)
            weights.setdefault("l1", 5.0)
            weights.setdefault("focal", 1.0)
            phases.append(Phase(
                name=p.get("name", f"phase{i}"),
                start_round=lo, end_round=hi,
                trainable_groups=tuple(p.get("trainable") or ALL_GROUPS),
                modality_aware=bool(p.get("modality_aware", True)),
                loss_weights=weights,
                downweight_missing_aux=bool(p.get("downweight_missing_aux", False)),
                missing_client_weight=float(p.get("missing_client_weight", 1.0)),
                clients_per_round=p.get("clients_per_round"),
                note=p.get("note", ""),
            ))
        return cls(phases)

    # ------------------------------------------------ query

    @property
    def num_rounds(self) -> int:
        return self.phases[-1].end_round

    def phase_at(self, round_idx: int) -> Phase:
        for p in self.phases:
            if p.contains(round_idx):
                return p
        # Past the end: stay in the last phase rather than crashing a long run,
        # but say so -- a silent extension would misreport the schedule.
        last = self.phases[-1]
        if round_idx >= last.end_round:
            return last
        raise IndexError(f"no phase covers round {round_idx}")

    def resolve(self, round_idx: int) -> RoundPlan:
        p = self.phase_at(round_idx)
        prog = p.progress(round_idx)
        return RoundPlan(
            round_idx=round_idx,
            phase_name=p.name,
            phase_index=self.phases.index(p),
            progress=prog,
            trainable_groups=p.trainable_groups,
            frozen_groups=tuple(g for g in ALL_GROUPS if g not in p.trainable_groups),
            modality_aware=p.modality_aware,
            loss_weights={},
            downweight_missing_aux=p.downweight_missing_aux,
            missing_client_weight=p.missing_client_weight,
            clients_per_round=p.clients_per_round,
        )

    # ------------------------------------------------ reporting

    def describe(self) -> str:
        lines = [f"{'phase':<16}{'rounds':<14}{'trainable':<44}"
                 f"{'mod_aware':<11}loss weights"]
        for p in self.phases:
            w = "(client objective)"
            lines.append(
                f"{p.name:<16}[{p.start_round},{p.end_round})".ljust(30)
                + f"{'+'.join(p.trainable_groups):<44}"
                + f"{str(p.modality_aware):<11}{w}"
            )
        return "\n".join(lines)


def _parse_rounds(spec, i, raw, num_rounds):
    """Accept [0, 20], (0, 20), '[0, 20)' or None (chain from the previous)."""
    if spec is None:
        lo = 0 if i == 0 else _parse_rounds(raw[i - 1].get("rounds"), i - 1, raw, num_rounds)[1]
        hi = num_rounds or (lo + 1)
        return lo, hi
    if isinstance(spec, str):
        nums = re.findall(r"-?\d+", spec)
        if len(nums) != 2:
            raise ValueError(f"cannot parse rounds {spec!r}")
        return int(nums[0]), int(nums[1])
    lo, hi = spec
    return int(lo), int(hi)
