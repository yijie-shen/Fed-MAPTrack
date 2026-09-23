#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mask-aware, modality-aware aggregation (spec section 9).

This is one of the two places the paper's contribution actually lives (the
other is the progressive warm-up scheduler), so the rules are spelled out
rather than implied.

-------------------------------------------------------------------------
The problem
-------------------------------------------------------------------------
79% of what a ViPT client uploads is AUX_SPECIFIC (see param_groups.py):
patch_embed_prompt and conv0_1, the tensors that consume the auxiliary
modality. A thermal client's patch_embed_prompt learns "temperature -> token";
a depth client's learns "metres -> token". Averaging them with FedAvg is
averaging two different unit systems. That is the whole difficulty.

-------------------------------------------------------------------------
What this module does about it
-------------------------------------------------------------------------
The server keeps modality-sensitive tensors in PER-MODALITY BANKS and shared
tensors in one global bank:

    GlobalState.shared        {key: tensor}                 rgb_shared/fusion/norm
    GlobalState.per_modality  {aux_type: {key: tensor}}     aux_specific

A client with aux_type='thermal' is sent shared + per_modality['thermal'],
trains, and its aux tensors only ever average with other thermal clients.
RGB-path tensors still average across everyone -- which is exactly what makes
spec's Phase 1 ("RGB common warm-up") meaningful: in Phase 1 every client,
whatever sensor it holds, contributes to the same shared tensors.

-------------------------------------------------------------------------
Rules implemented (spec 9.1 / 9.2), each one testable
-------------------------------------------------------------------------
 R1  shared groups      -> weighted mean over every client that PROVIDED the key
 R2  aux_specific       -> weighted mean over clients whose aux_type matches
                           AND whose param_mask[key] is True
 R3  aux_type == 'none' -> contributes to shared only, never to any aux bank
 R4  nobody provided a key this round -> KEEP the previous value.
                           "Missing is skip, not zero." Overwriting a bank with
                           zeros (or with an all-client average) would dilute an
                           adapter that simply had no data this round.
 R5  weights            -> w_i = num_samples_i
                           * clamp(reliability_i, 0.2, 1.5)      if enabled
                           * missing_client_weight               if aux_type=='none'
 R6  degradation        -> modality_aware=False collapses R2/R3 into R1, i.e.
                           plain FedAvg. Required by the project's hard
                           constraint that every new mechanism can be switched
                           off from config to isolate problems.

Everything is recorded in AggregationStats so param_group_stats.jsonl
(spec section 15) can be written without re-deriving anything.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .param_groups import (
    TRAINABLE_GROUPS,
    build_param_groups,
    is_modality_sensitive,
    normalize_key,
)

NO_AUX = "none"


# ---------------------------------------------------------------- data types

@dataclass
class ClientUpdate:
    """What one client sends back (spec section 7.1, trimmed to what the
    aggregator actually needs)."""
    client_id: str
    num_samples: int
    aux_type: str                      # 'thermal' | 'depth' | 'event' | 'none'
    state: Dict[str, "Tensor"]         # absolute values of the trainable tensors
    param_mask: Optional[Dict[str, bool]] = None   # None == everything in state is valid
    reliability: float = 1.0

    def provides(self, key: str) -> bool:
        """Did this client actually train and upload a usable value for `key`?

        A key present in `state` but masked False is NOT provided -- that is the
        difference between "I have no thermal data this round" and "here is a
        zero tensor", which R4 exists to keep apart.
        """
        if key not in self.state:
            return False
        if self.param_mask is None:
            return True
        return bool(self.param_mask.get(key, True))


@dataclass
class GlobalState:
    """Server-side model state, split by how it may be averaged."""
    shared: Dict[str, "Tensor"] = field(default_factory=OrderedDict)
    per_modality: Dict[str, Dict[str, "Tensor"]] = field(default_factory=dict)
    initial_aux: Dict[str, "Tensor"] = field(default_factory=OrderedDict)

    def view_for(self, aux_type: str) -> Dict[str, "Tensor"]:
        """Exactly what gets dispatched to a client holding `aux_type`."""
        out = OrderedDict(self.shared)
        out.update(self.per_modality.get(aux_type, {}))
        if aux_type == NO_AUX:
            for k, v in self.initial_aux.items():
                out[k] = _clone(v)
        return out

    def ensure_modality(self, aux_type: str, template: Dict[str, "Tensor"]):
        """Create a bank for a modality seen for the first time, cloned from
        `template` so it starts from the pretrained initialisation rather than
        from zeros."""
        if aux_type == NO_AUX or aux_type in self.per_modality:
            return
        self.per_modality[aux_type] = OrderedDict(
            (k, _clone(v)) for k, v in template.items()
        )

    def to_flat_dict(self) -> Dict[str, "Tensor"]:
        """Flatten for torch.save. Modality banks are namespaced '@<aux>/<key>'."""
        flat = OrderedDict(self.shared)
        for k, v in self.initial_aux.items():
            flat[f"@__initial_aux__/{k}"] = v
        for aux, bank in self.per_modality.items():
            for k, v in bank.items():
                flat[f"@{aux}/{k}"] = v
        return flat

    @classmethod
    def from_flat_dict(cls, flat: Dict[str, "Tensor"]) -> "GlobalState":
        shared, per_mod = OrderedDict(), {}
        for k, v in flat.items():
            if k.startswith("@") and "/" in k:
                aux, real = k[1:].split("/", 1)
                if aux == "__initial_aux__":
                    continue
                per_mod.setdefault(aux, OrderedDict())[real] = v
            else:
                shared[k] = v
        initial = OrderedDict((k[1:].split("/", 1)[1], v)
                              for k, v in flat.items()
                              if k.startswith("@__initial_aux__/"))
        return cls(shared=shared, per_modality=per_mod, initial_aux=initial)


@dataclass
class KeyStat:
    """Per-key record of what happened, for param_group_stats.jsonl."""
    key: str
    group: str
    bank: str                     # 'shared' or the aux_type
    n_contributors: int
    contributors: List[str]
    total_weight: float
    skipped: List[str]            # "client_id:reason"
    kept_previous: bool = False


@dataclass
class AggregationStats:
    round_idx: int
    per_key: List[KeyStat] = field(default_factory=list)

    def group_summary(self):
        """{group: {tensors, kept_previous, mean_contributors}} for quick logs."""
        agg = defaultdict(lambda: {"tensors": 0, "kept_previous": 0, "contrib_sum": 0})
        for st in self.per_key:
            a = agg[st.group]
            a["tensors"] += 1
            a["kept_previous"] += int(st.kept_previous)
            a["contrib_sum"] += st.n_contributors
        out = OrderedDict()
        for g, a in agg.items():
            out[g] = {
                "tensors": a["tensors"],
                "kept_previous": a["kept_previous"],
                "mean_contributors": (a["contrib_sum"] / a["tensors"]) if a["tensors"] else 0.0,
            }
        return out


# ---------------------------------------------------------------- helpers

def _clone(t):
    return t.clone() if hasattr(t, "clone") else type(t)(t)


def _weighted_mean(pairs):
    """pairs: [(weight, tensor)] -> sum(w*t)/sum(w). Weights are floats."""
    total_w = float(sum(w for w, _ in pairs))
    if total_w <= 0:
        return None
    acc = None
    for w, t in pairs:
        term = t * (w / total_w)
        acc = term if acc is None else acc + term
    return acc


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------- aggregator

class Aggregator:
    """Implements R1-R6 above.

    modality_aware=False turns this into textbook FedAvg over the union of
    uploaded keys -- the degradation path required by the project's hard
    constraints. Use it whenever a result looks wrong and you need to tell
    "the method is broken" apart from "the plumbing is broken".
    """

    def __init__(
        self,
        param_groups: Dict[str, str],
        modality_aware: bool = True,
        use_reliability_weight: bool = False,
        missing_client_weight: float = 1.0,
        reliability_clamp=(0.2, 1.5),
        modality_alpha: float = 1.0,
    ):
        if not 0.0 <= modality_alpha <= 1.0:
            raise ValueError(
                "modality_alpha must be in [0, 1], got %r. 1 = full isolation "
                "(each modality keeps its own adapter), 0 = the pooled FedAvg "
                "adapter." % (modality_alpha,))
        self.param_groups = param_groups
        self.modality_aware = modality_aware
        self.modality_alpha = float(modality_alpha)
        self.use_reliability_weight = use_reliability_weight
        self.missing_client_weight = float(missing_client_weight)
        self.reliability_clamp = reliability_clamp

    # -------------------------------------------------- weighting (R5)

    def weight_of(self, upd: ClientUpdate, phase=None) -> float:
        w = float(upd.num_samples)
        if self.use_reliability_weight:
            w *= _clamp(float(upd.reliability), *self.reliability_clamp)
        if upd.aux_type == NO_AUX:
            w *= self.missing_client_weight
        if phase is not None and getattr(phase, "downweight_missing_aux", False) \
                and upd.aux_type == NO_AUX:
            w *= getattr(phase, "missing_client_weight", 1.0)
        return w

    # -------------------------------------------------- which keys are open

    def _trainable_keys(self, phase=None) -> List[str]:
        keys = [k for k, g in self.param_groups.items() if g in TRAINABLE_GROUPS]
        if phase is None:
            return keys
        allowed = getattr(phase, "trainable_groups", None)
        if not allowed:
            return keys
        allowed = set(allowed)
        return [k for k in keys if self.param_groups[k] in allowed]

    # -------------------------------------------------- main entry

    def aggregate(
        self,
        state: GlobalState,
        updates: Iterable[ClientUpdate],
        round_idx: int = 0,
        phase=None,
    ):
        """Return (new_state, AggregationStats). `state` is not mutated."""
        updates = list(updates)
        new_state = GlobalState(
            shared=OrderedDict((k, _clone(v)) for k, v in state.shared.items()),
            per_modality={
                aux: OrderedDict((k, _clone(v)) for k, v in bank.items())
                for aux, bank in state.per_modality.items()
            },
            initial_aux=OrderedDict((k, _clone(v)) for k, v in state.initial_aux.items()),
        )
        stats = AggregationStats(round_idx=round_idx)

        open_keys = self._trainable_keys(phase)
        shared_keys, aux_keys = [], []
        for k in open_keys:
            if self.modality_aware and is_modality_sensitive(self.param_groups[k]):
                aux_keys.append(k)
            else:
                shared_keys.append(k)

        # -- R1: shared tensors, every provider participates -------------
        for key in shared_keys:
            self._reduce_into(
                target=new_state.shared, key=key, bank_name="shared",
                candidates=updates, stats=stats, phase=phase,
                previous=state.shared.get(key),
            )

        # -- R2 / R3: modality banks, matched clients only ----------------
        if aux_keys:
            reporting = {u.aux_type for u in updates if u.aux_type != NO_AUX}
            # Also walk banks that already exist but got NO client this round.
            # They keep their value either way (new_state was copied), but they
            # must still produce a kept_previous record -- spec 9.1 requires the
            # skip reason to be auditable, and "the thermal bank silently sat
            # still for 12 rounds" is exactly the kind of thing that has to be
            # visible in param_group_stats.jsonl rather than inferred.
            existing = set(state.per_modality)
            for aux in sorted(reporting | existing):
                # First time we see this sensor: start it from the shared
                # initialisation, never from zeros.
                if aux not in new_state.per_modality:
                    template = {k: state.shared[k] for k in aux_keys if k in state.shared}
                    if not template:
                        template = {
                            k: u.state[k] for k in aux_keys
                            for u in updates if k in u.state
                        }
                    new_state.per_modality[aux] = OrderedDict(
                        (k, _clone(v)) for k, v in template.items()
                    )
                matched = [u for u in updates if u.aux_type == aux]
                for key in aux_keys:
                    wrote = self._reduce_into(
                        target=new_state.per_modality[aux], key=key, bank_name=aux,
                        candidates=matched, stats=stats, phase=phase,
                        previous=state.per_modality.get(aux, {}).get(key),
                        excluded=[u for u in updates if u.aux_type != aux],
                    )
                    # ---- partial sharing (2026-09-21) -------------------
                    # alpha = 1 is full isolation (what R2 describes and what
                    # every result before today used). alpha = 0 reproduces the
                    # pooled FedAvg tensor. In between, a bank keeps most of its
                    # modality's identity while still seeing the other sensors'
                    # data.
                    #
                    # Why this exists: measured on LasHeR, full isolation costs
                    # accuracy because it cuts each adapter's training data --
                    # the thermal bank sees 3 clients / 238,827 samples where
                    # the pooled one sees 9 / 474,647. The penalty shrank from
                    # -2.38 AUC at 3.2% of the data budget to -0.95 at 19.2%,
                    # i.e. it is a data-volume effect, not a modality effect.
                    # alpha buys back data at a proportional cost in purity
                    # instead of trading all of one for all of the other.
                    #
                    # NOT blended when this bank had no contributor: R4 says a
                    # bank with no data keeps its previous value, and folding a
                    # pooled tensor into it would quietly overwrite an adapter
                    # that merely sat out a round -- the exact failure R4 exists
                    # to prevent.
                    if self.modality_alpha < 1.0 and wrote:
                        pooled = self._pooled_value(key, updates, phase)
                        if pooled is not None:
                            a = self.modality_alpha
                            new_state.per_modality[aux][key] = (
                                new_state.per_modality[aux][key] * a
                                + pooled * (1.0 - a))
        else:
            # 2026-09-20 BUG FIX. Without this the FedAvg baseline was broken.
            #
            # With modality_aware=False every key -- aux_specific included --
            # is reduced into `shared` above. But GlobalState.view_for() is
            #     shared, then per_modality[aux] laid ON TOP,
            # and make_initial_state() seeds a bank per modality. Those banks,
            # never written in this branch, kept overriding the pooled result:
            # clients were dispatched the UNTRAINED initial adapters every
            # single round, and export_checkpoint() wrote those same initial
            # values out. Measured on the 20-round pilot: all 26 aux_specific
            # tensors of the exported fedavg checkpoint were bit-identical to
            # initialisation, while its other 84 trainable tensors had moved.
            #
            # Mirroring the pooled value into every bank is not a patch over the
            # view -- it is what FedAvg MEANS. R6 says modality_aware=False
            # collapses R2/R3 into R1: one adapter shared by every modality.
            # Every bank holding that same tensor states exactly that.
            modality_keys = [k for k in open_keys
                             if is_modality_sensitive(self.param_groups[k])]
            for bank in new_state.per_modality.values():
                for key in modality_keys:
                    if key in new_state.shared:
                        bank[key] = _clone(new_state.shared[key])
        return new_state, stats

    # -------------------------------------------------- one key, one bank

    def _reduce_into(self, target, key, bank_name, candidates, stats, phase,
                     previous=None, excluded=()):
        group = self.param_groups.get(normalize_key(key), "unknown")
        pairs, contributors, skipped = [], [], []

        for u in excluded:
            skipped.append(f"{u.client_id}:aux_type={u.aux_type}!={bank_name}")

        for u in candidates:
            if not u.provides(key):
                reason = "not_uploaded" if key not in u.state else "param_mask=False"
                skipped.append(f"{u.client_id}:{reason}")
                continue
            w = self.weight_of(u, phase)
            if w <= 0:
                skipped.append(f"{u.client_id}:weight<=0")
                continue
            pairs.append((w, u.state[key]))
            contributors.append(u.client_id)

        merged = _weighted_mean(pairs) if pairs else None

        # R4: nobody provided it -> keep what the server already had.
        if merged is None:
            if previous is not None:
                target[key] = _clone(previous)
            stats.per_key.append(KeyStat(
                key=key, group=group, bank=bank_name, n_contributors=0,
                contributors=[], total_weight=0.0, skipped=skipped,
                kept_previous=True,
            ))
            return False

        target[key] = merged
        stats.per_key.append(KeyStat(
            key=key, group=group, bank=bank_name,
            n_contributors=len(contributors), contributors=contributors,
            total_weight=float(sum(w for w, _ in pairs)), skipped=skipped,
        ))
        return True

    def _pooled_value(self, key, updates, phase):
        """The FedAvg tensor for `key`: weighted mean over EVERY client that
        provides it, regardless of modality.

        Used only by the partial-sharing blend. Records no KeyStat -- the bank's
        own reduction already logged who contributed to it, and a second record
        for a value that is only an intermediate would make
        param_group_stats.jsonl lie about how many banks exist.
        """
        pairs = []
        for u in updates:
            if not u.provides(key):
                continue
            w = self.weight_of(u, phase)
            if w > 0:
                pairs.append((w, u.state[key]))
        return _weighted_mean(pairs) if pairs else None


# ---------------------------------------------------------------- convenience

def make_initial_state(model_state, param_groups=None, aux_types=()):
    """Build the round-0 GlobalState from a freshly built model.

    Only trainable tensors are carried: the frozen backbone never moves across
    the network, which is the entire point of prompt tuning here.
    Every modality bank starts as a copy of the pretrained initialisation.
    """
    param_groups = param_groups or build_param_groups(model_state.keys())
    shared, aux_template = OrderedDict(), OrderedDict()
    for key, group in param_groups.items():
        if group not in TRAINABLE_GROUPS:
            continue
        if is_modality_sensitive(group):
            aux_template[key] = _clone(model_state[key])
        else:
            shared[key] = _clone(model_state[key])

    per_modality = {
        aux: OrderedDict((k, _clone(v)) for k, v in aux_template.items())
        for aux in aux_types if aux != NO_AUX
    }
    # Keep a copy in `shared` too, so a non-modality-aware (FedAvg) run and the
    # "first time we see this sensor" path both have something to start from.
    shared.update(aux_template)
    return GlobalState(shared=shared, per_modality=per_modality,
                       initial_aux=OrderedDict((k, _clone(v))
                                               for k, v in aux_template.items()))
