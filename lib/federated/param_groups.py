#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parameter grouping for federated ViPT.

Every key of the model state_dict is assigned to exactly one group. The
aggregator then applies a different rule per group (spec section 9.1).

This module is deliberately torch-free: grouping is pure string logic, so it
can be unit-tested without a GPU or even without torch installed.

-------------------------------------------------------------------------
Why these groups -- grounded in what the code actually computes, not naming
-------------------------------------------------------------------------
ViPT feeds Prompt_block a channel-wise concat of [RGB features | aux features]
(lib/models/vipt/vit_ce_prompt.py:210-213 and 299-302):

    z_feat = torch.cat([z_feat, z_dte_feat], dim=1)
    z_feat = self.prompt_blocks[i](z_feat)

and inside Prompt_block.forward (vit_ce_prompt.py:62-71):

    x0 = x[:, 0:C/2]  -> conv0_0 -> fovea     # the RGB half
    x1 = x[:, C/2:]   -> conv0_1              # the aux-modality half
    out = conv1x1(fovea(conv0_0(x0)) + conv0_1(x1))

So each trainable tensor is defined by WHICH SIGNAL IT CONSUMES:

  AUX_SPECIFIC  patch_embed_prompt, conv0_1
        Only ever sees the auxiliary modality (or the prompt derived from it).
        Depth is "metres from the lens", thermal is "temperature", event is
        "brightness change" -- physically incommensurable. Averaging these
        across clients that hold different sensors is the core difficulty
        this project is about.

  RGB_SHARED    conv0_0, fovea
        Only ever sees the RGB half. RGB is the one modality EVERY client has,
        so there is no cross-modality conflict here -- safe to aggregate
        globally from round one. This is what makes spec's Phase 1
        ("RGB common warm-up") implementable.

  FUSION        conv1x1
        Projects the summed RGB+aux branches back to embed_dim. Sits on both
        sides, so it is treated as shared but opened later (spec Phase 3).

  NORM          prompt_norms
        The SAME prompt_norms[i] is applied to z, x, z_dte and x_dte
        (vit_ce_prompt.py:206-209) -- shared across modalities by construction.

  FROZEN        everything else (the ViT backbone and box_head)
        ViPT freezes anything whose name lacks "prompt"
        (lib/train/base_functions.py:179-181). Never trained, never uploaded.
        This is why one round costs 0.84 M instead of 93.36 M.

Measured sizes (2026-09-17, deep_rgbt.yaml):
    patch_embed_prompt   0.5906 M   70.3% of all trainable params
    conv0_0              0.0738 M
    conv0_1              0.0738 M
    conv1x1              0.0829 M
    fovea.smooth         ~0
    prompt_norms         0.0184 M
    total                0.8396 M

Note the asymmetry: AUX_SPECIFIC is ~0.66 M of the 0.84 M, i.e. about 79% of
what gets uploaded is modality-specific. Whatever we do about cross-modality
conflict dominates the communication story.
"""

from __future__ import annotations

import re
from collections import OrderedDict

# ---------------------------------------------------------------- group names

FROZEN = "frozen"                # not trained, never aggregated
AUX_SPECIFIC = "aux_specific"    # consumes the auxiliary modality
RGB_SHARED = "rgb_shared"        # consumes the RGB half
FUSION = "fusion"                # fuses the two branches
NORM = "norm"                    # normalisation, shared by construction

#: Groups that are actually trained and therefore uploaded each round.
TRAINABLE_GROUPS = (AUX_SPECIFIC, RGB_SHARED, FUSION, NORM)

#: Groups whose meaning differs per auxiliary modality. The aggregator must
#: only average these across clients that hold the SAME aux_type.
MODALITY_SENSITIVE_GROUPS = (AUX_SPECIFIC,)

# ---------------------------------------------------------------- rules
# Order matters: first match wins. Patterns are matched against the full
# state_dict key, so they survive a "module." prefix from DataParallel/DDP
# (we strip that first, see normalize_key).

_RULES = (
    # --- auxiliary-modality path -------------------------------------------
    (re.compile(r"(^|\.)patch_embed_prompt\."), AUX_SPECIFIC),
    (re.compile(r"(^|\.)prompt_blocks\.\d+\.conv0_1\."), AUX_SPECIFIC),
    # --- RGB path -----------------------------------------------------------
    (re.compile(r"(^|\.)prompt_blocks\.\d+\.conv0_0\."), RGB_SHARED),
    (re.compile(r"(^|\.)prompt_blocks\.\d+\.fovea\."), RGB_SHARED),
    # --- fusion -------------------------------------------------------------
    (re.compile(r"(^|\.)prompt_blocks\.\d+\.conv1x1\."), FUSION),
    # --- normalisation ------------------------------------------------------
    (re.compile(r"(^|\.)prompt_norms\.\d+\."), NORM),
)


class UnknownPromptParam(RuntimeError):
    """A key contains 'prompt' but matches no rule.

    This is raised rather than silently bucketed, because a new trainable
    tensor appearing in the model without a grouping rule would otherwise be
    quietly dropped from aggregation -- a bug that produces no error and only
    shows up as slightly worse numbers.
    """


def normalize_key(key: str) -> str:
    """Strip the 'module.' prefix that DataParallel / DDP adds."""
    return key[len("module."):] if key.startswith("module.") else key


def group_of(key: str, strict: bool = True) -> str:
    """Return the group name for one state_dict key.

    ViPT's own freeze rule is 'trainable iff "prompt" in name'
    (base_functions.py:179-181), so anything without 'prompt' is FROZEN.
    """
    k = normalize_key(key)
    if "prompt" not in k:
        return FROZEN
    for pattern, group in _RULES:
        if pattern.search(k):
            return group
    if strict:
        raise UnknownPromptParam(
            f"key {key!r} contains 'prompt' but matches no grouping rule. "
            f"A new trainable tensor was probably added to the model; add a "
            f"rule to _RULES rather than letting it fall through, otherwise "
            f"it would be silently excluded from aggregation."
        )
    return FROZEN


def build_param_groups(keys, strict: bool = True) -> "OrderedDict[str, str]":
    """Map every state_dict key to its group. Returns {key: group}."""
    return OrderedDict((k, group_of(k, strict=strict)) for k in keys)


def group_to_keys(param_groups) -> "OrderedDict[str, list]":
    """Invert {key: group} into {group: [keys]}, groups in a stable order."""
    out = OrderedDict((g, []) for g in (FROZEN,) + TRAINABLE_GROUPS)
    for key, group in param_groups.items():
        out.setdefault(group, []).append(key)
    return out


def trainable_keys(param_groups):
    """Keys that are trained and therefore uploaded each round."""
    return [k for k, g in param_groups.items() if g in TRAINABLE_GROUPS]


def is_modality_sensitive(group: str) -> bool:
    """True if this group's meaning depends on which sensor the client has."""
    return group in MODALITY_SENSITIVE_GROUPS


def summarize(param_groups, numels=None):
    """Human-readable per-group counts (and parameter totals if numels given).

    Returns {group: {"tensors": int, "params": int|None}}.
    """
    out = OrderedDict()
    for group in (FROZEN,) + TRAINABLE_GROUPS:
        out[group] = {"tensors": 0, "params": 0 if numels else None}
    for key, group in param_groups.items():
        slot = out.setdefault(group, {"tensors": 0, "params": 0 if numels else None})
        slot["tensors"] += 1
        if numels is not None:
            slot["params"] += int(numels.get(key, 0))
    return out


def format_summary(param_groups, numels=None) -> str:
    """One-line-per-group table, for logging at the start of a run."""
    s = summarize(param_groups, numels)
    total_params = sum(v["params"] or 0 for v in s.values()) if numels else 0
    trainable_params = sum(
        s[g]["params"] or 0 for g in TRAINABLE_GROUPS if g in s
    ) if numels else 0

    lines = [f"{'group':<14} {'tensors':>8} {'params':>12} {'share of upload':>16}"]
    for group, v in s.items():
        if v["tensors"] == 0:
            continue
        if numels is None:
            lines.append(f"{group:<14} {v['tensors']:>8} {'-':>12} {'-':>16}")
            continue
        share = (
            f"{v['params'] / trainable_params * 100:>15.1f}%"
            if group in TRAINABLE_GROUPS and trainable_params
            else " " * 16
        )
        lines.append(f"{group:<14} {v['tensors']:>8} {v['params']:>12,}{share}")
    if numels is not None:
        lines.append(
            f"{'-'*14} {'-'*8} {'-'*12} {'-'*16}\n"
            f"{'uploaded':<14} {'':>8} {trainable_params:>12,} "
            f"({trainable_params / total_params * 100:.2f}% of {total_params:,})"
        )
    return "\n".join(lines)
