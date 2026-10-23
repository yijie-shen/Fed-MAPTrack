#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Federated wrapper around the official ViPT model (spec sections 4.2, 11.1, 16).

design note 11.1: "if the original forward cannot take aux_type/modality_mask, add fed_wrapper.py
as a wrapper instead of modifying the original model." So nothing here edits lib/models/vipt/*; this module
only wraps, and the official single-machine training path never imports it.

-------------------------------------------------------------------------
What the official forward actually wants
-------------------------------------------------------------------------
ViPT takes ONE 6-channel tensor and splits it itself
(lib/models/vipt/vit_ce_prompt.py:190-196):

    x_rgb = x[:, :3]      # RGB
    x_dte = x[:, 3:]      # Depth / Thermal / Event

So the wrapper's main job is assembling that tensor from the separate
template_rgb / template_aux the spec interface asks for, and deciding what to
put in channels 3-5 when a client has no auxiliary sensor at all.

-------------------------------------------------------------------------
Missing modality: what goes in channels 3..5
-------------------------------------------------------------------------
design note 17: "do not pretend a missing auxiliary modality is a real one by passing zero tensors, unless
a modality_mask is passed as well; otherwise the adapter learns wrong statistics."

So zeros are allowed exactly when the absence is declared. That is what this
wrapper does, and the declaration is not optional -- every forward returns a
`modality_mask` and an `aux_present` flag, and `param_mask_for()` turns the
same fact into the per-key mask the aggregator uses to skip that client's
AUX_SPECIFIC tensors (rule R2/R3 in aggregator.py).

Why zeros rather than duplicating RGB (`aux_fill='rgb'`, also available):
  * zeros inject no information. patch_embed_prompt(0) is just its bias, so the
    prompt branch contributes a constant -- well-defined and stable.
  * duplicating RGB injects WRONG information: conv0_1 would be trained to read
    RGB as if it were temperature/depth. That is a worse lie than "nothing here",
    and it is invisible in the loss.
Neither is free, which is exactly why the choice is recorded in the output and
why the aggregator is told to skip those tensors instead of averaging them in.

-------------------------------------------------------------------------
adapter_stats (spec 4.2 return_aux_stats)
-------------------------------------------------------------------------
Collected with forward hooks on prompt_blocks / prompt_norms, so no model code
is touched. Reports the norm of each layer's prompt output, which is the
cheapest honest answer to "how much is the auxiliary modality actually
contributing in this client?" -- useful for the paper and for noticing that a
modality has quietly stopped mattering.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from .param_groups import (
    AUX_SPECIFIC,
    TRAINABLE_GROUPS,
    build_param_groups,
    is_modality_sensitive,
)

NO_AUX = "none"
KNOWN_AUX = ("thermal", "depth", "event")
RGB_CHANNELS = 3


class FedViPTWrapper(nn.Module):
    """Give ViPT the spec 4.2 interface without editing ViPT.

    The wrapped module is reachable as `.net`, so the official trainer can keep
    using it directly and stays unaffected.
    """

    def __init__(self, net: nn.Module, param_groups=None, aux_fill: str = "zeros"):
        super().__init__()
        if aux_fill not in ("zeros", "rgb"):
            raise ValueError(f"aux_fill must be 'zeros' or 'rgb', got {aux_fill!r}")
        self.net = net
        self.aux_fill = aux_fill
        self.param_groups = param_groups or build_param_groups(net.state_dict().keys())
        self._hooks = []
        self._captured = OrderedDict()

    # ------------------------------------------------------------ input prep

    @staticmethod
    def _split_if_packed(t):
        """Accept either a 6-channel tensor (what the existing LTRLoader gives)
        or a 3-channel RGB tensor. Returns (rgb, aux_or_None)."""
        if t is None:
            return None, None
        if t.shape[1] == 2 * RGB_CHANNELS:
            return t[:, :RGB_CHANNELS], t[:, RGB_CHANNELS:]
        if t.shape[1] == RGB_CHANNELS:
            return t, None
        raise ValueError(
            f"expected 3 or 6 channels, got {t.shape[1]}; ViPT splits "
            f"x[:, :3] as RGB and x[:, 3:] as the auxiliary modality"
        )

    def _assemble(self, rgb, aux, aux_present):
        """Build the 6-channel tensor ViPT expects."""
        if aux is not None and aux_present:
            return torch.cat([rgb, aux], dim=1)
        if self.aux_fill == "rgb":
            return torch.cat([rgb, rgb], dim=1)
        return torch.cat([rgb, torch.zeros_like(rgb)], dim=1)

    # ------------------------------------------------------------ hooks

    def _attach_hooks(self):
        backbone = getattr(self.net, "backbone", None)
        if backbone is None:
            return
        for name in ("prompt_blocks",):
            mod = getattr(backbone, name, None)
            if mod is None:
                continue
            for i, block in enumerate(mod):
                self._hooks.append(
                    block.register_forward_hook(self._make_hook(f"{name}.{i}"))
                )

    def _make_hook(self, tag):
        def hook(_module, _inp, out):
            if isinstance(out, torch.Tensor):
                # Mean L2 norm per sample: scale-comparable across batch sizes.
                v = out.detach().float().flatten(1).norm(dim=1).mean().item()
                self._captured.setdefault(tag, []).append(v)
        return hook

    def _detach_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ------------------------------------------------------------ forward

    def forward(
        self,
        template_rgb,
        search_rgb,
        template_aux=None,
        search_aux=None,
        aux_type: str = NO_AUX,
        modality_mask=None,
        return_aux_stats: bool = False,
        ce_template_mask=None,
        ce_keep_rate=None,
        return_last_attn: bool = False,
    ):
        # Allow callers to hand us the packed 6-channel tensors the existing
        # data pipeline already produces.
        template_rgb, packed_t_aux = self._split_if_packed(template_rgb)
        search_rgb, packed_s_aux = self._split_if_packed(search_rgb)
        template_aux = template_aux if template_aux is not None else packed_t_aux
        search_aux = search_aux if search_aux is not None else packed_s_aux

        aux_present = (
            aux_type != NO_AUX
            and template_aux is not None
            and search_aux is not None
        )
        if modality_mask is None:
            modality_mask = self.build_modality_mask(aux_type, aux_present)
        else:
            if not isinstance(modality_mask, dict):
                raise ValueError("modality_mask must be a dict")
            if any(k not in ("rgb", *KNOWN_AUX) for k in modality_mask):
                raise ValueError("unsupported modality_mask key")
            requested = bool(modality_mask.get(aux_type, 0))
            aux_present = aux_present and requested
            modality_mask = self.build_modality_mask(aux_type, aux_present)

        z = self._assemble(template_rgb, template_aux, aux_present)
        x = self._assemble(search_rgb, search_aux, aux_present)

        if return_aux_stats:
            self._captured = OrderedDict()
            self._attach_hooks()
        try:
            out = self.net(
                template=z, search=x,
                ce_template_mask=ce_template_mask,
                ce_keep_rate=ce_keep_rate,
                return_last_attn=return_last_attn,
            )
        finally:
            if return_aux_stats:
                self._detach_hooks()

        # ViPT returns a single dict (unlike SDSTrack, which returns a list of
        # two branches for self-distillation). Guard anyway so a future change
        # upstream fails loudly rather than silently taking the wrong branch.
        if isinstance(out, (list, tuple)):
            raise RuntimeError(
                "wrapped model returned a sequence; ViPT is expected to return a "
                "single dict. If the backbone was swapped, update this wrapper "
                "rather than indexing blindly."
            )

        result = OrderedDict(out)
        result["features"] = out.get("backbone_feat")
        result["aux_present"] = aux_present
        result["aux_type"] = aux_type if aux_present else NO_AUX
        result["modality_mask"] = modality_mask
        result["aux_fill"] = None if aux_present else self.aux_fill
        result["loss_inputs"] = {
            k: out[k] for k in ("pred_boxes", "score_map", "size_map", "offset_map")
            if k in out
        }
        result["adapter_stats"] = (
            self._summarize_captured() if return_aux_stats else {}
        )
        return result

    def _summarize_captured(self):
        stats = OrderedDict()
        for tag, values in self._captured.items():
            if values:
                stats[tag] = sum(values) / len(values)
        if stats:
            stats["prompt_norm_mean"] = sum(stats.values()) / len(stats)
        return stats

    # ------------------------------------------------------------ masks

    @staticmethod
    def build_modality_mask(aux_type: str, aux_present: bool = True):
        """spec 3.1: {"rgb":1,"thermal":0/1,"depth":0/1,"event":0/1}."""
        mask = OrderedDict(rgb=1)
        for m in KNOWN_AUX:
            mask[m] = int(aux_present and aux_type == m)
        return mask

    def param_mask_for(self, aux_type: str, aux_present: bool = True):
        """Which trainable keys this client may legitimately upload.

        AUX_SPECIFIC tensors of a client with no auxiliary sensor are marked
        False and are frozen locally; the server also dispatches their initial
        pretrained values to such clients.
        """
        valid = bool(aux_present) and aux_type != NO_AUX
        return OrderedDict(
            (k, (valid or not is_modality_sensitive(g)))
            for k, g in self.param_groups.items()
            if g in TRAINABLE_GROUPS
        )

    # ------------------------------------------------------------ fed state

    def trainable_state(self, detach: bool = True):
        """Exactly what a client uploads: the 0.84 M, never the frozen 92.5 M."""
        sd = self.net.state_dict()
        out = OrderedDict()
        for k, g in self.param_groups.items():
            if g in TRAINABLE_GROUPS and k in sd:
                v = sd[k]
                out[k] = v.detach().clone() if detach else v
        return out

    def load_trainable_state(self, state, strict_groups: bool = True):
        """Apply what the server dispatched. Frozen tensors are never touched.

        Returns the list of keys actually written, so the caller can assert the
        dispatch contained what it expected instead of hoping.
        """
        sd = self.net.state_dict()
        written = []
        for k, v in state.items():
            if k not in sd:
                if strict_groups:
                    raise KeyError(f"{k!r} is not a parameter of the wrapped model")
                continue
            g = self.param_groups.get(k)
            if g not in TRAINABLE_GROUPS:
                if strict_groups:
                    raise ValueError(
                        f"refusing to overwrite {k!r}: it is in group {g!r}, not "
                        f"trainable. The frozen backbone must never travel."
                    )
                continue
            sd[k].copy_(v)
            written.append(k)
        return written

    # ------------------------------------------------------------ misc

    def aux_specific_keys(self):
        return [k for k, g in self.param_groups.items() if g == AUX_SPECIFIC]

    def __repr__(self):
        n_train = sum(1 for g in self.param_groups.values() if g in TRAINABLE_GROUPS)
        return (f"FedViPTWrapper(aux_fill={self.aux_fill!r}, "
                f"trainable_tensors={n_train})")
