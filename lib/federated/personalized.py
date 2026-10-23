"""Server-side maths of the two personalised baselines added 2026-10-16.

Both come from Phung et al., "Federated Prompt-Tuning with Heterogeneous and
Incomplete Multimodal Client Data" (ICCV'25; code: github.com/hangpt01/FedPrime):

  FedMSplit-P   the paper's prompt-tuning adaptation of FedMSplit (KDD'22).
  FED-PRIME     the paper's own method (two prompt pools per client).

Everything here is pure torch/scipy on CPU tensors so it can be unit-tested
without a GPU and without the ViPT network.  The wiring (dispatch, storage,
export) lives in server.py.

-------------------------------------------------------------------------
FedMSplit-P  (official FedMSplit_P.py, algorithm/multimodal/food101)
-------------------------------------------------------------------------
Each client keeps its own model.  The model is cut into BLOCKS (official: 5,
here: the 4 ViPT parameter groups).  Per block b the server stacks the clients'
flattened block vectors x_k^b and computes

    A_b[k, l] = softmax_l( <x_k^b, x_l^b> / sqrt(d_b) )

Client k's new value of block b is a weighted average over clients l with

    w_kl = A_b[k, l] * c_kl ,   c_kl = mean over blocks b' of A_b'[k, l]

(official: sum_b' |A_b'[k,l]| / 5; A >= 0 so |.| is a no-op and /5 is a mean
over the blocks that exist).  A client that does not provide a block (an
RGB-only client has no valid auxiliary tensors) is left out of that block, and
c_kl averages only over blocks that both clients provide.

The official code ALSO returns a data-volume-weighted global average; the
proximal term of local training pulls toward it (lambda = 0.01, the official
argparse default).  We return it as `global_average`.

-------------------------------------------------------------------------
FED-PRIME inter-client prompt alignment  (official nonparametric_aggregation.py)
-------------------------------------------------------------------------
Probabilistic-matching aggregation of the clients' "global pool" prompts:
the server keeps tau cluster centres; every client's prompts are matched
ONE-TO-ONE to centres (Hungarian algorithm) under a Gaussian likelihood whose
per-dimension std comes from a small learned network, plus a learned "centre
popularity" term.  See `align_global_prompts` for the exact loop and the three
places where it differs from the official code.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

FEDMSPLIT_PROX_LAMBDA = 0.01      # official argparse default
PRIME_POOL_SIZE = 20              # paper, "Prompt pools contain 20 prompts each"
PRIME_TOP_K = 5                   # paper, top-5 per pool
PRIME_SIM_SCALAR = 0.01           # official argparse default (--reduce_sim_scalar)
PRIME_OUTER_LOOPS = 5             # official argparse default (--num_outer_loops)
PRIME_ALIGN_HIDDEN = 128          # NonparametricAgg(768, n_hidden=128)
PRIME_ALIGN_LR = 1e-3             # official: Adam with PyTorch's default lr


# ------------------------------------------------------------- FedMSplit-P

def fedmsplit_p_aggregate(states: "OrderedDict[str, Dict[str, torch.Tensor]]",
                          blocks: "OrderedDict[str, List[str]]",
                          provides: Dict[str, Dict[str, bool]],
                          weights: Dict[str, float]):
    """Personalised update for every client in `states`.

    states    {client_id: {key: tensor}}  what each client uploaded this round
    blocks    {block: [keys]}             how the model is cut
    provides  {client_id: {block: bool}}  does the client hold a valid block
    weights   {client_id: float}          data volume, for the global average

    Returns (personal, global_average, info):
      personal        {client_id: {key: tensor}}  new start point per client
                      (only blocks the client provides)
      global_average  {key: tensor}                weighted mean of the uploads
                      (only over clients that provide the block)
      info            {block: {"self_weight": mean A_b[k,k]-share, ...}}
    """
    cids = list(states)
    members = {b: [c for c in cids if provides[c].get(b, False)] for b in blocks}
    attn = {}                                   # block -> {(k,l): A_b[k,l]}
    for b, keys in blocks.items():
        P = members[b]
        if not P:
            continue
        flat = torch.stack([
            torch.cat([states[c][k].detach().double().reshape(-1) for k in keys])
            for c in P])                         # |P| x d_b
        d = flat.shape[1]
        A = torch.softmax(flat @ flat.T / math.sqrt(d), dim=1)
        for i, k in enumerate(P):
            for j, l in enumerate(P):
                attn[(b, k, l)] = float(A[i, j])

    personal = {c: OrderedDict() for c in cids}
    info = OrderedDict()
    for b, keys in blocks.items():
        P = members[b]
        if not P:
            continue
        W = {}
        for k in P:
            row = {}
            for l in P:
                shared = [bb for bb in blocks
                          if (bb, k, l) in attn]
                c_kl = sum(attn[(bb, k, l)] for bb in shared) / len(shared)
                row[l] = attn[(b, k, l)] * c_kl
            tot = sum(row.values())
            W[k] = {l: v / tot for l, v in row.items()}
        for k in P:
            for key in keys:
                acc = None
                for l in P:
                    term = states[l][key].detach().double() * W[k][l]
                    acc = term if acc is None else acc + term
                personal[k][key] = acc.to(states[k][key].dtype)
        info[b] = OrderedDict(
            providers=len(P),
            mean_self_weight=float(np.mean([W[k][k] for k in P])),
            max_off_diagonal=float(max([W[k][l] for k in P for l in P if k != l]
                                       or [0.0])))

    global_average = OrderedDict()
    for b, keys in blocks.items():
        P = members[b]
        if not P:
            continue
        tot = float(sum(weights[c] for c in P))
        for key in keys:
            acc = None
            for c in P:
                term = states[c][key].detach().double() * (weights[c] / tot)
                acc = term if acc is None else acc + term
            global_average[key] = acc.to(states[P[0]][key].dtype)
    return personal, global_average, info


# ----------------------------------------------------------- FED-PRIME align

class _AlignNets(nn.Module):
    """The two small networks of the official `NonparametricAgg`.

    cov_net        centre -> per-dimension std (sigmoid)
    bernoulli_net  centre -> probability that the centre receives a prompt
                   from a client (sigmoid)
    """

    def __init__(self, dim: int, n_hidden: int):
        super().__init__()
        self.cov_net = nn.Sequential(nn.Linear(dim, n_hidden), nn.ReLU(),
                                     nn.Linear(n_hidden, dim), nn.Sigmoid())
        self.bernoulli_net = nn.Sequential(nn.Linear(dim, n_hidden), nn.ReLU(),
                                           nn.Linear(n_hidden, 1), nn.Sigmoid())

    def std(self, centres):
        return self.cov_net(centres).clamp_min(1e-4)

    def pop(self, centres):
        return self.bernoulli_net(centres).squeeze(-1).clamp(1e-4, 1 - 1e-4)


def _log_gauss(rows, centres, std):
    """log N(rows[i]; centres[q], diag(std[q]^2)) -> (m, tau)."""
    diff = (rows[:, None, :] - centres[None, :, :]) / std[None, :, :]
    return -0.5 * (diff.pow(2).sum(-1)
                   + 2.0 * std.log().sum(-1)[None, :]
                   + rows.shape[1] * math.log(2 * math.pi))


def _m_step(previous, rows_by_client, assign):
    """Centre = mean of {previous centre} U {matched rows}.

    The previous centre enters with weight 1 (a prior), so a centre nobody was
    matched to keeps its value instead of being pruned.
    """
    num = previous.clone()
    cnt = torch.ones(previous.shape[0])
    for c, rows in rows_by_client.items():
        for i, q in enumerate(assign[c]):
            num[q] += rows[i]
            cnt[q] += 1
    return num / cnt[:, None], cnt - 1


def align_global_prompts(rows_by_client: "OrderedDict[str, torch.Tensor]",
                         previous: torch.Tensor,
                         outer_loops: int = PRIME_OUTER_LOOPS,
                         seed: int = 0,
                         n_hidden: int = PRIME_ALIGN_HIDDEN,
                         lr: float = PRIME_ALIGN_LR):
    """FED-PRIME's server step for the inter-client ("global") prompt pool.

    rows_by_client  {client_id: (m_c, D)} prompts the client actually used this
                    round (m_c <= tau); unused rows are not uploaded
    previous        (tau, D) pool the server dispatched this round

    Returns (new_pool (tau, D), info).

    Loop (mirrors the official NonparametricAgg.forward, 5 outer loops):
      0. initial matching of every client to `previous` (squared distance)
      for each outer loop, for each client c (sorted id order):
        a. one Adam step (lr 1e-3) on the two networks, minimising
              -[ sum_{matched rows} log N(row; centre, std(centre)^2)
                 + sum_q log Bernoulli(z_cq ; popularity(centre_q)) ]
        b. re-solve c's matching with the Hungarian algorithm, maximising
              log N(row; centre_q, std_q^2) + logit(popularity_q)
        c. recompute the centres (closed form, `_m_step`)

    DIFFERENCES FROM THE OFFICIAL CODE (to be disclosed):
      * tau centres, not up to n*tau.  A client's pool has a fixed size tau, so
        the dispatched pool must too; the official code lets the pool grow and
        prunes empty centres.  Here an empty centre keeps its previous value.
      * centres are updated in closed form (member mean).  The official code
        moves them with the same single Adam step at lr 1e-3, which for
        768-dim prompts of scale 0.02 leaves them where they started.
      * the networks are re-initialised (seeded) every round, so the server is
        stateless and a resumed run reproduces an uninterrupted one.
    """
    cids = sorted(rows_by_client)
    tau, dim = previous.shape
    if not cids:
        return previous.clone(), {"clients": 0}
    previous = previous.detach().double().float()
    rows = OrderedDict((c, rows_by_client[c].detach().float()) for c in cids)
    for c, r in rows.items():
        if r.shape[0] > tau or r.shape[1] != dim:
            raise ValueError("client %s uploaded %s rows for a pool %s"
                             % (c, tuple(r.shape), tuple(previous.shape)))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        nets = _AlignNets(dim, n_hidden)
    opt = torch.optim.Adam(nets.parameters(), lr=lr)

    assign: Dict[str, List[int]] = {}
    for c in cids:
        cost = torch.cdist(rows[c], previous).pow(2).numpy()
        r_idx, q_idx = linear_sum_assignment(cost)
        assign[c] = [int(q) for _, q in sorted(zip(r_idx, q_idx))]
    centres, _ = _m_step(previous, rows, assign)

    for _ in range(int(outer_loops)):
        for c in cids:
            # a. one Adam step on the networks, centres held fixed
            std = nets.std(centres)
            pop = nets.pop(centres)
            loglik = torch.zeros(())
            z = torch.zeros(tau)
            for cc in cids:
                lg = _log_gauss(rows[cc], centres, std)
                for i, q in enumerate(assign[cc]):
                    loglik = loglik + lg[i, q]
            for cc in cids:
                z_c = torch.zeros(tau)
                z_c[assign[cc]] = 1.0
                loglik = loglik + (z_c * pop.log()
                                   + (1 - z_c) * (1 - pop).log()).sum()
            opt.zero_grad()
            (-loglik).backward()
            opt.step()
            # b. Hungarian re-assignment of client c
            with torch.no_grad():
                std = nets.std(centres)
                pop = nets.pop(centres)
                score = (_log_gauss(rows[c], centres, std)
                         + (pop / (1 - pop)).log()[None, :]).numpy()
            r_idx, q_idx = linear_sum_assignment(score, maximize=True)
            assign[c] = [int(q) for _, q in sorted(zip(r_idx, q_idx))]
            # c. centres
            centres, _ = _m_step(previous, rows, assign)

    _, counts = _m_step(previous, rows, assign)
    info = OrderedDict(clients=len(cids),
                       rows=int(sum(r.shape[0] for r in rows.values())),
                       empty_centres=int((counts == 0).sum()),
                       mean_shift=float((centres - previous).norm(dim=1).mean()))
    return centres.detach(), info
