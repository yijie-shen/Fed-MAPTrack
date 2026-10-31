# Federated RGB-X Tracking with Group-Specific Prompt Aggregation

Code for **Fed-MAPTrack**: prompt-based federated RGB-X tracking built on the frozen
[ViPT](https://github.com/jiawen-zhu/ViPT) tracker. Only ViPT's prompt parameters are trained and exchanged
(about 0.84 M of 93 M parameters). Twelve simulated clients hold RGB-T (LasHeR), RGB-D (DepthTrack) and RGB-E (VisEvent)
videos; one client per dataset has no auxiliary sensor (RGB-only). The server keeps each group of prompt parameters
in the aggregation domain that fits its role:

| group | what it consumes | tensors (params) |
|---|---|---|
| `aux_specific` | the auxiliary modality (thermal / depth / event) | 664,416 |
| `rgb_shared` | the RGB half | 73,836 |
| `fusion` | the fused RGB + auxiliary features | 82,944 |
| `norm` | normalisation shared by construction | 18,432 |

This repository also contains the federated baselines used in the paper's comparison, so that every row can be rerun
from one code base.

> This is a research code base, forked from [ViPT](https://github.com/jiawen-zhu/ViPT) (MIT licence, see `LICENSE`).
> Only what the federated experiments need is kept: the tracker model, ViPT's data loading and loss, the federated layer
> in `lib/federated/`, and the scripts that evaluate the trained checkpoints. ViPT's own centralised training scripts,
> demo material and other benchmarks' tooling were removed; use the ViPT repository for those.

## What is where

| path | content |
|---|---|
| `lib/federated/server.py` | the federated server: variants, rounds, aggregation, checkpoint export |
| `lib/federated/client.py` | one client's local training (uses ViPT's official actor and loss, unchanged) |
| `lib/federated/aggregator.py` | mask-aware, group-aware aggregation (rules R1-R6 are documented in the file) |
| `lib/federated/param_groups.py` | which tensor belongs to which group |
| `lib/federated/personalized.py` | aggregation maths of the FedMSplit-P and FED-PRIME baselines |
| `lib/federated/test_*.py` | unit tests (plain scripts, no pytest needed) |
| `configs/fed/client_splits/` | the client partition used in the paper (`..._seed42_v2.json`) |
| `scripts/eval/` | evaluation helpers (LasHeR scorer, VisEvent scorer) |
| `scripts/supp/` | builds the DepthTrack-only partitions of the sensitivity / centralised-reference runs |
| `docs/` | how the two newest baselines are adapted, and how this tree was merged from the development branches |
| `experiments/vipt/` | ViPT configs; `deep_*_prime.yaml` are the configs for FED-PRIME checkpoints |

## Variants (`--variant`)

| row in the paper's Table 1 | command-line arguments |
|---|---|
| Local | `--variant local` |
| FedAvg | `--variant fedavg` |
| FedProx | `--variant fedprox --prox-mu 0.01` |
| FedPer | `--variant fedper` |
| Ditto | `--variant ditto --ditto-lambda 0.1` |
| FedMSplit-P | `--variant fedmsplit_p` |
| FED-PRIME | `--variant fedprime` |
| Independent FL | `--variant modality --clients <the four clients of one dataset>` (one run per dataset) |
| Fed-MAPTrack (aux-specific buckets only) | `--variant modality` |
| Fed-MAPTrack (all three groups bucketed, RGB-only clients participate) | `--variant modality --bucket-groups aux_specific,fusion,norm` |
| **Fed-MAPTrack** (selective participation, the final method) | `--variant modality --bucket-groups aux_specific,fusion,norm --selective-participation` |

Common arguments of every reported run: `--rounds 120 --local-steps 50 --workers 0 --clients-per-round 1.0
--save-every 20 --export-at-end`, seeds 42 / 0 / 123.

FedMSplit-P and FED-PRIME are the prompt-tuning baselines of Phung et al. (ICCV 2025), adapted from image-text
classification to ViPT. They are adapted references, not exact reproductions; every deviation from the original method
is listed in `docs/NEW_BASELINES.md`.

## Setup

1. Install the ViPT environment (`install_vipt.sh`, PyTorch 2.5.1 / CUDA 12.4 was used).
2. Download the pretrained foundation tracker (OSTrack, file `OSTrack_ep0300.pth.tar`) from the
   [ViPT release folder](https://drive.google.com/drive/folders/1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy?usp=sharing)
   and put it in `pretrained/`.
3. Download LasHeR, DepthTrack and VisEvent yourself (this repository ships no data) and point the code at them:

   ```
   export CVR_ROOT=/path/to/workdir          # default: the repository directory
   export CVR_DATA=/path/to/workdir/data     # default: $CVR_ROOT/data
   ```
   Expected layout: `data/lasher/trainingset`, `data/depthtrack/train`, `data/visevent/train_subset`
   (plus `trainlist.txt` for VisEvent) and the matching test sets (see `lib/test/evaluation/local.py`).
4. The client partition `configs/fed/client_splits/lasher_depthtrack_visevent_seed42_v2.json` fixes which sequences each
   of the twelve clients owns; its fingerprint is written into every run's `run_config.json`.

## Train

```
python lib/federated/server.py --variant modality --bucket-groups aux_specific,fusion,norm \
    --selective-participation --rounds 120 --local-steps 50 --workers 0 \
    --clients-per-round 1.0 --save-every 20 --export-at-end --seed 42 --out-dir output/fed/fed_maptrack_s42
```

`--dry-run` prints the round plan without training. An interrupted run continues bit-for-bit from its newest
`fed_state_rXXXX.pt` with `--resume <that file> --out-dir <the same directory>` (saved every `--save-every` rounds). A 120-round run (12 clients, 50 local steps) takes about 17-20 hours
on one RTX 4080 SUPER / 4090.

Personalised baselines (Local, FedPer, Ditto, FedMSplit-P, FED-PRIME) export one checkpoint per sensor client
(`ViPTrack_<client>_ep0120.pth.tar`); the other methods export one checkpoint per modality.

## Partition sensitivity and centralised reference (DepthTrack only)

`configs/fed/client_splits/depthtrack_only_alpha{0.1,0.5,1.0}_seed42.json` hold the four DepthTrack clients (three depth
sensors and one RGB-only client) under different Dirichlet concentrations `alpha` of the sequence partition (small
`alpha` = more skewed quantities); `depthtrack_only_centralized_seed42.json` has one client with all 146 training sequences.
`python scripts/supp/build_supp_splits.py` regenerates them (it builds the full twelve-client split first, because the three
datasets share one random stream). Runs use the same protocol as above (120 rounds, seed 42):

```
python lib/federated/server.py --split configs/fed/client_splits/depthtrack_only_alpha0.1_seed42.json --variant fedavg \
    --local-steps 50 --rounds 120 --clients-per-round 1.0 --workers 0 --seed 42 --save-every 10 --export-at-end --out-dir output/fed/sens_fa_a0.1
python lib/federated/server.py --split configs/fed/client_splits/depthtrack_only_alpha0.1_seed42.json --variant modality \
    --bucket-groups aux_specific,fusion,norm --selective-participation --local-steps 50 ... (same remaining arguments)
python lib/federated/server.py --split configs/fed/client_splits/depthtrack_only_centralized_seed42.json --variant fedavg \
    --local-steps 200 ... (one client, 120 x 200 = 24,000 steps, the same total number of steps as the four-client federation)
```

## Evaluate

Exported checkpoints load into the unchanged ViPT tracker. Evaluate the final round (120) only. Personalised methods are
scored per sensor client and averaged over the three clients of each benchmark. FED-PRIME checkpoints contain two extra
token pools and need the matching config: use `--yaml_name deep_rgbt_prime / deep_rgbd_prime / deep_rgbe_prime`.

* LasHeR: `RGBT_workspace/test_rgbt_mgpus.py`, then `scripts/eval/lasher_eval.py` (needs the LasHeR toolkit in `tools/LasHeR_Toolkit`)
* VisEvent: `RGBE_workspace/test_rgbe_mgpus.py`, then `scripts/eval/visevent_eval.py` (needs the VisEvent benchmark toolkit in `tools/VisEvent_SOT_Benchmark`)
* DepthTrack: `Depthtrack_workspace/run_offline_depthtrack.py`, then the VOT toolkit's analysis (F-score)

## Reproducibility notes

* All paper numbers are mean +- sample standard deviation over three seeds at round 120 (see the paper).
* Single-GPU evaluation of the same checkpoint on two different GPU models differs by up to about 1 point on LasHeR;
  expect numbers of that order, not bit-exact agreement, when you rerun on other hardware.
* Everything not trained (the 92.5 M backbone) is identical across runs.
* `python lib/federated/test_server.py` (and the other `test_*.py` files) must pass before trusting a modification.

## Licence and acknowledgements

MIT, see `LICENSE`. The tracker, training and evaluation code are from ViPT (Zhu et al., CVPR 2023), which builds on
OSTrack. The prompt-tuning baselines follow Phung et al. (ICCV 2025) and Chen and Zhang (KDD 2022).
