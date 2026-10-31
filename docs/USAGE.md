# Usage details

The short version is in the main [README](../README.md). This page keeps the details.

## Where things are

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
is listed in [`NEW_BASELINES.md`](NEW_BASELINES.md).

## Environment and data

* Python 3.10, PyTorch 2.5.1 (CUDA 12.4), packages as in `requirements.txt` (see the README for the install commands).
* Data paths come from two optional environment variables: `CVR_ROOT` (working directory, default: this repository) and
  `CVR_DATA` (datasets, default `$CVR_ROOT/data`). Layout: `data/lasher/trainingset`, `data/depthtrack/train`,
  `data/visevent/train_subset` (with `trainlist.txt`), and the test sets `data/lasher/testingset`, `data/depthtrack/test`,
  `data/visevent/test_subset`.
* The client partition of the paper, `configs/fed/client_splits/lasher_depthtrack_visevent_seed42_v2.json`, fixes which
  sequences each of the twelve clients owns; its fingerprint is written into every run's `run_config.json`.

## Running

Every run uses the arguments of the table above. `--dry-run` prints the round plan without training. An interrupted run
continues bit-for-bit from its newest `fed_state_rXXXX.pt` with `--resume <that file> --out-dir <the same directory>`
(saved every `--save-every` rounds). Personalised methods (Local, FedPer, Ditto, FedMSplit-P, FED-PRIME) export one
checkpoint per sensor client (`ViPTrack_<client>_ep0120.pth.tar`); the other methods export one checkpoint per modality.

The DepthTrack-only partition and centralised-reference runs are described in [`EXTRA_RESULTS.md`](EXTRA_RESULTS.md).

## Evaluation

Exported checkpoints load into the unchanged ViPT tracker. Evaluate the final round (120) only. Personalised methods are
scored per sensor client and averaged over the three clients of each benchmark. FED-PRIME checkpoints contain two extra
token pools and need the matching config: use `--yaml_name deep_rgbt_prime / deep_rgbd_prime / deep_rgbe_prime`.

* LasHeR: `RGBT_workspace/test_rgbt_mgpus.py`, then `scripts/eval/lasher_eval.py` (needs the LasHeR toolkit in `tools/LasHeR_Toolkit`)
* VisEvent: `RGBE_workspace/test_rgbe_mgpus.py`, then `scripts/eval/visevent_eval.py` (needs the VisEvent benchmark toolkit in `tools/VisEvent_SOT_Benchmark`)
* DepthTrack: `Depthtrack_workspace/run_offline_depthtrack.py`, then the VOT toolkit's analysis (F-score)

## Reproducibility notes

* All paper numbers are mean ± sample standard deviation over three seeds at round 120 (see the paper).
* Single-GPU evaluation of the same checkpoint on two different GPU models differs by up to about 1 point on LasHeR;
  expect numbers of that order, not bit-exact agreement, when you rerun on other hardware.
* Everything not trained (the 92.5 M backbone) is identical across runs.
* `python lib/federated/test_server.py` (and the other `test_*.py` files) must pass before trusting a modification.
