# Fed-MAPTrack

**Fed-MAPTrack: Federated Modality-aware Prompt Tuning for RGB-X Tracking**

Paper: link will be added after publication.

## Method

- Prompt-based federated RGB-X tracking on a frozen [ViPT](https://github.com/jiawen-zhu/ViPT) tracker. Only the prompt
  parameters (about 0.84 M of 93 M) are trained and exchanged.
- Twelve simulated clients hold RGB-T (LasHeR), RGB-D (DepthTrack) and RGB-E (VisEvent) videos; each dataset has three
  sensor-equipped clients and one RGB-only client.
- The server aggregates each group of prompt parameters in the domain that fits its role: RGB-shared parameters use one
  global bank, auxiliary-specific, fusion and normalisation parameters use one bank per sensor.
- Selective participation: RGB-only clients still contribute to the RGB-shared bank, but their updates to the
  modality-related groups are left out of the sensor banks.

## Framework

<p align="center"><img src="assets/figures/framework.png" width="92%"></p>

More figures from the paper are in [`docs/FIGURES.md`](docs/FIGURES.md).

## Main Results

Mean ± sample standard deviation over three seeds (42, 0, 123) at communication round 120, one fixed client partition.
Methods marked † are personalised: each seed first averages the scores of the three sensor-equipped client models of a
benchmark. Bold = best mean in the column.

| Method | LasHeR (RGB-T) AUC (%) | VisEvent (RGB-E) SR (%) | DepthTrack (RGB-D) F-score (%) |
|---|---|---|---|
| Local † | 47.34 ± 0.31 | 52.27 ± 0.25 | 49.91 ± 1.46 |
| FedAvg | 46.55 ± 0.99 | 52.93 ± 0.59 | 53.85 ± 0.73 |
| FedProx | 46.78 ± 0.67 | 53.11 ± 0.66 | 54.06 ± 0.64 |
| FedPer † | 47.35 ± 0.54 | 54.47 ± 0.36 | 50.81 ± 0.68 |
| Ditto † | 46.46 ± 0.86 | 54.14 ± 0.42 | 53.90 ± 0.58 |
| FedMSplit-P † | 46.43 ± 0.33 | 54.85 ± 0.35 | 54.24 ± 0.89 |
| FED-PRIME † | 46.78 ± 0.45 | 52.09 ± 0.26 | 54.32 ± 0.17 |
| Independent FL | 48.25 ± 0.65 | **56.45 ± 0.83** | 55.28 ± 1.35 |
| **Fed-MAPTrack (ours)** | **48.90 ± 0.63** | 56.38 ± 0.41 | **55.96 ± 0.50** |

Ablation of the aggregation design (same protocol; RGB-shared parameters are globally aggregated in every row):

| Variant | Auxiliary bucket | Fusion/Norm bucket | Selective participation | RGB-T AUC (%) | RGB-E SR (%) | RGB-D F-score (%) |
|---|:-:|:-:|:-:|---|---|---|
| FedAvg | | | | 46.55 ± 0.99 | 52.93 ± 0.59 | 53.85 ± 0.73 |
| Auxiliary-only bucketing | ✓ | | | 47.47 ± 1.24 | **56.46 ± 0.37** | **56.02 ± 0.32** |
| Extended bucketing without selective participation | ✓ | ✓ | | 43.70 ± 0.95 | 54.21 ± 0.27 | 55.31 ± 0.57 |
| **Fed-MAPTrack** | ✓ | ✓ | ✓ | **48.90 ± 0.63** | 56.38 ± 0.41 | 55.96 ± 0.50 |

FedMSplit-P and FED-PRIME are adapted from image-text classification to this tracker, see
[`docs/NEW_BASELINES.md`](docs/NEW_BASELINES.md). Additional DepthTrack-only runs are in
[`docs/EXTRA_RESULTS.md`](docs/EXTRA_RESULTS.md).

## Installation

```
conda create -n fedmap python=3.10
conda activate fedmap
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

Tested with Python 3.10, PyTorch 2.5.1 (CUDA 12.4) on one RTX 4080 SUPER / RTX 4090. DepthTrack F-score is computed with
the [VOT toolkit](https://github.com/votchallenge/toolkit) (`pip install vot-toolkit`, tested with 0.5.3) in a separate
environment.

## Data Preparation

Download LasHeR, DepthTrack and VisEvent yourself and arrange them as below (`CVR_DATA` points to the `data` directory,
default `./data`). Download the pretrained foundation tracker `OSTrack_ep0300.pth.tar` from the
[ViPT release folder](https://drive.google.com/drive/folders/1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy?usp=sharing) into
`pretrained/`.

```
data
-- lasher/trainingset        (LasHeR training sequences)
-- depthtrack/train          (DepthTrack training sequences)
-- visevent/train_subset     (VisEvent training sequences, with trainlist.txt)
-- lasher/testingset, depthtrack/test, visevent/test_subset   (test sets)
```

The client partition used in the paper is `configs/fed/client_splits/lasher_depthtrack_visevent_seed42_v2.json`.

## Training

```
python lib/federated/server.py --variant modality --bucket-groups aux_specific,fusion,norm --selective-participation \
    --rounds 120 --local-steps 50 --workers 0 --clients-per-round 1.0 --seed 42 --save-every 20 --export-at-end \
    --out-dir output/fed/fed_maptrack_s42
```

This is Fed-MAPTrack. The baselines and ablations of the tables are other `--variant` values of the same script, listed in
[`docs/USAGE.md`](docs/USAGE.md). A 120-round run takes about 17-20 hours on one RTX 4080 SUPER / 4090.

## Testing

Exported checkpoints load into the unchanged ViPT tracker; evaluate the final round only.

- LasHeR: `RGBT_workspace/test_rgbt_mgpus.py`, then `scripts/eval/lasher_eval.py`
- VisEvent: `RGBE_workspace/test_rgbe_mgpus.py`, then `scripts/eval/visevent_eval.py`
- DepthTrack: `Depthtrack_workspace/run_offline_depthtrack.py`, then the VOT toolkit's analysis

Details (including the configs for FED-PRIME checkpoints) are in [`docs/USAGE.md`](docs/USAGE.md).

## Citation

The BibTeX entry will be added after publication.

## Acknowledgments

- This code base is built on [ViPT](https://github.com/jiawen-zhu/ViPT) and [OSTrack](https://github.com/botaoye/OSTrack).
- FedMSplit-P and FED-PRIME follow Phung et al., *Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal
  Client Data* (ICCV 2025) and Chen and Zhang, *FedMSplit* (KDD 2022).

## License

MIT, see [`LICENSE`](LICENSE).
