# Partition sensitivity and centralised reference (DepthTrack only, one seed)

DepthTrack F-score (%), seed 42, a federation of only the four DepthTrack clients (3 depth sensors + 1 RGB-only), alpha =
Dirichlet concentration of the sequence partition (smaller = more skewed). One training run per cell.

| Method | alpha = 0.1 | alpha = 0.5 | alpha = 1.0 |
|---|---|---|---|
| FedAvg | 53.81 | 56.83 (a) | 57.38 |
| Fed-MAPTrack | 54.93 | 55.47 | 53.86 |
| Centralised reference (one client, all 146 training sequences, same total number of steps) | 57.40 | | |

(a) All cells of this table were trained and evaluated on the same RTX 4080 SUPER. An earlier run of the same configuration
(the independent-FL run of the main comparison, which is equivalent to FedAvg in a single-modality federation) was trained
on an RTX 4090 and scored 54.02. The two differ by 2.81 points for the same configuration, more than the three-seed
standard deviations of the main tables (0.5-1.4), so differences of about three points or less in this single-run table
should not be interpreted. The share of RGB-only data in the
four-client federation changes with alpha (0.5%, 3.0% and 38.5% of the frames for alpha = 0.1, 0.5 and 1.0).

## How the runs were made

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
