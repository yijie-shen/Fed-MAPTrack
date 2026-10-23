# Provenance of this code base

The federated layer was developed on separate branches, each producing the results of one group of Table 1 rows. This branch
(`release`) merges them so that every row can be rerun from one tree:

| merged branch | adds |
|---|---|
| `main` (b5bfa29) | local / FedAvg / FedProx / modality banks (`--variant local\|fedavg\|fedprox\|modality\|ours`) |
| `bucketgroups` (e47644b) | `--bucket-groups`, `--selective-participation` (the final method) |
| `fedper` (35650ed) | `--variant fedper` |
| `ditto` (7611cb7) | `--variant ditto` |
| `newbaselines` (0c7a27e) | `--variant fedmsplit_p`, `--variant fedprime`, FED-PRIME token pools in the backbone |
| `newbaselines_eval` | evaluation harness (offline DepthTrack runner, evaluation configs, repository-relative paths) |

## Equivalence check of the merge

For each variant the original branch and the merged tree were run for two rounds (4 local steps, seed 42, three clients:
`rgbt_000`, `rgbt_001`, `rgbd_001`) and compared: the saved federated state (every tensor) and every client's mean loss are
bit-identical for `fedavg`, `modality`, `local`, `fedprox`, `fedper`, `ditto`, the final method
(`modality --bucket-groups aux_specific,fusion,norm --selective-participation`), `fedmsplit_p` and `fedprime`.
The unit tests of all branches (`lib/federated/test_*.py`) pass on the merged tree.
`test_client_data.py` and `test_real_path.py` need the real datasets (`CVR_DATA`).

## Reported numbers

The reported numbers were produced by the branches above before the merge. Training of the two newest baselines ran on
RTX 4090 GPUs, evaluation of all checkpoints on one RTX 4080 SUPER.
