# FedMSplit-P and FED-PRIME: how they are adapted to federated RGB-X tracking

Both baselines come from Phung et al., *Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data*
(ICCV 2025, code: github.com/hangpt01/FedPrime). The original methods work on a frozen ViLT for image-text classification
with missing modalities. We re-implemented them on the frozen ViPT tracker. They are **adapted references, not exact
reproductions**. All hyperparameters are the paper's or the official implementation's defaults and were not tuned.

Both are personalised: every client keeps its own model and the evaluation scores each sensor client's checkpoint
(`ViPTrack_<client>_ep0120.pth.tar`), then averages the three clients of a benchmark.

## FedMSplit-P (`--variant fedmsplit_p`)

* Each client keeps its own model and uploads all trainable tensors every round.
* The model is cut into blocks (the four ViPT parameter groups). For every block the server computes
  `A[k,l] = softmax_l(<x_k, x_l> / sqrt(d))` over the clients' flattened block vectors and sets client k's new block to
  `sum_l w_kl x_l / sum_l w_kl` with `w_kl = A_b[k,l] * mean_b' A_b'[k,l]`.
* Local training adds a proximal term (lambda = 0.01, the official default) towards the data-volume-weighted average of the uploads.
* Adaptations: blocks are ViPT's groups (the original uses prompts, pooler and classifier); an RGB-only client has no valid
  auxiliary tensors, so it takes no part in the `aux_specific` block and always receives the pretrained auxiliary tensors.
* Observation: with the original scaling the similarity weights are almost uniform (self-weight 0.1114 for nine clients,
  uniform would be 0.1111), so the method behaves close to an equal-weight average plus the proximal term.

## FED-PRIME (`--variant fedprime`)

* Every client owns two pools of 20 prompt tokens (768-d): an intra-client pool that never leaves the client and an
  inter-client pool that is aligned on the server, plus a bias-free 768x768 query projection (local).
* Per batch: query = max over the detached RGB patch embeddings of template and search region, projected and L2-normalised;
  keys are the L2-normalised pool rows. Each sample picks its top five rows per pool, the batch keeps the five most
  frequently picked rows per pool, and these ten tokens are prepended to the transformer input and removed before the head.
  Candidate elimination and ViPT's per-layer prompt blocks skip the prefix.
* Loss = ViPT loss - 0.01 * (mean summed cosine similarity of the selected rows to the query).
* Server: the four ViPT groups use the same FedAvg code path as `--variant fedavg` (sample-weighted); the intra pool and
  query projection are not aggregated; the inter pool is aligned by probabilistic matching: 20 centres, Hungarian one-to-one
  matching of each client's used prompts to the centres under a Gaussian likelihood whose std comes from a small network,
  plus a learned centre-popularity term, five outer loops, Adam lr 1e-3 for the two small networks.
* New trainable parameters: 620,544 (two pools and the query projection), 74% on top of ViPT's 839,628.
* Evaluating a FED-PRIME checkpoint needs a model with the pools: use `experiments/vipt/deep_*_prime.yaml`.

### Deviations from the original method

1. ViPT instead of ViLT; the token pools are new trainable parameters of the tracker.
2. The prefix tokens sit at the front of the sequence, so candidate elimination and the deep prompt blocks were changed to
   skip them (`n_prefix`); with the pools switched off the model is bit-identical to the original.
3. The query comes from the RGB patch embeddings of template and search region (the original uses text and image embeddings).
4. The relevance term is subtracted (pull selected prompts towards the input, as the paper states); the official code adds it.
5. Pools are initialised with small random values (std 0.02); the official code uses zeros, which makes row normalisation meaningless.
6. The server keeps exactly 20 centres (the original lets the pool grow up to n x 20 and prunes empty centres); a centre nobody
   was matched to keeps its previous value.
7. Centres are updated in closed form (previous centre and matched prompts averaged); the original moves them with the same
   single Adam step as the networks, which barely moves 768-d prompts of scale 0.02.
8. The intra-client pool stays on the client as the paper describes; the official code also averages it.

## Cost

Per sensor client and round, both methods exchange the 839,628 ViPT prompt parameters (same as FedAvg); FED-PRIME adds at
most 20 x 768 inter-pool parameters. The ten extra tokens lengthen the sequence from 320 to 330 and did not change the
round time measurably (about 530 s per round for 12 clients x 50 local steps on an RTX 4080 SUPER).
