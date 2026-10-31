"""Slow Stage-A checks on the real ViPT model and real client data.

Run with: python lib\\federated\\test_real_path.py
"""
from __future__ import annotations
import json, os, sys, tempfile
from types import SimpleNamespace
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from lib.config.vipt.config import cfg, update_config_from_file
from lib.models.vipt.ostrack_prompt import build_viptrack
from lib.train.base_functions import update_settings
from lib.federated.aggregator import make_initial_state
from lib.federated.client import FederatedClient, seed_everything
from lib.federated.client_data import build_client_dataset
from lib.federated.fed_wrapper import FedViPTWrapper
from lib.federated.param_groups import FROZEN, TRAINABLE_GROUPS
from lib.federated.server import DEFAULT_SPLIT
from lib.federated.warmup import WarmupScheduler
from lib.federated.split_builder import count_frames

PY = torch.cuda.is_available()
DEVICE = "cuda" if PY else "cpu"
SPLIT = DEFAULT_SPLIT

def _setup():
    seed_everything(42)
    update_config_from_file(os.path.join(REPO, "experiments/vipt/repro_rgbt.yaml"))
    cfg.TRAIN.BATCH_SIZE = 4
    cfg.TRAIN.NUM_WORKER = 0
    settings = SimpleNamespace(device=DEVICE, local_rank=-1, num_template=1,
                               num_search=1)
    update_settings(settings, cfg)
    settings.accum_iter = 1
    net = build_viptrack(cfg, training=True).to(DEVICE)
    net.train()
    doc = json.load(open(SPLIT, encoding="utf-8"))
    by_id = {c["client_id"]: c for c in doc["clients"]}
    return net, settings, by_id

def _run(cid, steps=2):
    net, settings, by_id = _setup()
    client = FederatedClient(by_id[cid], cfg, settings, base_seed=42)
    wrapper = FedViPTWrapper(net)
    groups = wrapper.param_groups
    before = {k: v.detach().clone() for k, v in net.state_dict().items()}
    state = make_initial_state(net.state_dict(), groups,
                              tuple(c["aux_type"] for c in by_id.values()
                                    if c["aux_type"] != "none"))
    plan = WarmupScheduler.no_warmup(1).resolve(0)
    seen_bn = []
    hooks = []
    for m in net.box_head.modules():
        if "BatchNorm" in m.__class__.__name__:
            hooks.append(m.register_forward_pre_hook(lambda *_a, _m=m: seen_bn.append(_m.training)))
    update, _ = client.local_train(net, state.view_for(client.aux_type), plan, 0,
                                   steps, device=DEVICE)
    for h in hooks: h.remove()
    if os.environ.get("SAVE_UPLOAD"):
        torch.save({k: v.cpu() for k, v in update.state.items()},
                   os.environ["SAVE_UPLOAD"])
    return net, wrapper, before, update, seen_bn, by_id

def test_fix_bn_and_real_training():
    net, wrapper, before, update, seen_bn, _ = _run("rgbd_003", 2)
    after = net.state_dict()
    frozen = [k for k, g in wrapper.param_groups.items() if g == FROZEN]
    assert all(torch.equal(before[k], after[k]) for k in frozen)
    changed = [k for k, g in wrapper.param_groups.items()
               if g in TRAINABLE_GROUPS and not torch.equal(before[k], after[k])]
    assert changed, "no trainable parameter changed"
    assert seen_bn and not any(seen_bn), sorted(set(seen_bn))

def test_rgb_only_aux_tensors_do_not_change():
    _, wrapper, before, _, _, _ = _run("rgbt_000", 3)
    after = wrapper.net.state_dict()
    aux = [k for k, g in wrapper.param_groups.items() if g == "aux_specific"]
    assert len(aux) == 26
    assert all(torch.equal(before[k], after[k]) for k in aux)

def test_toy07_and_v2_split():
    doc = json.load(open(SPLIT, encoding="utf-8"))
    c = next(x for x in doc["clients"] if x["client_id"] == "rgbd_001")
    ds = build_client_dataset(c)
    idx = ds.sequence_list.index("toy07_indoor_320")
    info = ds.get_sequence_info(idx)
    assert len(info["bbox"]) == 1367
    assert count_frames(os.path.join(os.environ.get("CVR_DATA", os.path.join(os.environ.get("CVR_ROOT", REPO), "data")), "depthtrack", "train", "toy07_indoor_320"), "groundtruth.txt") == 1367
    old = json.load(open(os.path.join(REPO, "configs/fed/client_splits/lasher_depthtrack_visevent_seed42.json"), encoding="utf-8"))
    oldc = next(x for x in old["clients"] if x["client_id"] == "rgbd_001")
    assert oldc["num_samples"] - c["num_samples"] == 39
    assert [x["client_id"] for x in old["clients"] if x["num_samples"] != next(y["num_samples"] for y in doc["clients"] if y["client_id"] == x["client_id"])] == ["rgbd_001"]

def main():
    failures = []
    for name, fn in [("fix_bn", test_fix_bn_and_real_training),
                     ("rgb_only", test_rgb_only_aux_tensors_do_not_change),
                     ("toy07", test_toy07_and_v2_split)]:
        try:
            fn(); print("[OK]", name)
        except Exception as e:
            failures.append((name, repr(e))); print("[FAIL]", name, repr(e))
    if failures: raise SystemExit(1)
    print("3/3 passed", DEVICE)

if __name__ == "__main__": main()
