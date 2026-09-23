"""Give one federated client a dataset containing only ITS sequences.

Why this file exists
--------------------
`client_splits.json` assigns sequence NAMES to clients. The three ViPT dataset
classes do not all offer a way to load a name subset:

  lib/train/dataset/lasher.py      __init__(..., seq_ids=None, ...)   <- has it,
                                   but seq_ids are INTEGER INDICES into
                                   data_specs/lasher_train.txt, not names
  lib/train/dataset/depthtrack.py  signature has `vid_ids`/`data_fraction`
                                   COMMENTED OUT -- no subset support
  lib/train/dataset/visevent.py    same, and it reads its sequence list from
                                   <root>/trainlist.txt, not from data_specs/

Three ways to deal with that:

  A. build the official dataset, then replace `sequence_list` in place
  B. add a seq_ids argument to depthtrack.py and visevent.py
  C. write one spec file per client and pass it as `split`

We do (A). It touches ZERO lines of ViPT: every one of the three classes stores
its sequences in `self.sequence_list` and addresses them by index into that
list, so narrowing the list narrows the dataset. (B) would edit two official
files for something only we need; (C) cannot work for VisEvent, whose __init__
asserts `split in ['train']` and looks for the list inside the dataset root.

The one derived structure that must be rebuilt is DepthTrack's
`seq_per_class` / `class_list` / `class_to_id`, which map class names to
positions in `sequence_list` -- stale ones would point at the wrong sequences.
DepthTrack is also the only one of the three with `has_class_info() == True`.
We rebuild by calling the class's own `_build_class_list()`, so the rebuilt
structures cannot drift from what the class expects.

Sequence-name order
-------------------
`split_builder.load_pool()` SORTS names before splitting, so the order in
client_splits.json is alphabetical. The dataset classes read their list files in
FILE order. Indices are therefore resolved against `dataset.sequence_list`
itself, never against a sorted copy.
"""

import os
import sys
from collections import OrderedDict

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lib.federated.split_builder import DATASETS, count_frames  # noqa: E402

#: dataset name -> (class path, the dtype the OFFICIAL trainer uses)
#: These dtypes are copied from lib/train/base_functions.py:36-47. Changing one
#: changes what the 6-channel input contains, so they are pinned here rather
#: than defaulted.
_SPEC = OrderedDict(
    lasher=("LasHeR", "rgbrgb", "train"),
    depthtrack=("DepthTrack", "rgbcolormap", "train"),
    visevent=("VisEvent", "rgbrgb", "train"),
)


class ClientDataError(RuntimeError):
    """Raised instead of silently training on the wrong sequences."""


def build_dataset(name, root=None, dtype=None, split=None):
    """Construct one official ViPT dataset, unrestricted.

    Roots default to split_builder.DATASETS so a client dataset can be built
    without lib/train/admin/local.py being loaded -- handy in tests and in any
    process that is not the official trainer.
    """
    if name not in _SPEC:
        raise ClientDataError(
            "unknown dataset %r; known: %s" % (name, list(_SPEC)))
    cls_name, def_dtype, def_split = _SPEC[name]
    import lib.train.dataset as ds_mod
    cls = getattr(ds_mod, cls_name)
    root = root or DATASETS[name]["root"]
    if not os.path.isdir(root):
        raise ClientDataError("dataset root does not exist: %s" % root)
    return cls(root, dtype=dtype or def_dtype, split=split or def_split)


def restrict_to_sequences(dataset, names, dataset_name=""):
    """Narrow `dataset` in place to exactly `names`, in the order given.

    Fails loudly on a name the dataset does not have. A federated client
    silently training on the wrong subset would corrupt every downstream number
    while looking perfectly healthy, so this must never degrade gracefully.

    Returns the list of indices that were selected (useful for logging).
    """
    have = list(dataset.sequence_list)
    pos = {n: i for i, n in enumerate(have)}
    if len(pos) != len(have):
        dupes = [n for n in pos if have.count(n) > 1]
        raise ClientDataError(
            "%s sequence_list has duplicate names (%d unique of %d): %s"
            % (dataset_name or dataset.get_name(), len(pos), len(have), dupes[:5]))

    missing = [n for n in names if n not in pos]
    if missing:
        raise ClientDataError(
            "%s: %d of %d client sequences are not in the dataset list "
            "(e.g. %s). The split was probably built from a different list "
            "file than the dataset class reads."
            % (dataset_name or dataset.get_name(), len(missing), len(names),
               missing[:5]))

    idx = [pos[n] for n in names]
    dataset.sequence_list = [have[i] for i in idx]

    # Rebuild anything derived from sequence_list. DepthTrack is the only one of
    # the three that has these, but ask rather than assume, so this keeps
    # working if a dataset class gains class info later.
    if hasattr(dataset, "_build_class_list"):
        dataset.seq_per_class, dataset.class_list = dataset._build_class_list()
        dataset.class_list.sort()
        dataset.class_to_id = {c: i for i, c in enumerate(dataset.class_list)}

    return idx


def build_client_dataset(client, root=None, verbose=False):
    """Build the dataset for one entry of client_splits.json.

    `client` is the dict straight out of the json: it must carry `dataset` and
    `sequence_ids`. The client's declared aux_type is NOT applied here -- a
    client with aux_type 'none' still reads the same files; whether the
    auxiliary half is used is decided by FedViPTWrapper's modality mask. Keeping
    those two concerns apart is what lets the same split be reused for the
    modality-missing ablation.
    """
    name = client["dataset"]
    names = list(client["sequence_ids"])
    if not names:
        raise ClientDataError("client %r has no sequences" % client.get("client_id"))

    dataset = build_dataset(name, root=root)
    n_before = len(dataset.sequence_list)
    restrict_to_sequences(dataset, names, dataset_name=name)

    # Official dataset classes derive annotation length from the text file and
    # the failsafe image loader otherwise silently accepts missing tail frames.
    # Clamp the annotation arrays to the common RGB/auxiliary image prefix.
    limits = {}
    cfg = DATASETS[name]
    for idx, seq_name in enumerate(dataset.sequence_list):
        limits[idx] = count_frames(os.path.join(cfg["root"], seq_name), cfg["anno"])
    original_info = dataset.get_sequence_info
    def bounded_info(seq_id, _orig=original_info, _limits=limits):
        info = _orig(seq_id)
        n = _limits.get(seq_id, len(next(iter(info.values()))))
        return {k: v[:n] for k, v in info.items()}
    dataset.get_sequence_info = bounded_info

    if verbose:
        print("  %-12s %-11s %d of %d sequences (%d frames declared)"
              % (client.get("client_id", "?"), name, len(names), n_before,
                 client.get("num_samples", -1)))
    return dataset


def load_split(path):
    """Read client_splits.json and return (document, {client_id: client})."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    by_id = OrderedDict((c["client_id"], c) for c in doc["clients"])
    if len(by_id) != len(doc["clients"]):
        raise ClientDataError("duplicate client_id in %s" % path)
    return doc, by_id
