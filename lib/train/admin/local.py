import os


class EnvironmentSettings:
    """Paths used by ViPT's training code.

    Everything is derived from two optional environment variables:
      CVR_ROOT  working directory (default: this repository)
      CVR_DATA  dataset directory (default: $CVR_ROOT/data)
    """

    def __init__(self):
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        root = os.environ.get('CVR_ROOT', repo)
        data = os.environ.get('CVR_DATA', os.path.join(root, 'data'))
        self.workspace_dir = repo
        self.tensorboard_dir = os.path.join(self.workspace_dir, 'tensorboard')
        self.pretrained_networks = os.path.join(self.workspace_dir, 'pretrained_networks')

        # Datasets this project does not use. The fields must exist (the training code reads them) and point to
        # placeholders under data/, so that an accidental use fails with a clear path.
        self.got10k_val_dir = os.path.join(data, 'got10k', 'val')
        self.lasot_lmdb_dir = os.path.join(data, 'lasot_lmdb')
        self.got10k_lmdb_dir = os.path.join(data, 'got10k_lmdb')
        self.trackingnet_lmdb_dir = os.path.join(data, 'trackingnet_lmdb')
        self.coco_lmdb_dir = os.path.join(data, 'coco_lmdb')
        self.coco_dir = os.path.join(data, 'coco')
        self.lasot_dir = os.path.join(data, 'lasot')
        self.got10k_dir = os.path.join(data, 'got10k', 'train')
        self.trackingnet_dir = os.path.join(data, 'trackingnet')

        # The three datasets of the federated benchmark. Each root is the directory that directly contains the
        # sequence folders.
        self.depthtrack_dir = os.path.join(data, 'depthtrack', 'train')
        self.lasher_dir = os.path.join(data, 'lasher', 'trainingset')
        # VisEvent unpacks to `train_subset` (not `train` as in the official README); the directory needs a
        # `trainlist.txt` listing the sequences.
        self.visevent_dir = os.path.join(data, 'visevent', 'train_subset')
