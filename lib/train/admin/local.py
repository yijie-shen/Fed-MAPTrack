import os

class EnvironmentSettings:
    def __init__(self):
        # 2026-09-10:把原作者的 /media/jiawen/Datasets/... 换成本机路径。
        # 这个文件本就是官方留给用户填路径的(README 有说明),不算改源码逻辑。
        root = os.environ.get('CVR_ROOT', r'D:\cvresearch')
        data = os.environ.get('CVR_DATA', os.path.join(root, 'data'))
        self.workspace_dir = os.path.join(root, 'ViPT')
        self.tensorboard_dir = os.path.join(self.workspace_dir, 'tensorboard')
        self.pretrained_networks = os.path.join(self.workspace_dir, 'pretrained_networks')

        # 本项目用不到的数据集,保留字段但指向本机 data/ 下的占位路径,
        # 免得哪天误用时报出一个 Linux 路径让人困惑。
        self.got10k_val_dir = r'D:\cvresearch\data\got10k\val'
        self.lasot_lmdb_dir = r'D:\cvresearch\data\lasot_lmdb'
        self.got10k_lmdb_dir = r'D:\cvresearch\data\got10k_lmdb'
        self.trackingnet_lmdb_dir = r'D:\cvresearch\data\trackingnet_lmdb'
        self.coco_lmdb_dir = r'D:\cvresearch\data\coco_lmdb'
        self.coco_dir = r'D:\cvresearch\data\coco'
        self.lasot_dir = r'D:\cvresearch\data\lasot'
        self.got10k_dir = r'D:\cvresearch\data\got10k\train'
        self.trackingnet_dir = r'D:\cvresearch\data\trackingnet'

        # ---- 本项目真正用到的三个 ----
        # 各自的 root 都必须是"直接包含序列目录"的那一层,与 SDSTrack 时期完全相同,
        # 因为两个仓库对三个数据集的目录约定一致(已逐一核对)。
        self.depthtrack_dir = os.path.join(data, 'depthtrack', 'train')
        self.lasher_dir = os.path.join(data, 'lasher', 'trainingset')
        # ⚠️ VisEvent 解压出的顶层目录叫 train_subset,不是官方 README 写的 train。
        #    这一层里必须有 trainlist.txt(归档里没有,由 scripts/extract_queue.py 生成)。
        self.visevent_dir = os.path.join(data, 'visevent', 'train_subset')
