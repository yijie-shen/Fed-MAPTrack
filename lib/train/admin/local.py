class EnvironmentSettings:
    def __init__(self):
        # 2026-09-10:把原作者的 /media/jiawen/Datasets/... 换成本机路径。
        # 这个文件本就是官方留给用户填路径的(README 有说明),不算改源码逻辑。
        self.workspace_dir = r'D:\cvresearch\ViPT'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = r'D:\cvresearch\ViPT\tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = r'D:\cvresearch\ViPT\pretrained_networks'

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
        self.depthtrack_dir = r'D:\cvresearch\data\depthtrack\train'      # 152 个序列(146 train + 6 val)
        self.lasher_dir = r'D:\cvresearch\data\lasher\trainingset'        # 979 个序列(ViPT 用 882 train + 97 val)
        # ⚠️ VisEvent 解压出的顶层目录叫 train_subset,不是官方 README 写的 train。
        #    这一层里必须有 trainlist.txt(归档里没有,由 scripts/extract_queue.py 生成)。
        self.visevent_dir = r'D:\cvresearch\data\visevent\train_subset'   # 500 个序列
