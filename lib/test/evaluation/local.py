from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # 2026-09-17: replaced the author's /media/jiawen/... paths with local ones.
    # This file is the test-side counterpart of lib/train/admin/local.py and is
    # what the official README tells users to edit, so this is configuration,
    # not a code change.

    settings.prj_dir = r'D:\cvresearch\ViPT'
    settings.save_dir = r'D:\cvresearch\ViPT\output'
    settings.network_path = r'D:\cvresearch\ViPT\output\test\networks'          # Where tracking networks are stored.
    settings.results_path = r'D:\cvresearch\ViPT\output\test\tracking_results'  # Where to store tracking results
    settings.result_plot_path = r'D:\cvresearch\ViPT\output\test\result_plots'
    settings.segmentation_path = r'D:\cvresearch\ViPT\output\test\segmentation_results'

    # Datasets this project does not use. Kept as fields (the EnvSettings
    # consumers expect them to exist) but pointed at local placeholders so a
    # stray use fails with a path under D:\cvresearch rather than a confusing
    # Linux path from the author's machine.
    settings.davis_dir = ''
    settings.got10k_lmdb_path = r'D:\cvresearch\data\got10k_lmdb'
    settings.got10k_path = r'D:\cvresearch\data\got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = r'D:\cvresearch\data\itb'
    settings.lasot_extension_subset_path_path = r'D:\cvresearch\data\lasot_extension_subset'
    settings.lasot_lmdb_path = r'D:\cvresearch\data\lasot_lmdb'
    settings.lasot_path = r'D:\cvresearch\data\lasot'
    settings.nfs_path = r'D:\cvresearch\data\nfs'
    settings.otb_path = r'D:\cvresearch\data\otb'
    settings.tc128_path = r'D:\cvresearch\data\TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = r'D:\cvresearch\data\tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = r'D:\cvresearch\data\trackingnet'
    settings.uav_path = r'D:\cvresearch\data\uav'
    settings.vot18_path = r'D:\cvresearch\data\vot2018'
    settings.vot22_path = r'D:\cvresearch\data\vot2022'
    settings.vot_path = r'D:\cvresearch\data\VOT2019'
    settings.youtubevos_dir = ''

    return settings
