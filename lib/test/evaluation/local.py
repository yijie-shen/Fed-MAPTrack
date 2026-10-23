import os
from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Test-side counterpart of lib/train/admin/local.py (the official README tells users to edit it).
    # CVR_DATA (default: <repo>/data) holds the datasets.

    # The project root is this checkout, so the model code that loads a checkpoint, the
    # experiments/vipt/<tag>.yaml it reads and output/checkpoints/train/vipt/<tag>/ all come from one tree.
    workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    data = os.environ.get('CVR_DATA', os.path.join(os.environ.get('CVR_ROOT', workspace), 'data'))
    settings.prj_dir = workspace
    settings.save_dir = os.path.join(workspace, 'output')
    settings.network_path = os.path.join(workspace, 'output', 'test', 'networks')          # Where tracking networks are stored.
    settings.results_path = os.path.join(workspace, 'output', 'test', 'tracking_results')  # Where to store tracking results
    settings.result_plot_path = os.path.join(workspace, 'output', 'test', 'result_plots')
    settings.segmentation_path = os.path.join(workspace, 'output', 'test', 'segmentation_results')

    # Datasets this project does not use are kept as fields (EnvSettings consumers expect them) with placeholder paths.
    settings.davis_dir = ''
    settings.got10k_lmdb_path = os.path.join(data, 'got10k_lmdb')
    settings.got10k_path = os.path.join(data, 'got10k')
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = os.path.join(data, 'itb')
    settings.lasot_extension_subset_path_path = os.path.join(data, 'lasot_extension_subset')
    settings.lasot_lmdb_path = os.path.join(data, 'lasot_lmdb')
    settings.lasot_path = os.path.join(data, 'lasot')
    settings.nfs_path = os.path.join(data, 'nfs')
    settings.otb_path = os.path.join(data, 'otb')
    settings.tc128_path = os.path.join(data, 'TC128')
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = os.path.join(data, 'tnl2k')
    settings.tpl_path = ''
    settings.trackingnet_path = os.path.join(data, 'trackingnet')
    settings.uav_path = os.path.join(data, 'uav')
    settings.vot18_path = os.path.join(data, 'vot2018')
    settings.vot22_path = os.path.join(data, 'vot2022')
    settings.vot_path = os.path.join(data, 'VOT2019')
    settings.youtubevos_dir = ''

    return settings
