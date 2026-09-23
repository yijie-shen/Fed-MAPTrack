from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.vipt.config import cfg, update_config_from_file


def parameters(yaml_name: str, epoch=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/vipt/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    #
    # 2026-09-17: the official line pointed only at ./models/ViPT_<yaml>.pth,
    # which is where the RELEASED weights go. Our own training writes to
    # output/checkpoints/train/vipt/<yaml>/ViPTrack_ep%04d.pth.tar instead --
    # the line the authors left commented out just above. Evaluating a model we
    # trained ourselves therefore looked for a file that does not exist.
    #
    # Now: prefer our training output when it exists, fall back to the released
    # location. Both stay usable, and the resolved path is printed so the log
    # always records WHICH checkpoint produced the numbers -- otherwise a stale
    # file silently becomes "the result".
    # The `epoch` argument was accepted but never used -- cfg.TEST.EPOCH always
    # won, so the CLI's --epoch flag did nothing. Honour it when given, which is
    # what lets us evaluate an intermediate checkpoint without editing the yaml.
    use_epoch = int(epoch) if epoch is not None else cfg.TEST.EPOCH
    trained = os.path.join(
        save_dir, "checkpoints/train/vipt/%s/ViPTrack_ep%04d.pth.tar"
        % (yaml_name, use_epoch))
    released = os.path.join(prj_dir, "./models/ViPT_%s.pth" % yaml_name)
    params.checkpoint = trained if os.path.isfile(trained) else released
    print("[params] checkpoint -> %s%s" % (
        params.checkpoint,
        "" if os.path.isfile(params.checkpoint) else "   !! FILE NOT FOUND"))
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
