# -*- coding: utf-8 -*-
"""Federated training layer for ViPT.

Lives outside the official training/eval path on purpose: nothing in
lib/models, lib/train or lib/test imports this package, so the official
single-machine entry points keep working unchanged (project hard constraint).
"""

from .param_groups import (  # noqa: F401
    AUX_SPECIFIC,
    FROZEN,
    FUSION,
    MODALITY_SENSITIVE_GROUPS,
    NORM,
    RGB_SHARED,
    TRAINABLE_GROUPS,
    UnknownPromptParam,
    build_param_groups,
    format_summary,
    group_of,
    group_to_keys,
    is_modality_sensitive,
    summarize,
    trainable_keys,
)
from .aggregator import (  # noqa: F401
    NO_AUX,
    AggregationStats,
    Aggregator,
    ClientUpdate,
    GlobalState,
    KeyStat,
    make_initial_state,
)
from .fed_wrapper import FedViPTWrapper, KNOWN_AUX  # noqa: F401
from .warmup import (  # noqa: F401
    Phase,
    RoundPlan,
    Schedule,
    WarmupScheduler,
    default_phases,
)

# client_data.py and split_builder.py are deliberately NOT re-exported here.
# Importing either pulls in lib.train.dataset, and with it jpeg4py, cv2 and
# pandas. param_groups.py and aggregator.py are pure dictionary logic with no
# heavy dependency, and their tests run in well under a second because of it.
# Re-exporting the data modules would make `import lib.federated` fail outright
# whenever one of those optional decoders is unhappy, which would take the
# aggregation tests down with it for no reason. Import by full path instead:
#
#     from lib.federated.client_data import build_client_dataset
