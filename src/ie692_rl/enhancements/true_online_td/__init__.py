"""Enhancement 2: True Online TD(lambda) policy evaluation (with TD(0) and TD(lambda) for comparison)."""

from ie692_rl.enhancements.true_online_td.features import FeatureMap, linear_features, tabular_features
from ie692_rl.enhancements.true_online_td.td import (
    TDResult,
    TrueOnlineTD,
    td_lambda,
    td_zero,
    true_online_td_lambda,
)

__all__ = [
    "FeatureMap",
    "TDResult",
    "TrueOnlineTD",
    "linear_features",
    "tabular_features",
    "td_lambda",
    "td_zero",
    "true_online_td_lambda",
]
