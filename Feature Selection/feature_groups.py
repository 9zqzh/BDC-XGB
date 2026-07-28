"""Feature-family definitions for the current 158+39 feature scheme."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "code" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train import feature_columns_map  # noqa: E402

GROUP_NAMES = (
    "volume_liquidity",
    "range_breakout",
    "momentum_trend",
    "volatility_risk",
    "other",
)

_EXACT_GROUPS = {
    "volume_liquidity": {
        "成交量", "成交额", "换手率", "volume_change", "obv", "volume_ma_5",
        "volume_ma_20", "volume_ratio", "VMA5", "VMA10", "VMA20", "VMA30",
        "VMA60", "VSTD5", "VSTD10", "VSTD20", "VSTD30", "VSTD60", "WVMA5",
        "WVMA10", "WVMA20", "WVMA30", "WVMA60", "CORR5", "CORR10", "CORR20",
        "CORR30", "CORR60", "CORD5", "CORD10", "CORD20", "CORD30", "CORD60",
        "VSUMP5", "VSUMP10", "VSUMP20", "VSUMP30", "VSUMP60", "VSUMN5",
        "VSUMN10", "VSUMN20", "VSUMN30", "VSUMN60", "VSUMD5", "VSUMD10",
        "VSUMD20", "VSUMD30", "VSUMD60",
    },
    "range_breakout": {
        "MAX5", "MAX10", "MAX20", "MAX30", "MAX60", "MIN5", "MIN10", "MIN20",
        "MIN30", "MIN60", "QTLU5", "QTLU10", "QTLU20", "QTLU30", "QTLU60",
        "QTLD5", "QTLD10", "QTLD20", "QTLD30", "QTLD60", "RANK5", "RANK10",
        "RANK20", "RANK30", "RANK60", "RSV5", "RSV10", "RSV20", "RSV30",
        "RSV60", "IMAX5", "IMAX10", "IMAX20", "IMAX30", "IMAX60", "IMIN5",
        "IMIN10", "IMIN20", "IMIN30", "IMIN60", "IMXD5", "IMXD10", "IMXD20",
        "IMXD30", "IMXD60",
    },
    "momentum_trend": {
        "涨跌幅", "return_1", "return_5", "return_10", "ROC5", "ROC10", "ROC20",
        "ROC30", "ROC60", "MA5", "MA10", "MA20", "MA30", "MA60", "sma_5",
        "sma_20", "ema_12", "ema_26", "ema_60", "rsi", "macd", "macd_signal",
        "BETA5", "BETA10", "BETA20", "BETA30", "BETA60", "RESI5", "RESI10",
        "RESI20", "RESI30", "RESI60", "CNTP5", "CNTP10", "CNTP20", "CNTP30",
        "CNTP60", "CNTN5", "CNTN10", "CNTN20", "CNTN30", "CNTN60", "CNTD5",
        "CNTD10", "CNTD20", "CNTD30", "CNTD60", "SUMP5", "SUMP10", "SUMP20",
        "SUMP30", "SUMP60", "SUMN5", "SUMN10", "SUMN20", "SUMN30", "SUMN60",
        "SUMD5", "SUMD10", "SUMD20", "SUMD30", "SUMD60",
    },
    "volatility_risk": {
        "振幅", "STD5", "STD10", "STD20", "STD30", "STD60", "RSQR5", "RSQR10",
        "RSQR20", "RSQR30", "RSQR60", "boll_std", "atr_14", "volatility_10",
        "volatility_20", "high_low_spread", "open_close_spread", "high_close_spread",
        "low_close_spread",
    },
}


def get_feature_columns(feature_num: str = "158+39") -> list[str]:
    """Return a copy of the project's configured base feature list."""
    return list(feature_columns_map[feature_num])


def build_feature_groups(feature_num: str = "158+39") -> dict[str, list[str]]:
    """Assign every configured feature to exactly one family."""
    features = get_feature_columns(feature_num)
    groups = {name: [] for name in GROUP_NAMES}
    assigned: set[str] = set()
    for group_name in GROUP_NAMES[:-1]:
        for feature in features:
            if feature in _EXACT_GROUPS[group_name] and feature not in assigned:
                groups[group_name].append(feature)
                assigned.add(feature)
    groups["other"] = [feature for feature in features if feature not in assigned]
    return groups


def validate_feature_groups(groups: dict[str, list[str]], features: list[str]) -> None:
    """Raise when a group mapping misses or duplicates a configured feature."""
    flattened = [feature for values in groups.values() for feature in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("因子族定义存在重复特征")
    if set(flattened) != set(features):
        missing = sorted(set(features) - set(flattened))
        extra = sorted(set(flattened) - set(features))
        raise ValueError(f"因子族定义不完整: missing={missing}, extra={extra}")
