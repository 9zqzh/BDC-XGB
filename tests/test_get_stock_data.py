from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from get_stock_data import (
    OUTPUT_COLUMNS,
    build_membership,
    clean_quote,
    fetch_akshare_history,
    fetch_baostock_history,
    filter_by_membership,
    query_latest_available_date,
    quote_cache_needs_update,
    update_quote_cache,
)


def test_filter_by_membership_respects_effective_boundaries() -> None:
    quotes = pd.DataFrame(
        [
            {"股票代码": "000001", "日期": "2018-01-02", **{column: 1 for column in OUTPUT_COLUMNS[2:]}},
            {"股票代码": "000001", "日期": "2018-02-01", **{column: 2 for column in OUTPUT_COLUMNS[2:]}},
            {"股票代码": "000002", "日期": "2018-01-02", **{column: 3 for column in OUTPUT_COLUMNS[2:]}},
        ]
    )
    membership = pd.DataFrame(
        [
            {"股票代码": "000001", "股票名称": "A", "生效日期": "2018-01-01", "失效日期": "2018-01-31"},
            {"股票代码": "000002", "股票名称": "B", "生效日期": "2018-02-01", "失效日期": "2018-12-31"},
        ]
    )

    result = filter_by_membership(quotes, membership)

    assert result["日期"].tolist() == ["2018-01-02"]
    assert result["股票代码"].tolist() == ["000001"]


def test_filter_returns_required_columns_and_sorted_rows() -> None:
    row = {"股票代码": "000002", "日期": "2018-01-03"}
    row.update({column: 1 for column in OUTPUT_COLUMNS[2:]})
    quotes = pd.DataFrame([row])
    membership = pd.DataFrame(
        [{"股票代码": "000002", "股票名称": "B", "生效日期": "2018-01-01", "失效日期": "2018-12-31"}]
    )

    result = filter_by_membership(quotes, membership)

    assert result.columns.tolist() == OUTPUT_COLUMNS
    assert result.iloc[0]["股票代码"] == "000002"


def _quote_row(**overrides: object) -> dict[str, object]:
    row = {"股票代码": "000001", "日期": "2018-01-02"}
    row.update({column: 1 for column in OUTPUT_COLUMNS[2:]})
    row.update(overrides)
    return row


def test_clean_quote_rejects_invalid_rows() -> None:
    with pytest.raises(ValueError, match="invalid stock codes"):
        clean_quote(pd.DataFrame([_quote_row(股票代码="bad")]))
    with pytest.raises(ValueError, match="invalid dates"):
        clean_quote(pd.DataFrame([_quote_row(日期="bad")]))
    with pytest.raises(ValueError, match="non-positive"):
        clean_quote(pd.DataFrame([_quote_row(开盘=0)]))
    with pytest.raises(ValueError, match="duplicate"):
        clean_quote(pd.DataFrame([_quote_row(), _quote_row()]))


def test_fetch_sources_use_unadjusted_data() -> None:
    bs_response = Mock(error_code="0", fields=OUTPUT_COLUMNS)
    bs_response.next.side_effect = [False]
    with patch("get_stock_data.bs.query_history_k_data_plus", return_value=bs_response) as query:
        fetch_baostock_history("000001", "2018-01-01", "2018-01-02")
    assert query.call_args.kwargs["adjustflag"] == "3"

    ak_frame = pd.DataFrame(columns=OUTPUT_COLUMNS)
    with patch("get_stock_data.ak.stock_zh_a_hist", return_value=ak_frame) as query:
        fetch_akshare_history("000001", "2018-01-01", "2018-01-02")
    assert query.call_args.kwargs["adjust"] == ""


def test_query_latest_available_date_uses_latest_returned_bar() -> None:
    response = Mock(error_code="0", error_msg="success", fields=["date"])
    response.next.side_effect = [True, True, False]
    response.get_row_data.side_effect = [["2026-07-29"], ["2026-07-30"]]

    with patch("get_stock_data.bs.query_history_k_data_plus", return_value=response) as query:
        latest = query_latest_available_date(
            "2026-12-31", retries=1, sleep_seconds=0, as_of="2026-07-31"
        )

    assert latest == "2026-07-30"
    assert query.call_args.args[:2] == ("sh.000300", "date")
    assert query.call_args.kwargs["end_date"] == "2026-07-31"
    assert query.call_args.kwargs["adjustflag"] == "3"


def test_quote_cache_staleness_uses_required_end() -> None:
    stale = pd.DataFrame([_quote_row(日期="2026-07-29")])
    current = pd.DataFrame([_quote_row(日期="2026-07-30")])

    assert quote_cache_needs_update(stale, "2026-07-30")
    assert not quote_cache_needs_update(current, "2026-07-30")


def test_update_quote_cache_fetches_only_missing_tail(tmp_path: Path) -> None:
    cache_path = tmp_path / "quotes" / "000001.csv"
    cache_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [_quote_row(日期="2026-07-28"), _quote_row(日期="2026-07-29")]
    ).to_csv(cache_path, index=False)
    fresh = pd.DataFrame([_quote_row(日期="2026-07-30", 收盘=2)])

    with patch("get_stock_data.fetch_baostock_history", return_value=fresh) as fetch:
        result, fallback_used = update_quote_cache(
            code="000001",
            cache_path=cache_path,
            start_date="2018-01-01",
            required_end="2026-07-30",
            force_refresh=False,
            no_akshare_fallback=True,
            max_retries=1,
            sleep_seconds=0,
        )

    fetch.assert_called_once_with("000001", "2026-07-30", "2026-07-30")
    assert result["日期"].tolist() == ["2026-07-28", "2026-07-29", "2026-07-30"]
    assert result.iloc[-1]["收盘"] == 2
    assert fallback_used is False
    persisted = pd.read_csv(cache_path, dtype={"股票代码": str})
    assert persisted["日期"].max() == "2026-07-30"


def test_update_quote_cache_skips_network_when_current(tmp_path: Path) -> None:
    cache_path = tmp_path / "000001.csv"
    pd.DataFrame([_quote_row(日期="2026-07-30")]).to_csv(cache_path, index=False)

    with patch("get_stock_data.fetch_baostock_history") as fetch:
        result, fallback_used = update_quote_cache(
            code="000001",
            cache_path=cache_path,
            start_date="2018-01-01",
            required_end="2026-07-30",
            force_refresh=False,
            no_akshare_fallback=True,
            max_retries=1,
            sleep_seconds=0,
        )

    fetch.assert_not_called()
    assert result["日期"].max() == "2026-07-30"
    assert fallback_used is False


def test_update_quote_cache_allows_partial_historical_tail(tmp_path: Path) -> None:
    cache_path = tmp_path / "000001.csv"
    pd.DataFrame([_quote_row(日期="2025-03-04")]).to_csv(cache_path, index=False)
    fresh = pd.DataFrame([_quote_row(日期="2025-03-05")])

    with patch("get_stock_data.fetch_baostock_history", return_value=fresh):
        result, fallback_used = update_quote_cache(
            code="000001",
            cache_path=cache_path,
            start_date="2018-01-01",
            required_end="2025-03-07",
            force_refresh=False,
            no_akshare_fallback=True,
            max_retries=1,
            sleep_seconds=0,
            allow_partial_end=True,
        )

    assert result["日期"].tolist() == ["2025-03-04", "2025-03-05"]
    assert fallback_used is False


def test_build_membership_probes_latest_trading_day(tmp_path: Path) -> None:
    constituents = {f"{code:06d}": f"stock-{code}" for code in range(300)}

    with patch("get_stock_data.query_constituents", return_value=constituents) as query:
        membership = build_membership(
            ["2026-07-01", "2026-07-30"],
            tmp_path,
            retries=1,
            sleep_seconds=0,
        )

    queried_days = [call.args[0] for call in query.call_args_list]
    assert queried_days == ["2026-07-01", "2026-07-30"]
    assert len(
        membership[
            (membership["生效日期"] <= "2026-07-30")
            & (membership["失效日期"] >= "2026-07-30")
        ]
    ) == 300
