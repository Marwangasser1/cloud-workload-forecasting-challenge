import numpy as np
import pandas as pd
import pytest

from cwfc.validation import rolling_origin_splits


def test_rolling_origin_splits_are_temporal():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC")
    })
    splits = rolling_origin_splits(frame, horizon=4, min_train_size=8, step=4)
    assert len(splits) == 3
    assert splits[0].train_end < splits[0].validation_start
    assert splits[0].validation_end < splits[1].validation_start


@pytest.fixture
def int_frame() -> pd.DataFrame:
    return pd.DataFrame({"timestamp": np.arange(110, dtype="int64") * 300})


def test_110_integer_timestamps_give_four_folds(int_frame):
    assert int_frame["timestamp"].iloc[-1] == 32700
    splits = rolling_origin_splits(int_frame, horizon=24, min_train_size=48, step=12)
    assert len(splits) == 4
    assert [s.train_end_idx for s in splits] == [48, 60, 72, 84]
    assert all(s.val_end_idx - s.val_start_idx == 24 for s in splits)
    assert all(s.val_start_idx == s.train_end_idx for s in splits)
    assert splits[-1].val_end_idx <= len(int_frame)


def test_integer_timestamps_are_preserved_not_converted_to_datetime(int_frame):
    for s in rolling_origin_splits(int_frame, horizon=24, min_train_size=48, step=12):
        for value in (s.train_end, s.validation_start, s.validation_end):
            assert isinstance(value, (int, np.integer))
            assert not isinstance(value, (pd.Timestamp, np.datetime64))
            assert np.asarray(value).dtype == np.int64
            assert value % 300 == 0  # still raw seconds, not nanoseconds


def test_split_boundaries_match_positions(int_frame):
    ts = int_frame["timestamp"]
    for s in rolling_origin_splits(int_frame, horizon=24, min_train_size=48, step=12):
        assert s.train_end == ts.iloc[s.train_end_idx - 1]
        assert s.validation_start == ts.iloc[s.val_start_idx]
        assert s.validation_end == ts.iloc[s.val_end_idx - 1]
        assert s.validation_end - s.validation_start == 23 * 300


def test_train_strictly_precedes_test(int_frame):
    splits = rolling_origin_splits(int_frame, horizon=24, min_train_size=48, step=12)
    for s in splits:
        train = int_frame.iloc[: s.train_end_idx]
        test = int_frame.iloc[s.val_start_idx : s.val_end_idx]
        assert train["timestamp"].max() < test["timestamp"].min()
        assert train["timestamp"].max() == s.train_end
        assert test["timestamp"].min() == s.validation_start
        assert test["timestamp"].max() == s.validation_end
        assert len(train) >= 48 and len(test) == 24


def test_expanding_window_and_ordered_folds(int_frame):
    splits = rolling_origin_splits(int_frame, horizon=24, min_train_size=48, step=12)
    train_sizes = [s.train_end_idx for s in splits]
    assert train_sizes == sorted(train_sizes) and len(set(train_sizes)) == len(train_sizes)
    assert all(a.validation_start < b.validation_start for a, b in zip(splits, splits[1:]))


def test_default_step_equals_horizon(int_frame):
    splits = rolling_origin_splits(int_frame, horizon=24, min_train_size=48)
    assert [s.train_end_idx for s in splits] == [48, 72]


def test_not_enough_rows_gives_no_folds():
    frame = pd.DataFrame({"timestamp": np.arange(60, dtype="int64") * 300})
    assert rolling_origin_splits(frame, horizon=24, min_train_size=48) == []


def test_unsorted_timestamps_rejected():
    frame = pd.DataFrame({"timestamp": [0, 600, 300, 900] * 20})
    with pytest.raises(ValueError):
        rolling_origin_splits(frame, horizon=4, min_train_size=8)


@pytest.mark.parametrize(
    "kwargs",
    [{"horizon": 0}, {"min_train_size": 0}, {"horizon": -1}, {"step": 0}, {"step": -2}],
)
def test_invalid_parameters_rejected(int_frame, kwargs):
    with pytest.raises(ValueError):
        rolling_origin_splits(int_frame, **kwargs)
