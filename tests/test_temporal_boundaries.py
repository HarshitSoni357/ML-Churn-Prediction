from pathlib import Path
import pandas as pd


def test_feature_boundary_excludes_future_event():
    t = pd.Timestamp("2026-01-05 00:00:00")
    events = pd.DataFrame({
        "event_time": pd.to_datetime([
            "2026-01-04 23:59:59",
            "2026-01-05 00:00:00",
            "2026-01-05 00:00:01",
        ]),
        "value": [10, 20, 30],
    })
    features = events.loc[events["event_time"].le(t), "value"].tolist()
    assert features == [10, 20]


def test_label_window_is_open_then_closed():
    t = pd.Timestamp("2026-01-05 00:00:00")
    events = pd.DataFrame({
        "event_time": pd.to_datetime([
            "2026-01-05 00:00:00",
            "2026-01-05 00:00:01",
            "2026-02-04 00:00:00",
            "2026-02-04 00:00:01",
        ])
    })
    labels = events.loc[
        events["event_time"].gt(t)
        & events["event_time"].le(t + pd.Timedelta(days=30))
    ]
    assert len(labels) == 2


def test_generated_dataset_invariants_when_available():
    path = Path("data/processed/weekly_customer_features.csv")
    if not path.exists():
        return
    df = pd.read_csv(path)
    assert not df.duplicated(["customer_id", "prediction_timestamp"]).any()
    assert set(df["inactive_next_30d"].unique()).issubset({0, 1})
    assert (df["orders_count_7d"] <= df["orders_count_30d"]).all()
    assert (df["orders_count_30d"] <= df["orders_count_90d"]).all()
    assert (df["days_since_last_order"] >= 0).all()
