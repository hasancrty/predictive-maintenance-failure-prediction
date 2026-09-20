import numpy as np
import pandas as pd

from src.features import HORIZON, build_dataset, feature_columns, future_failure_label
from src.generate_data import generate
from src.modeling import best_threshold, time_folds, total_cost


def test_future_failure_label_window():
    failure = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    label = future_failure_label(failure, horizon=3)
    # t=0,1,2 -> arıza (t, t+3] içinde; t=3 (arıza günü) -> sonraki 3 günde arıza yok
    assert label[:3].tolist() == [1, 1, 1]
    assert label[3:7].tolist() == [0, 0, 0, 0]
    assert np.isnan(label[7:]).all()   # ufku aşan son 3 gün bilinmiyor


def test_generated_data_shape_and_rate():
    raw = generate(seed=3, n_machines=20, n_days=365)
    assert {"machine_id", "day", "vibration", "temperature", "pressure", "load", "failure"} <= set(raw.columns)
    assert len(raw) == 20 * 365
    data = build_dataset(raw, HORIZON)
    assert 0.02 < data["label"].mean() < 0.12
    assert not data[feature_columns(data)].isna().any().any()


def test_features_do_not_use_future():
    """Bir günün özellikleri, o günden sonraki sensör değerleri değiştirilse de aynı kalmalı."""
    raw = generate(seed=5, n_machines=5, n_days=200)
    base = build_dataset(raw, HORIZON)
    tampered = raw.copy()
    late = (tampered["machine_id"] == 0) & (tampered["day"] > 120)
    tampered.loc[late, ["vibration", "temperature", "pressure"]] += 100.0
    changed = build_dataset(tampered, HORIZON)
    feats = feature_columns(base)
    a = base[(base["machine_id"] == 0) & (base["day"] <= 120)].set_index("day")[feats]
    b = changed[(changed["machine_id"] == 0) & (changed["day"] <= 120)].set_index("day")[feats]
    pd.testing.assert_frame_equal(a, b)


def test_time_folds_have_no_overlap():
    days = np.repeat(np.arange(200), 5)
    for tr, va in time_folds(days, n_splits=4, gap_days=7):
        assert days[tr].max() + 7 < days[va].min()   # eğitim etiketleri doğrulama dönemine sızmaz


def test_threshold_minimizes_cost():
    y = np.array([1, 1, 0, 0, 0, 0])
    proba = np.array([0.9, 0.6, 0.55, 0.3, 0.2, 0.1])
    t = best_threshold(y, proba, thresholds=np.array([0.05, 0.5, 0.58, 0.95]), cost_missed=1000, cost_alarm=100)
    # 0.58: iki pozitif alarm (200) ve 0 kaçırma -> en ucuz
    assert t == 0.58
    assert total_cost(y, proba >= t, 1000, 100) == 200
