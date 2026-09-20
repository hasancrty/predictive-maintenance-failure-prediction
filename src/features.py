"""Özellik mühendisliği ve etiketleme.

Sızıntı (leakage) önlemi: tüm pencereler yalnızca geçmişe bakar (kayan pencere, `t` dahil).
Pencereler onarımdan sonra sıfırlanır, böylece arıza öncesi okumalar yeni döngüye karışmaz.
Etiket, `t` gününden SONRAKİ H gün içinde arıza olup olmadığıdır; özellikler bu bilgiyi içermez.
"""
import numpy as np
import pandas as pd

SENSORS = ["vibration", "temperature", "pressure"]
HORIZON = 7   # kaç gün içindeki arıza tahmin ediliyor


def future_failure_label(failure: np.ndarray, horizon: int) -> np.ndarray:
    """Her gün için (t, t+horizon] aralığında arıza var mı? Ufku aşan son günler için NaN."""
    n = len(failure)
    cum = np.concatenate([[0], np.cumsum(failure)])
    label = np.full(n, np.nan)
    t = np.arange(n)
    valid = t + horizon <= n - 1
    label[valid] = (cum[t[valid] + horizon + 1] - cum[t[valid] + 1] > 0).astype(float)
    return label


def build_dataset(raw: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    df = raw.sort_values(["machine_id", "day"]).reset_index(drop=True)

    # Döngü numarası: satırdan önceki arıza sayısı (arıza günü eski döngüye ait)
    df["cycle"] = df.groupby("machine_id")["failure"].cumsum() - df["failure"]
    grp = df.groupby(["machine_id", "cycle"])
    df["age"] = grp.cumcount()   # son onarımdan (veya veri başından) beri geçen gün

    for s in SENSORS:
        g = grp[s]
        df[f"{s}_mean3"] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f"{s}_mean7"] = g.transform(lambda x: x.rolling(7, min_periods=1).mean())
        df[f"{s}_std7"] = g.transform(lambda x: x.rolling(7, min_periods=2).std()).fillna(0.0)
        mean14 = g.transform(lambda x: x.rolling(14, min_periods=1).mean())
        df[f"{s}_trend"] = df[f"{s}_mean3"] - mean14   # kısa ortalama - uzun ortalama: yükseliş eğilimi

    df["label"] = np.concatenate([
        future_failure_label(g["failure"].to_numpy(), horizon) for _, g in df.groupby("machine_id", sort=True)
    ])
    # Arıza günü makine duruyor (karar verilecek bir şey yok) ve son H günün etiketi bilinmiyor
    df = df[(df["failure"] == 0) & df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    return df.sort_values(["day", "machine_id"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    derived = [c for c in df.columns if any(c.startswith(s + "_") for s in SENSORS)]
    return ["age", "load", *SENSORS, *derived]
