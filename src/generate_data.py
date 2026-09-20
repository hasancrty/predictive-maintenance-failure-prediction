"""Sentetik makine sensör verisi üretir (60 makine × 365 gün, günlük ortalama okumalar).

Mekanizma: her makinenin gizli bir aşınma durumu (wear) vardır; 0'dan başlar, günlük yükle orantılı artar
ve 1'e ulaşınca arıza olur, onarımla sıfırlanır. Sensörler aşınmayla birlikte bozulur (titreşim ve sıcaklık artar,
basınç düşer). Her döngünün aşınma hızı farklıdır ve döngülerin bir kısmı "ani arıza"dır (sensörler
aşınmayı zayıf yansıtır). Bu yüzden ne yalnızca yaş, ne de sensörler tek başına arızayı mükemmel tahmin eder.

Gerçek veriyle çalışmak için aynı sütunlara sahip bir CSV kullanın:
machine_id, day, vibration, temperature, pressure, load, failure (arıza günü = 1)
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def generate(seed: int = 7, n_machines: int = 60, n_days: int = 365) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_machines):
        vib0, temp0, pres0 = rng.normal(2.0, 0.2), rng.normal(60, 3), rng.normal(5.0, 0.3)
        base_rate = rng.uniform(0.005, 0.010)
        wear = rng.uniform(0, 0.6)                       # makineler yaşam döngüsünün farklı yerlerinde başlar
        cycle_mult, signal = _new_cycle(rng)
        for day in range(n_days):
            load = rng.uniform(0.6, 1.0)
            wear += base_rate * cycle_mult * load * rng.gamma(4, 0.25)   # gamma(4, .25) ortalaması 1
            w = min(wear, 1.0)
            vib = vib0 + signal * 2.5 * w**3 + rng.normal(0, 0.15)
            temp = temp0 + signal * 10 * w**2 + 6 * (load - 0.8) + rng.normal(0, 1.0)
            pres = pres0 - signal * 0.8 * w**2 + rng.normal(0, 0.1)
            if rng.random() < 0.01:                      # ara sıra sensör sıçraması (aykırı değer)
                vib += rng.uniform(1, 3)
            failure = int(wear >= 1.0)
            rows.append((m, day, vib, temp, pres, load, failure))
            if failure:
                wear = 0.0
                cycle_mult, signal = _new_cycle(rng)
    df = pd.DataFrame(rows, columns=["machine_id", "day", "vibration", "temperature", "pressure", "load", "failure"])
    return df.round({"vibration": 3, "temperature": 2, "pressure": 3, "load": 3})


def _new_cycle(rng):
    """Döngü başına aşınma hızı çarpanı ve sensör sinyal gücü (%15 ani arıza döngüsü)."""
    cycle_mult = rng.lognormal(0.0, 0.35)
    signal = 1.0 if rng.random() > 0.15 else 0.3
    return cycle_mult, signal


def main():
    DATA_DIR.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(DATA_DIR / "sensor_data.csv", index=False)
    print(f"{len(df)} satır, {df['failure'].sum()} arıza -> {DATA_DIR / 'sensor_data.csv'}")


if __name__ == "__main__":
    main()
