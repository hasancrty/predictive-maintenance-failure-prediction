"""Uçtan uca çalıştırma: veri -> özellikler -> model karşılaştırması -> maliyet bazlı eşik -> test sonucu.

Kullanım (proje klasöründen):  python -m src.run
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.features import HORIZON, build_dataset, feature_columns
from src.generate_data import DATA_DIR, main as generate_data
from src.modeling import COST_ALARM, COST_MISSED, best_threshold, cost_curve, cross_validate, make_models, total_cost

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
TEST_START_DAY = 265   # bu günden sonrası test; eğitim etiketleri en fazla HORIZON gün ileriye baktığı için boşluk bırakılır


def plot_sensor_example(raw: pd.DataFrame, path: Path):
    m = raw[raw["machine_id"] == raw.groupby("machine_id")["failure"].sum().idxmax()]
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for ax, s, unit in zip(axes, ["vibration", "temperature", "pressure"], ["mm/s", "°C", "bar"]):
        ax.plot(m["day"], m[s], lw=0.8)
        for d in m.loc[m["failure"] == 1, "day"]:
            ax.axvline(d, color="red", ls="--", lw=1)
        ax.set_ylabel(f"{s} ({unit})")
    axes[0].set_title(f"Makine {int(m['machine_id'].iloc[0])}: sensörler arıza öncesi bozuluyor (kırmızı çizgi = arıza)")
    axes[-1].set_xlabel("Gün")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_pr_curves(results: dict, y_test, path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in results.items():
        p, r, _ = precision_recall_curve(y_test, proba)
        ax.plot(r, p, label=f"{name} (AP={average_precision_score(y_test, proba):.2f})")
    ax.axhline(y_test.mean(), color="gray", ls=":", label=f"Rastgele ({y_test.mean():.2f})")
    ax.set_xlabel("Duyarlılık (recall)")
    ax.set_ylabel("Kesinlik (precision)")
    ax.set_title("Test dönemi: kesinlik-duyarlılık eğrisi")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_threshold(oof_y, oof_p, y_test, p_test, chosen, path: Path):
    t, c_oof = cost_curve(oof_y, oof_p)
    _, c_test = cost_curve(y_test, p_test)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, c_oof / len(oof_y), label="Doğrulama (out-of-fold)")
    ax.plot(t, c_test / len(y_test), label="Test")
    ax.axvline(chosen, color="red", ls="--", label=f"Seçilen eşik = {chosen:.2f}")
    ax.set_xlabel("Alarm eşiği")
    ax.set_ylabel("Makine-gün başına maliyet")
    ax.set_title("Eşik seçimi: toplam maliyeti en aza indiren eşik")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_importance(names, importances, path: Path):
    order = np.argsort(importances)[-12:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(names)[order], np.array(importances)[order], color="#2980b9")
    ax.set_xlabel("Permütasyon önemi (AP düşüşü)")
    ax.set_title("En önemli özellikler (test dönemi)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_strategies(table: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#95a5a6", "#95a5a6", "#e67e22", "#27ae60"]
    ax.bar(table.index, table["total_cost"] / 1e3, color=colors)
    ax.set_ylabel("Test dönemi toplam maliyet (bin birim)")
    ax.set_title("Bakım stratejilerinin maliyeti")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    if not (DATA_DIR / "sensor_data.csv").exists():
        generate_data()
    raw = pd.read_csv(DATA_DIR / "sensor_data.csv")
    data = build_dataset(raw, HORIZON)
    feats = feature_columns(data)
    print(f"Veri: {len(data)} makine-gün, pozitif oranı %{data['label'].mean() * 100:.1f}, {len(feats)} özellik")

    train = data[data["day"] <= TEST_START_DAY - HORIZON - 1]
    test = data[data["day"] >= TEST_START_DAY]
    Xtr, ytr, dtr = train[feats].to_numpy(), train["label"].to_numpy(), train["day"].to_numpy()
    Xte, yte = test[feats].to_numpy(), test["label"].to_numpy()
    print(f"Eğitim: {len(train)} satır ({ytr.sum()} pozitif), Test: {len(test)} satır ({yte.sum()} pozitif)")

    # 1) Zaman bazlı çapraz doğrulama ile model karşılaştırması
    models, cv, rows = make_models(), {}, []
    for name, model in models.items():
        cv[name] = cross_validate(model, Xtr, ytr, dtr, n_splits=4, gap_days=HORIZON)
        rows.append({"model": name, "cv_roc_auc": cv[name]["roc_auc"], "cv_pr_auc": cv[name]["pr_auc"]})
    comparison = pd.DataFrame(rows).set_index("model")

    # 2) Tüm modelleri tüm eğitim verisiyle eğit, test döneminde değerlendir
    fitted = {n: clone(m).fit(Xtr, ytr) for n, m in models.items()}
    test_proba = {n: m.predict_proba(Xte)[:, 1] for n, m in fitted.items()}
    comparison["test_roc_auc"] = [roc_auc_score(yte, test_proba[n]) for n in comparison.index]
    comparison["test_pr_auc"] = [average_precision_score(yte, test_proba[n]) for n in comparison.index]

    # 3) En iyi modeli CV PR-AUC ile seç (test sonucuna bakmadan)
    best = comparison["cv_pr_auc"].idxmax()
    print(f"\nSeçilen model (CV PR-AUC): {best}")

    # 4) Eşiği yalnızca eğitim dönemi out-of-fold tahminlerinden seç; test'e dokunma
    oof_y = ytr[cv[best]["oof_idx"]]
    threshold = best_threshold(oof_y, cv[best]["oof_proba"])

    # 5) Stratejilerin test maliyeti
    p = test_proba[best]
    strategies = {
        "Bakım yok (hiç alarm yok)": np.zeros(len(yte), bool),
        "Her gün alarm": np.ones(len(yte), bool),
        "Model, eşik = 0.50": p >= 0.5,
        f"Model, maliyet-optimal eşik = {threshold:.2f}": p >= threshold,
    }
    table = pd.DataFrame({
        name: {
            "alarm_count": int(a.sum()),
            "recall": (a & (yte == 1)).sum() / max(yte.sum(), 1),
            "precision": (a & (yte == 1)).sum() / max(a.sum(), 1),
            "total_cost": total_cost(yte, a),
        } for name, a in strategies.items()
    }).T
    table["cost_per_machine_day"] = table["total_cost"] / len(yte)
    table["saving_vs_no_maintenance"] = table["total_cost"].iloc[0] - table["total_cost"]

    # 6) Özellik önemi (permütasyon, test dönemi)
    perm = permutation_importance(fitted[best], Xte, yte, scoring="average_precision", n_repeats=5, random_state=0)

    OUT_DIR.mkdir(exist_ok=True)
    comparison.round(4).to_csv(OUT_DIR / "model_comparison.csv")
    table.round(3).to_csv(OUT_DIR / "strategy_costs.csv")
    (pd.Series(perm.importances_mean, index=feats, name="permutation_importance").rename_axis("feature")
       .sort_values(ascending=False).round(4).to_csv(OUT_DIR / "feature_importance.csv"))
    plot_sensor_example(raw, OUT_DIR / "sensor_example.png")
    plot_pr_curves(test_proba, yte, OUT_DIR / "pr_curves.png")
    plot_threshold(oof_y, cv[best]["oof_proba"], yte, p, threshold, OUT_DIR / "threshold_cost.png")
    plot_importance(feats, perm.importances_mean, OUT_DIR / "feature_importance.png")
    plot_strategies(table, OUT_DIR / "strategy_costs.png")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== Model karşılaştırması ===")
    print(comparison.round(3))
    print(f"\n=== Test dönemi stratejileri (alarm={COST_ALARM:.0f}, kaçırılan={COST_MISSED:.0f}) ===")
    print(table.round(3))
    print(f"\nÇıktılar: {OUT_DIR}")


if __name__ == "__main__":
    main()
