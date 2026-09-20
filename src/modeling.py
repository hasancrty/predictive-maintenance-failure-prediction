"""Model karşılaştırması (zamana duyarlı çapraz doğrulama) ve maliyete dayalı eşik seçimi."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Karar birimi = bir makine-gün. Plansız arıza ≈ 10.000 birim; bir arıza penceresi ≈ 7 pozitif satır olduğundan
# kaçırılan her pozitif satır ≈ 1.430 sayılır. Alarm (denetim/önleyici bakım) satır başına 300 birimdir.
COST_MISSED = 1430.0
COST_ALARM = 300.0


def make_models(seed: int = 0) -> dict:
    return {
        "Lojistik Regresyon": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample", random_state=seed),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=200, class_weight="balanced", random_state=seed),
    }


def time_folds(days: np.ndarray, n_splits: int, gap_days: int):
    """Genişleyen pencereli zaman bazlı katlar. Eğitim, doğrulama başlangıcından `gap_days` önce biter;
    böylece eğitim etiketlerinin ileriye bakan penceresi doğrulama dönemine sızmaz."""
    edges = np.linspace(days.min(), days.max() + 1, n_splits + 2)
    for k in range(1, n_splits + 1):
        train = np.flatnonzero(days < edges[k] - gap_days)
        valid = np.flatnonzero((days >= edges[k]) & (days < edges[k + 1]))
        yield train, valid


def cross_validate(model, X, y, days, n_splits: int = 4, gap_days: int = 7):
    """Katlar üzerinde ROC-AUC ve PR-AUC (ortalama) ile birleştirilmiş out-of-fold olasılıkları döndürür."""
    from sklearn.base import clone
    roc, pr, oof_idx, oof_proba = [], [], [], []
    for tr, va in time_folds(days, n_splits, gap_days):
        m = clone(model).fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        roc.append(roc_auc_score(y[va], p))
        pr.append(average_precision_score(y[va], p))
        oof_idx.append(va)
        oof_proba.append(p)
    return {"roc_auc": np.mean(roc), "pr_auc": np.mean(pr),
            "oof_idx": np.concatenate(oof_idx), "oof_proba": np.concatenate(oof_proba)}


def total_cost(y, alarm, cost_missed: float = COST_MISSED, cost_alarm: float = COST_ALARM) -> float:
    y, alarm = np.asarray(y).astype(bool), np.asarray(alarm).astype(bool)
    missed = np.sum(y & ~alarm)
    return float(missed * cost_missed + alarm.sum() * cost_alarm)


def cost_curve(y, proba, thresholds=None, **costs):
    thresholds = np.linspace(0.01, 0.99, 99) if thresholds is None else thresholds
    return thresholds, np.array([total_cost(y, proba >= t, **costs) for t in thresholds])


def best_threshold(y, proba, thresholds=None, **costs) -> float:
    thresholds, c = cost_curve(y, proba, thresholds, **costs)
    return float(thresholds[np.argmin(c)])
