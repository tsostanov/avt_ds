"""Воспроизводимое решение задачи определения ботов по событиям Avito.

Скрипт собирает поведенческие признаки по событиям внутри временного окна
каждой cookie, проверяет качество на последних днях train и обучает итоговую
модель на всех размеченных данных.

Запуск:
    python solution.py evaluate
    python solution.py train
    python solution.py all
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline

from metric import precision_at_recall


SEED = 20260928
DATA_DIR = Path("data")
EVENT_NAMES = [
    "search_results_view",
    "item_view",
    "photo_swipe",
    "seller_page_view",
    "favorite_add",
    "contact_phone_show",
    "contact_chat_open",
    "contact_message_sent",
    "login",
    "captcha_shown",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    date_cols = ["cookie_created_at", "window_start_ts", "window_end_ts"]
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=date_cols)
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=date_cols)
    events = pd.read_csv(DATA_DIR / "events.csv.gz", parse_dates=["event_ts"])
    return train, test, events


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
        for col in frame.columns
    ]
    return frame


def _safe_name(value: object) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(value)).strip("_").lower()


def _mode(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    if values.empty:
        return "__missing__"
    return str(values.value_counts().index[0])


def _first_non_null(series: pd.Series) -> str:
    values = series.dropna()
    return "__missing__" if values.empty else str(values.iloc[0])


def _last_non_null(series: pd.Series) -> str:
    values = series.dropna()
    return "__missing__" if values.empty else str(values.iloc[-1])


def _mode_by_cookie(events: pd.DataFrame, column: str) -> pd.Series:
    """Находит самое частое непустое значение отдельно для каждой cookie."""
    counts = (
        events.dropna(subset=[column])
        .groupby(["cookie_id", column], observed=True, sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    if counts.empty:
        return pd.Series(dtype="object")
    return (
        counts.sort_values(["cookie_id", "count"], ascending=[True, False], kind="mergesort")
        .drop_duplicates("cookie_id")
        .set_index("cookie_id")[column]
        .astype(str)
    )


def _user_agent_family(value: object) -> str:
    ua = str(value).lower()
    if "headless" in ua:
        return "headless"
    for token, family in [
        ("scrapy", "scrapy"),
        ("python-requests", "python_requests"),
        ("python-urllib", "python_urllib"),
        ("node-fetch", "node_fetch"),
        ("go-http-client", "go_http"),
        ("curl/", "curl"),
        ("wget/", "wget"),
        ("okhttp", "okhttp"),
    ]:
        if token in ua:
            return family
    if "avito/" in ua:
        return "avito_app"
    if "firefox/" in ua:
        return "firefox"
    if "yabrowser/" in ua:
        return "yandex_browser"
    if "chrome/" in ua or "crios/" in ua:
        return "chrome"
    if "safari/" in ua:
        return "safari"
    return "other"


def _concentration_features(events: pd.DataFrame, column: str) -> pd.DataFrame:
    """Считает частоты, разнообразие и концентрацию значений поля."""
    prefix = _safe_name(column)
    group = events.groupby("cookie_id", observed=True)[column]
    out = group.agg(["count", "nunique"])
    out.columns = [f"{prefix}_non_null_count", f"{prefix}_nunique"]

    counts = (
        events.dropna(subset=[column])
        .groupby(["cookie_id", column], observed=True)
        .size()
        .rename("value_count")
    )
    if not counts.empty:
        maximum = counts.groupby(level=0).max().rename(f"{prefix}_max_frequency")
        total = counts.groupby(level=0).sum()
        sum_c_log_c = (counts * np.log(counts)).groupby(level=0).sum()
        entropy = (np.log(total) - sum_c_log_c / total).rename(f"{prefix}_entropy")
        out = out.join(maximum).join(entropy)
        out[f"{prefix}_top_share"] = out[f"{prefix}_max_frequency"] / out[
            f"{prefix}_non_null_count"
        ].clip(lower=1)
    return out


def build_features(meta: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Создаёт строку признаков на cookie и сохраняет порядок из ``meta``."""
    meta = meta.copy()
    cookie_order = meta["cookie_id"].copy()
    meta_idx = meta.set_index("cookie_id", drop=False)

    # Сначала присоединяем границы окна каждой cookie, затем фильтруем события.
    # Так в признаки не попадут более ранняя история и правая граница окна.
    window_cols = ["cookie_id", "window_start_ts", "window_end_ts"]
    ev = events.merge(meta[window_cols], on="cookie_id", how="inner", validate="many_to_one")
    ev = ev.loc[(ev.event_ts >= ev.window_start_ts) & (ev.event_ts < ev.window_end_ts)].copy()
    ev.sort_values(["cookie_id", "event_ts", "eid"], inplace=True, kind="mergesort")

    ev["platform_norm"] = ev["platform"].fillna("__missing__").astype(str).str.lower()
    ev["ua_family"] = ev["user_agent"].map(_user_agent_family)
    ev["seconds_from_start"] = (ev.event_ts - ev.window_start_ts).dt.total_seconds()
    ev["event_hour"] = ev.event_ts.dt.hour
    ev["event_minute_bucket"] = (ev["seconds_from_start"] // 60).astype("int32")
    ev["gap_seconds"] = ev.groupby("cookie_id", observed=True).event_ts.diff().dt.total_seconds()

    group = ev.groupby("cookie_id", observed=True)
    features = group.agg(
        event_count=("eid", "size"),
        event_type_nunique=("event_name", "nunique"),
        timestamp_nunique=("event_ts", "nunique"),
        active_hour_nunique=("event_hour", "nunique"),
        active_minute_nunique=("event_minute_bucket", "nunique"),
        first_event_second=("seconds_from_start", "min"),
        last_event_second=("seconds_from_start", "max"),
        event_second_mean=("seconds_from_start", "mean"),
        event_second_std=("seconds_from_start", "std"),
        event_second_median=("seconds_from_start", "median"),
        gap_count=("gap_seconds", "count"),
        gap_mean=("gap_seconds", "mean"),
        gap_std=("gap_seconds", "std"),
        gap_min=("gap_seconds", "min"),
        gap_max=("gap_seconds", "max"),
        gap_median=("gap_seconds", "median"),
        search_page_mean=("search_page", "mean"),
        search_page_std=("search_page", "std"),
        search_page_min=("search_page", "min"),
        search_page_max=("search_page", "max"),
        pointer_x_mean=("pointer_x", "mean"),
        pointer_x_std=("pointer_x", "std"),
        pointer_x_min=("pointer_x", "min"),
        pointer_x_max=("pointer_x", "max"),
        pointer_y_mean=("pointer_y", "mean"),
        pointer_y_std=("pointer_y", "std"),
        pointer_y_min=("pointer_y", "min"),
        pointer_y_max=("pointer_y", "max"),
    )

    features["activity_span_seconds"] = features.last_event_second - features.first_event_second
    features["events_per_active_minute"] = features.event_count / features.active_minute_nunique.clip(lower=1)
    features["events_per_timestamp"] = features.event_count / features.timestamp_nunique.clip(lower=1)
    features["gap_cv"] = features.gap_std / features.gap_mean.clip(lower=0.001)
    features["pointer_x_range"] = features.pointer_x_max - features.pointer_x_min
    features["pointer_y_range"] = features.pointer_y_max - features.pointer_y_min

    gap_quantiles = group.gap_seconds.quantile([0.10, 0.25, 0.75, 0.90]).unstack()
    gap_quantiles.columns = ["gap_q10", "gap_q25", "gap_q75", "gap_q90"]
    features = features.join(gap_quantiles, how="outer")

    for threshold in [0, 1, 2, 5, 10, 30, 60, 300, 600, 1800]:
        name = f"gap_le_{threshold}_share"
        features[name] = ev.gap_seconds.le(threshold).groupby(ev.cookie_id).mean()
    for threshold in [3600, 21600, 43200]:
        name = f"gap_ge_{threshold}_share"
        features[name] = ev.gap_seconds.ge(threshold).groupby(ev.cookie_id).mean()

    # Профили показывают повторяемость действий, платформ, часов и семейств UA.
    profiles = {
        "event": pd.crosstab(ev.cookie_id, ev.event_name),
        "platform": pd.crosstab(ev.cookie_id, ev.platform_norm),
        "hour": pd.crosstab(ev.cookie_id, ev.event_hour),
        "ua_family": pd.crosstab(ev.cookie_id, ev.ua_family),
    }
    for prefix, profile in profiles.items():
        profile.columns = [f"{prefix}_{_safe_name(col)}_count" for col in profile.columns]
        features = features.join(profile, how="outer")

    for event_name in EVENT_NAMES:
        count_col = f"event_{_safe_name(event_name)}_count"
        if count_col not in features:
            features[count_col] = 0.0
        features[f"event_{_safe_name(event_name)}_share"] = features[count_col] / features[
            "event_count"
        ].clip(lower=1)

    for column in [
        "user_agent",
        "item_id",
        "item_category",
        "item_location",
        "seller_type",
        "search_query",
        "search_page",
        "platform_norm",
    ]:
        features = features.join(_concentration_features(ev, column), how="outer")

    # Движения указателя помогают заметить регулярные автоматические действия.
    pointer = ev.loc[ev.pointer_x.notna() & ev.pointer_y.notna(), ["cookie_id", "pointer_x", "pointer_y"]].copy()
    if not pointer.empty:
        pointer["dx"] = pointer.groupby("cookie_id", observed=True).pointer_x.diff()
        pointer["dy"] = pointer.groupby("cookie_id", observed=True).pointer_y.diff()
        pointer["distance"] = np.hypot(pointer.dx, pointer.dy)
        pointer["zero_move"] = pointer.distance.eq(0).astype(float)
        pointer_features = pointer.groupby("cookie_id", observed=True).agg(
            pointer_count=("pointer_x", "size"),
            pointer_x_nunique=("pointer_x", "nunique"),
            pointer_y_nunique=("pointer_y", "nunique"),
            pointer_distance_mean=("distance", "mean"),
            pointer_distance_std=("distance", "std"),
            pointer_distance_max=("distance", "max"),
            pointer_zero_move_share=("zero_move", "mean"),
        )
        features = features.join(pointer_features, how="outer")

    categorical_sources = [
        "user_agent",
        "ua_family",
        "platform_norm",
        "item_category",
        "item_location",
        "seller_type",
        "search_query",
    ]
    for column in categorical_sources:
        features[f"mode_{column}"] = _mode_by_cookie(ev, column)
        features[f"first_{column}"] = group[column].first().astype(str)
        features[f"last_{column}"] = group[column].last().astype(str)

    sequences = group.event_name.agg(list)
    features["sequence_head"] = sequences.map(lambda x: ">".join(x[:4]) if x else "__missing__")
    features["sequence_tail"] = sequences.map(lambda x: ">".join(x[-4:]) if x else "__missing__")
    features["sequence_switch_share"] = sequences.map(
        lambda x: np.mean(np.asarray(x[1:]) != np.asarray(x[:-1])) if len(x) > 1 else 0.0
    )
    features["sequence_unique_bigram_count"] = sequences.map(
        lambda x: len(set(zip(x[:-1], x[1:]))) if len(x) > 1 else 0
    )
    # Эти сведения доступны при расчёте результата и не используют target.
    meta_features = pd.DataFrame(index=meta_idx.index)
    age_hours = (meta_idx.window_start_ts - meta_idx.cookie_created_at).dt.total_seconds() / 3600
    meta_features["cookie_age_hours"] = age_hours
    meta_features["log1p_cookie_age_hours"] = np.log1p(age_hours.clip(lower=0))
    meta_features["cookie_created_hour"] = meta_idx.cookie_created_at.dt.hour
    meta_features["cookie_created_dow"] = meta_idx.cookie_created_at.dt.dayofweek
    meta_features["window_start_dow"] = meta_idx.window_start_ts.dt.dayofweek
    meta_features["cookie_created_on_window_day"] = (
        meta_idx.cookie_created_at.dt.normalize() == meta_idx.window_start_ts.dt.normalize()
    ).astype(int)
    features = meta_features.join(features, how="left")

    categorical = [col for col in features if features[col].dtype == "object"]
    features[categorical] = features[categorical].fillna("__missing__").astype(str)
    numeric = [col for col in features if col not in categorical]
    features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features = features.reindex(cookie_order.to_numpy())
    features.index.name = "cookie_id"
    return features


def catboost_model(iterations: int = 900, seed: int = SEED) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=7,
        learning_rate=0.035,
        loss_function="Logloss",
        eval_metric="PRAUC:type=Classic",
        l2_leaf_reg=7.0,
        random_seed=seed,
        random_strength=0.5,
        bootstrap_type="Bayesian",
        bagging_temperature=0.7,
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
    )


def score_report(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "precision_at_recall_070": precision_at_recall(y_true, scores),
        "average_precision": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "positive_rate": float(np.mean(y_true)),
    }


def evaluate_models(train: pd.DataFrame, features: pd.DataFrame) -> dict[str, object]:
    """Проверяет модели на последних трёх днях train — приближении к будущему test."""
    validation_start = pd.Timestamp("2026-04-17")
    is_valid = train.window_start_ts.ge(validation_start).to_numpy()
    y = train.target.to_numpy(dtype=int)
    categorical = [col for col in features if features[col].dtype == "object"]
    numeric = [col for col in features if col not in categorical]

    x_train, x_valid = features.loc[~is_valid], features.loc[is_valid]
    y_train, y_valid = y[~is_valid], y[is_valid]
    results: dict[str, object] = {
        "split": {
            "train_rows": int((~is_valid).sum()),
            "valid_rows": int(is_valid.sum()),
            "valid_from": str(validation_start.date()),
        }
    }

    cb = catboost_model(iterations=1400)
    cb.fit(
        x_train,
        y_train,
        cat_features=categorical,
        eval_set=(x_valid, y_valid),
        early_stopping_rounds=150,
        verbose=False,
    )
    cb_scores = cb.predict_proba(x_valid)[:, 1]
    results["catboost"] = score_report(y_valid, cb_scores) | {
        "best_iteration": int(cb.get_best_iteration())
    }

    extra = make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=700,
            min_samples_leaf=2,
            max_features=0.8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
    )
    extra.fit(x_train[numeric], y_train)
    extra_scores = extra.predict_proba(x_valid[numeric])[:, 1]
    results["extra_trees"] = score_report(y_valid, extra_scores)

    hist = HistGradientBoostingClassifier(
        learning_rate=0.045,
        max_iter=350,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=5.0,
        random_state=SEED,
    )
    hist.fit(x_train[numeric], y_train)
    hist_scores = hist.predict_proba(x_valid[numeric])[:, 1]
    results["hist_gradient_boosting"] = score_report(y_valid, hist_scores)

    best_blend = None
    for cb_weight in np.linspace(0.0, 1.0, 21):
        # Усреднение рангов позволяет сравнить модели с разными шкалами оценок.
        cb_rank = pd.Series(cb_scores).rank(pct=True).to_numpy()
        extra_rank = pd.Series(extra_scores).rank(pct=True).to_numpy()
        blend = cb_weight * cb_rank + (1.0 - cb_weight) * extra_rank
        report = score_report(y_valid, blend)
        candidate = report | {"catboost_weight": float(cb_weight)}
        if best_blend is None or candidate["precision_at_recall_070"] > best_blend[
            "precision_at_recall_070"
        ]:
            best_blend = candidate
    results["best_rank_blend"] = best_blend
    return results


def train_final(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    iterations: int,
) -> pd.DataFrame:
    categorical = [col for col in train_features if train_features[col].dtype == "object"]
    model = catboost_model(iterations=iterations)
    model.fit(train_features, train.target.to_numpy(dtype=int), cat_features=categorical, verbose=False)
    scores = model.predict_proba(test_features)[:, 1]
    submission = pd.DataFrame({"cookie_id": test.cookie_id, "score": scores})
    assert len(submission) == len(test)
    assert submission.cookie_id.is_unique
    assert set(submission.cookie_id) == set(test.cookie_id)
    assert submission.score.between(0.0, 1.0).all()
    submission.to_csv("submission.csv", index=False)

    importance = pd.DataFrame(
        {"feature": train_features.columns, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False)
    importance.head(40).to_csv("feature_importance.csv", index=False)
    model.save_model("model.cbm")
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Проверка качества и обучение модели определения ботов."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["evaluate", "train", "all"],
        default="all",
        help="evaluate — валидация, train — обучение, all — выполнить оба этапа (по умолчанию)",
    )
    args = parser.parse_args()

    train, test, events = load_data()
    duplicate_count = int(events.duplicated().sum())
    events = events.drop_duplicates().copy()
    print(f"Удалено точных дубликатов событий: {duplicate_count}")
    meta = pd.concat([train.drop(columns="target"), test], ignore_index=True)
    print("Считаю признаки по событиям внутри окон наблюдения ...")
    all_features = build_features(meta, events)
    train_features = all_features.reindex(train.cookie_id)
    test_features = all_features.reindex(test.cookie_id)
    print(f"Матрица признаков: {all_features.shape[0]} cookie x {all_features.shape[1]} признаков")

    # Число итераций выбрано по позднему временному holdout. Для финальной
    # модели оно немного увеличено: обучающих меток становится примерно на 21% больше.
    best_iterations = 1600
    if args.mode in {"evaluate", "all"}:
        results = evaluate_models(train, train_features)
        Path("validation_results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))
        best_iterations = max(300, int(results["catboost"]["best_iteration"] * 1.15))

    if args.mode in {"train", "all"}:
        submission = train_final(
            train, test, train_features, test_features, iterations=best_iterations
        )
        print(
            f"Сохранён submission.csv: {len(submission)} строк; "
            f"диапазон score [{submission.score.min():.6f}, {submission.score.max():.6f}]"
        )


if __name__ == "__main__":
    main()
