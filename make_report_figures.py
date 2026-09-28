"""Собирает изображения для отчёта из данных и результатов обучения."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "report" / "images"
VALIDATION_START = pd.Timestamp("2026-04-17")
COLORS = ["#4c78a8", "#72b7b2", "#f58518", "#b279a2"]


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден файл {path}. Сначала запустите нужный этап решения.")
    return path


def save_data_overview(train: pd.DataFrame) -> None:
    counts = train.target.value_counts().reindex([0, 1], fill_value=0)
    daily_counts = (
        train.assign(day=train.window_start_ts.dt.floor("D"))
        .groupby("day")
        .size()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bars = axes[0].bar(
        ["Обычные cookie", "Боты"], counts.values, color=COLORS[:2], width=0.62
    )
    axes[0].bar_label(bars, fmt="%.0f", padding=4)
    axes[0].set_title("Классы в обучающей выборке", loc="left", weight="bold")
    axes[0].set_ylabel("Количество cookie")

    axes[1].plot(
        daily_counts.index,
        daily_counts.values,
        color=COLORS[0],
        linewidth=2.5,
        marker="o",
    )
    axes[1].axvspan(
        VALIDATION_START,
        daily_counts.index.max() + pd.Timedelta(days=1),
        color=COLORS[2],
        alpha=0.14,
        label="Период проверки",
    )
    axes[1].axvline(VALIDATION_START, color=COLORS[2], linestyle="--")
    axes[1].set_title("Примеры по датам", loc="left", weight="bold")
    axes[1].set_ylabel("Количество cookie")
    axes[1].set_xlabel("Дата начала окна")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "data_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_validation_metrics(results: dict[str, object]) -> None:
    model_labels = {
        "catboost": "CatBoost",
        "extra_trees": "Extra Trees",
        "hist_gradient_boosting": "Градиентный бустинг",
        "best_rank_blend": "Смесь моделей",
    }
    metric_labels = {
        "precision_at_recall_070": "Precision при Recall ≥ 70%",
        "average_precision": "Average Precision",
        "roc_auc": "ROC-AUC",
    }
    models = {
        label: results[key]
        for key, label in model_labels.items()
        if isinstance(results.get(key), dict)
    }
    if not models:
        raise ValueError("В validation_results.json нет результатов моделей.")

    fig, axis = plt.subplots(figsize=(10, 5))
    positions = range(len(models))
    width = 0.23
    for offset, (metric, label), color in zip(
        [-width, 0, width], metric_labels.items(), COLORS
    ):
        values = [model[metric] for model in models.values()]
        axis.bar(
            [position + offset for position in positions],
            values,
            width,
            color=color,
            label=label,
        )

    axis.set_xticks(list(positions), list(models), rotation=12, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Значение метрики")
    axis.set_title("Качество на отложенных датах", loc="left", weight="bold")
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncols=3,
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.18)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(IMAGE_DIR / "validation_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(features: pd.DataFrame) -> None:
    top_features = features.nlargest(15, "importance").sort_values("importance")
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(top_features.feature, top_features.importance, color=COLORS[0])
    axis.set_title("Самые важные признаки", loc="left", weight="bold")
    axis.set_xlabel("Важность признака")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_score_distribution(submission: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(
        submission.score,
        bins=30,
        color=COLORS[1],
        edgecolor="white",
    )
    axis.set_title("Оценки модели для тестовых cookie", loc="left", weight="bold")
    axis.set_xlabel("Оценка модели")
    axis.set_ylabel("Количество cookie")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "submission_scores.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(
        require_file(ROOT / "data" / "train.csv"),
        parse_dates=["window_start_ts"],
    )
    validation = json.loads(
        require_file(ROOT / "validation_results.json").read_text(encoding="utf-8")
    )
    importance = pd.read_csv(require_file(ROOT / "feature_importance.csv"))
    submission = pd.read_csv(require_file(ROOT / "submission.csv"))

    save_data_overview(train)
    save_validation_metrics(validation)
    save_feature_importance(importance)
    save_score_distribution(submission)
    print(f"Графики отчёта сохранены в {IMAGE_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
