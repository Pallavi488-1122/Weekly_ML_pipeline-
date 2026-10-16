import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import build_model, evaluate, load_data, validate_data


def test_dataset_is_valid():
    X, y = load_data()
    validate_data(X, y)
    assert len(X) == len(y)
    assert len(X) > 0


def test_model_trains_and_metrics_are_bounded():
    from sklearn.model_selection import train_test_split

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = build_model()
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
