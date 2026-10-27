
"""
Short Description:
Train, evaluate, compare, and save document classification models.

Approach:
1. Load labeled text documents.
2. Map documents into Technology, Science, and Sports categories.
3. Convert text into TF-IDF features.
4. Train Logistic Regression and Linear SVM.
5. Compare accuracy and macro F1-score on the same test set.
6. Save the best model and both models' evaluation metrics.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# Project paths
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "comp.graphics",
    "comp.sys.ibm.pc.hardware",
    "sci.space",
    "rec.sport.baseball",
    "rec.sport.hockey",
]

CATEGORY_MAP = {
    "comp.graphics": "Technology",
    "comp.sys.ibm.pc.hardware": "Technology",
    "sci.space": "Science",
    "rec.sport.baseball": "Sports",
    "rec.sport.hockey": "Sports",
}

CLASS_NAMES = ["Science", "Sports", "Technology"]


def load_dataset():
    """Load text documents and map them to three classes."""

    dataset = fetch_20newsgroups(
        subset="all",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )

    labels = [
        CATEGORY_MAP[dataset.target_names[target]]
        for target in dataset.target
    ]

    return dataset.data, labels


def build_pipeline(model):
    """Combine TF-IDF feature extraction and classification."""

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=30000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        ("classifier", model),
    ])


def main():
    print("Loading dataset...")
    texts, labels = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    print(f"Total documents: {len(texts)}")
    print(f"Training documents: {len(X_train)}")
    print(f"Testing documents: {len(X_test)}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Linear SVM": LinearSVC(
            random_state=42,
        ),
    }

    results = {}
    best_model = None
    best_name = None
    best_f1 = -1.0
    best_predictions = None

    for name, estimator in models.items():
        print(f"\n{'=' * 50}")
        print(f"Training {name}...")
        print("=" * 50)

        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test,
            predictions,
            labels=CLASS_NAMES,
            average="macro",
            zero_division=0,
        )

        results[name] = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
        }

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1-score: {macro_f1:.4f}")

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=CLASS_NAMES,
                target_names=CLASS_NAMES,
                zero_division=0,
            )
        )

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model = pipeline
            best_name = name
            best_predictions = predictions

    # Compare both models
    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<25}{'Accuracy':<15}{'Macro F1-score':<15}")

    for name, metrics in results.items():
        print(
            f"{name:<25}"
            f"{metrics['accuracy']:<15.4f}"
            f"{metrics['macro_f1']:<15.4f}"
        )

    print(f"\nBest model: {best_name}")
    print(f"Best macro F1-score: {best_f1:.4f}")

    # Confusion matrix for the selected model
    matrix = confusion_matrix(
        y_test,
        best_predictions,
        labels=CLASS_NAMES,
    )

    print("\nConfusion matrix:")
    print(matrix)

    # Save the best pipeline and evaluation results.
    # Keep these keys compatible with the existing app.py.
    model_path = ARTIFACTS_DIR / "document_classifier.joblib"

    joblib.dump(
        {
            "pipeline": best_model,
            "model_name": best_name,
            "macro_f1": best_f1,
            "classes": CLASS_NAMES,
            "evaluation_results": results,
        },
        model_path,
    )

    print(f"\nBest model saved to: {model_path}")

    # Save the confusion matrix image.
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    ).plot(xticks_rotation=30)

    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()

    image_path = ARTIFACTS_DIR / "confusion_matrix.png"
    plt.savefig(image_path, dpi=150)
    plt.close()

    print(f"Confusion matrix image saved to: {image_path}")


if __name__ == "__main__":
    main()
