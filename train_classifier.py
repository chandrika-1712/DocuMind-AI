
"""
Short Description:
Train and evaluate ML models for document classification.

Approach:
1. Download labeled text documents.
2. Group them into Technology, Science and Sports.
3. Convert text into TF-IDF features.
4. Train Logistic Regression and Linear SVM.
5. Compare their performance and save the best model.
"""

from pathlib import Path

import joblib
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
import matplotlib.pyplot as plt


# Project paths
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

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


def load_dataset():
    """Load real text documents and map them to three classes."""

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
    """Combine text vectorization and classification."""

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
    print("Loading the dataset...")
    texts, labels = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Linear SVM": LinearSVC(
            random_state=42,
        ),
    }

    best_model = None
    best_name = None
    best_f1 = -1.0

    for name, estimator in models.items():
        print(f"\nTraining {name}...")

        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1-score: {macro_f1:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, predictions))

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model = pipeline
            best_name = name

    # Evaluate the selected model
    final_predictions = best_model.predict(X_test)

    print(f"\nBest model: {best_name}")
    print(f"Best macro F1-score: {best_f1:.4f}")

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_test,
            final_predictions,
            labels=["Science", "Sports", "Technology"],
        )
    )

    # Save the complete pipeline, including TF-IDF
    model_path = ARTIFACTS_DIR / "document_classifier.joblib"

    joblib.dump(
        {
            "pipeline": best_model,
            "model_name": best_name,
            "macro_f1": best_f1,
            "classes": ["Science", "Sports", "Technology"],
        },
        model_path,
    )

    print(f"\nModel saved to: {model_path}")

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        final_predictions,
        labels=["Science", "Sports", "Technology"],
        xticks_rotation=30,
    )

    plt.tight_layout()
    plt.savefig(
        ARTIFACTS_DIR / "confusion_matrix.png",
        dpi=150,
    )
    plt.close()

    print("Confusion matrix image saved.")


if __name__ == "__main__":
    main()
