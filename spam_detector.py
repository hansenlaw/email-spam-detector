"""
SMS Spam Detector
-----------------
NLP pipeline: text cleaning → TF-IDF → 4 classifiers → model export.
Dataset: UCI SMS Spam Collection (5,572 messages).
"""

import os
import re
import string
import time
import logging
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns

from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging & constants
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "spam.csv")
OUT_DIR   = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "SVC":                 SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    log.info("Loading dataset → %s", path)
    df = pd.read_csv(path, encoding="latin1")
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})

    total  = len(df)
    n_ham  = (df["label"] == "ham").sum()
    n_spam = (df["label"] == "spam").sum()
    log.info(
        "Loaded %d messages  |  ham: %d (%.1f%%)  spam: %d (%.1f%%)",
        total, n_ham, 100 * n_ham / total, n_spam, 100 * n_spam / total,
    )
    return df


# ---------------------------------------------------------------------------
# 2. Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Six-step normalization applied before tokenization."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)       # URLs
    text = re.sub(r"<.*?>+", " ", text)                       # HTML tags
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Preprocessing  (clean → tokenize → remove stopwords → encode label)...")
    t0 = time.time()

    # Make sure NLTK data is available
    for pkg, kind in [("punkt_tab", "tokenizers"), ("stopwords", "corpora")]:
        try:
            nltk.data.find(f"{kind}/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    # fallback for older NLTK
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    sw = set(stopwords.words("english"))

    df = df.copy()
    df["message"]    = df["message"].apply(clean_text)
    df["tokens"]     = df["message"].apply(
        lambda t: [w for w in word_tokenize(t) if w not in sw]
    )
    df["message"]    = df["tokens"].apply(" ".join)
    df["target"]     = (df["label"] == "spam").astype(int)
    df["char_len"]   = df["message"].str.len()
    df["word_count"] = df["tokens"].apply(len)

    log.info("Preprocessing done in %.1fs", time.time() - t0)
    return df


# ---------------------------------------------------------------------------
# 3. EDA — class distribution + message length
# ---------------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts()
    colors = ["#4C9BE8", "#E85C5C"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#F8F9FA")

    # Bar chart
    bars = axes[0].bar(
        counts.index, counts.values,
        color=colors, width=0.5, edgecolor="white", linewidth=1.5,
    )
    for bar, val in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{val:,}", ha="center", fontweight="bold", fontsize=12,
        )
    axes[0].set_title("Message Class Distribution", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_ylim(0, counts.max() * 1.15)
    axes[0].set_facecolor("#F8F9FA")
    axes[0].spines[["top", "right"]].set_visible(False)

    # Message length histogram
    for label, color in zip(["ham", "spam"], colors):
        subset = df[df["label"] == label]["char_len"]
        axes[1].hist(subset, bins=40, alpha=0.65, color=color, label=label, edgecolor="white")
    axes[1].set_title("Message Length by Class", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Characters (after preprocessing)", fontsize=10)
    axes[1].set_ylabel("Frequency", fontsize=10)
    axes[1].legend(fontsize=11)
    axes[1].set_facecolor("#F8F9FA")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("Exploratory Data Analysis — UCI SMS Spam Collection", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "class_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# 4. Feature importance — top TF-IDF terms per class
# ---------------------------------------------------------------------------

def plot_top_terms(df: pd.DataFrame) -> None:
    """Average TF-IDF score per term grouped by class — reveals spam signal words."""
    tfidf_viz = TfidfVectorizer(stop_words="english", max_features=5000)
    X_all     = tfidf_viz.fit_transform(df["message"])
    features  = np.array(tfidf_viz.get_feature_names_out())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#F8F9FA")
    palette = {"spam": "#E85C5C", "ham": "#4C9BE8"}

    for ax, label in zip(axes, ["spam", "ham"]):
        mask = (df["label"] == label).values
        avg  = X_all[mask].mean(axis=0).A1          # mean TF-IDF weight per term
        top  = avg.argsort()[::-1][:15]
        t, s = features[top], avg[top]

        bars = ax.barh(range(15), s[::-1], color=palette[label], alpha=0.85, edgecolor="white")
        ax.set_yticks(range(15))
        ax.set_yticklabels(t[::-1], fontsize=11)
        ax.set_title(
            f"Top 15 Signal Words — {label.upper()}",
            fontsize=13, fontweight="bold", pad=10,
        )
        ax.set_xlabel("Mean TF-IDF Weight", fontsize=10)
        ax.set_facecolor("#F8F9FA")
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Most Discriminative Terms by Class", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "top_terms.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# 5. Feature engineering — TF-IDF vectorization
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame):
    log.info("Building TF-IDF feature matrix...")
    X = df["message"].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=0, stratify=y
    )

    tfidf        = TfidfVectorizer(stop_words="english")
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    log.info(
        "TF-IDF — vocab: %d  |  train: %s  test: %s",
        len(tfidf.vocabulary_), X_train_tfidf.shape, X_test_tfidf.shape,
    )
    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf


# ---------------------------------------------------------------------------
# 6. Train & evaluate all models
# ---------------------------------------------------------------------------

def train_and_evaluate(X_train, X_test, y_train, y_test) -> dict:
    log.info("Training %d models...", len(MODELS))
    results = {}

    for name, clf in MODELS.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        rep  = classification_report(y_test, y_pred, target_names=["Ham", "Spam"], output_dict=True)

        cv   = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)

        results[name] = {
            "model":     clf,
            "y_pred":    y_pred,
            "y_prob":    y_prob,
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "report":    rep,
            "cv_mean":   cv.mean(),
            "cv_std":    cv.std(),
            "train_sec": elapsed,
        }

        log.info(
            "  %-22s  acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  cv=%.4f±%.4f  (%.1fs)",
            name, acc, prec, rec, f1, cv.mean(), cv.std(), elapsed,
        )

    return results


# ---------------------------------------------------------------------------
# 7. Confusion matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrices(results: dict, y_test) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.patch.set_facecolor("#F8F9FA")

    for ax, (name, r) in zip(axes.flat, results.items()):
        cm = confusion_matrix(y_test, r["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"],
            linewidths=0.5, cbar=False, annot_kws={"size": 15, "weight": "bold"},
        )
        ax.set_title(
            f"{name}\nAcc: {r['accuracy']:.2%}",
            fontsize=12, fontweight="bold", pad=8,
        )
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_facecolor("#F8F9FA")

    plt.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "confusion_matrices.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# 8. ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(results: dict, y_test) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    palette = ["#4C9BE8", "#F5A623", "#27AE60", "#E85C5C"]
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random classifier (AUC = 0.5)")

    for (name, r), color in zip(results.items(), palette):
        if r["y_prob"] is not None:
            fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{name}  (AUC = {roc_auc:.4f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "roc_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# 9. Accuracy comparison with cross-validation overlay
# ---------------------------------------------------------------------------

def plot_accuracy_comparison(results: dict) -> None:
    names  = list(results.keys())
    accs   = [r["accuracy"] for r in results.values()]
    cv_m   = [r["cv_mean"]  for r in results.values()]
    cv_s   = [r["cv_std"]   for r in results.values()]
    y_pos  = np.arange(len(names))
    colors = ["#4C9BE8", "#F5A623", "#27AE60", "#E85C5C"]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    bars = ax.barh(y_pos, accs, color=colors, height=0.45, edgecolor="white", linewidth=1.2, label="Test accuracy")
    ax.barh(
        y_pos - 0.28, cv_m, xerr=cv_s, color=colors, height=0.25,
        alpha=0.45, edgecolor="white", linewidth=0.8, label="5-fold CV mean ± std",
    )

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{acc:.2%}", va="center", fontsize=11, fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Accuracy", fontsize=11)
    ax.set_xlim(0.88, 1.02)
    ax.set_title("Model Accuracy — Test Set vs. 5-Fold CV", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "accuracy_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# 10. Save best model (tiebreak: highest precision → fewest false positives)
# ---------------------------------------------------------------------------

def save_best_model(results: dict, tfidf) -> str:
    best_name = max(results, key=lambda k: (results[k]["accuracy"], results[k]["precision"]))
    best      = results[best_name]

    model_path = os.path.join(OUT_DIR, "best_model.joblib")
    tfidf_path = os.path.join(OUT_DIR, "tfidf_vectorizer.joblib")

    joblib.dump(best["model"], model_path)
    joblib.dump(tfidf, tfidf_path)

    log.info(
        "Best model: %s  (acc=%.4f  prec=%.4f  rec=%.4f)",
        best_name, best["accuracy"], best["precision"], best["recall"],
    )
    log.info("Saved → %s", model_path)
    log.info("Saved → %s", tfidf_path)
    return best_name


# ---------------------------------------------------------------------------
# 11. Summary table
# ---------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    sep = "-" * 88
    print(f"\n{sep}")
    print("  EVALUATION SUMMARY")
    print(sep)
    print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'CV Mean':>9} {'Time (s)':>9}")
    print(sep)
    for name, r in results.items():
        print(
            f"  {name:<22}"
            f" {r['accuracy']:>9.4f}"
            f" {r['precision']:>10.4f}"
            f" {r['recall']:>8.4f}"
            f" {r['f1']:>8.4f}"
            f" {r['cv_mean']:>9.4f}"
            f" {r['train_sec']:>9.2f}"
        )
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# 12. Demo prediction
# ---------------------------------------------------------------------------

def demo_predict(model, tfidf) -> None:
    samples = [
        ("WINNER!! You have been selected for a £900 prize reward. Call 09061743810 now!", "spam"),
        ("Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed FREE entry", "spam"),
        ("Hey, are we still on for lunch tomorrow? Let me know.", "ham"),
        ("I'll be home by 7. Can you pick up some groceries on the way?", "ham"),
    ]

    print("-" * 65)
    print("  DEMO PREDICTIONS")
    print("-" * 65)
    for text, actual in samples:
        vec   = tfidf.transform([clean_text(text)])
        pred  = model.predict(vec)[0]
        label = "SPAM" if pred == 1 else "HAM "
        mark  = "OK" if (pred == 1) == (actual == "spam") else "!!"
        snippet = text[:58] + "..." if len(text) > 58 else text
        print(f"  [{label}] {mark}  {snippet}")
    print("-" * 65 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print("=" * 65)
    print("   SMS SPAM DETECTION — NLP + Machine Learning Pipeline")
    print("=" * 65)

    df = load_data(DATA_PATH)
    df = preprocess(df)

    plot_class_distribution(df)
    plot_top_terms(df)

    X_train, X_test, y_train, y_test, tfidf = build_features(df)

    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_accuracy_comparison(results)

    best_name = save_best_model(results, tfidf)
    print_summary(results)

    demo_predict(results[best_name]["model"], tfidf)

    log.info("Pipeline complete in %.1fs — outputs saved to /%s", time.time() - t_start, OUT_DIR)


if __name__ == "__main__":
    main()
