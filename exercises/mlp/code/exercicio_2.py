"""Exercício 2 — dados, baseline, treinamento e Figuras 1 a 3.

Execute da raiz do repositório:
    python docs/exercises/mlp/code/exercicio_2.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlp import MLP

FIGURES = Path(__file__).resolve().parents[1] / "figures"
COLORS = ["tab:blue", "tab:orange"]
LAYERS = [2, 8, 1]
LEARNING_RATE = 0.1
EPOCHS = 2000


def generate_data():
    """Parâmetros e divisão fixados pelo enunciado, sem alterações."""
    centers = np.array([[0., 0.], [3., 3.], [-3., -3.]])
    X, c = make_blobs(n_samples=[500, 250, 250], centers=centers,
                      cluster_std=1.2, random_state=42)
    y = (c > 0).astype(int)
    split = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X, y, split


def save_figure(fig, filename, figures_dir=FIGURES):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures_dir / filename, dpi=160)
    plt.close(fig)


def plot_data(X, y, figures_dir=FIGURES):
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, color in enumerate(COLORS):
        points = X[y == label]
        ax.scatter(points[:, 0], points[:, 1], s=18, alpha=0.65,
                   color=color, label=f"Classe {label}")
    ax.set_title("Figura 1 — Dados de classificação binária")
    ax.set_xlabel("Característica 1 ($x_1$)")
    ax.set_ylabel("Característica 2 ($x_2$)")
    ax.legend(title="Classe correta")
    ax.set_aspect("equal", adjustable="box")
    save_figure(fig, "fig1.png", figures_dir)


def plot_loss(history, figures_dir=FIGURES):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.arange(len(history)), history, color="tab:purple",
            label="MLP — classes 0 e 1 (treino)")
    ax.set_title("Figura 2 — Perda durante o treinamento")
    ax.set_xlabel("Época (0 = inicialização)")
    ax.set_ylabel("Entropia cruzada binária média")
    ax.legend()
    ax.grid(alpha=0.2)
    save_figure(fig, "fig2.png", figures_dir)


def plot_boundary(model, X, X_test, y_test, figures_dir=FIGURES):
    # A malha apenas visualiza previsões; não participa do treinamento.
    x1 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 250)
    x2 = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 250)
    xx, yy = np.meshgrid(x1, x2)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    probabilities, _, _ = model.forward(grid)
    probabilities = probabilities[:, 0].reshape(xx.shape)
    regions = (probabilities >= 0.5).astype(int)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, regions, levels=[-0.5, 0.5, 1.5],
                cmap=ListedColormap(["#dbeafe", "#ffedd5"]), alpha=0.8)
    ax.contour(xx, yy, probabilities, levels=[0.5], colors="black", linewidths=1)
    for label, color in enumerate(COLORS):
        points = X_test[y_test == label]
        ax.scatter(points[:, 0], points[:, 1], color=color, s=28,
                   edgecolors="white", linewidths=0.5,
                   label=f"Classe {label} — teste")
    ax.set_title("Figura 3 — Fronteira da MLP e pontos de teste")
    ax.set_xlabel("Característica 1 ($x_1$)")
    ax.set_ylabel("Característica 2 ($x_2$)")
    ax.legend(title="Classe correta")
    ax.set_aspect("equal", adjustable="box")
    save_figure(fig, "fig3.png", figures_dir)


def run(figures_dir=FIGURES, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    X, y, (X_train, X_test, y_train, y_test) = generate_data()
    baseline = LogisticRegression(max_iter=1000, random_state=42)
    baseline.fit(X_train, y_train)
    baseline_accuracy = float(np.mean(baseline.predict(X_test) == y_test))
    model = MLP(LAYERS, rng)
    history = model.fit(X_train, y_train, learning_rate=LEARNING_RATE, epochs=EPOCHS)
    predictions = model.predict(X_test)
    accuracy = float(np.mean(predictions == y_test))
    plot_data(X, y, figures_dir)
    plot_loss(history, figures_dir)
    plot_boundary(model, X, X_test, y_test, figures_dir)
    results = {
        "architecture": LAYERS,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "parameters": model.parameter_count,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "baseline_accuracy": baseline_accuracy,
        "mlp_accuracy": accuracy,
        "correct_test_predictions": int(np.sum(predictions == y_test)),
        "initial_train_loss": float(history[0]),
        "final_train_loss": float(history[-1]),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results, model, history


if __name__ == "__main__":
    run()
