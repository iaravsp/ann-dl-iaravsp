"""Exercício 3 — classificação multiclasse com a mesma MLP.

Execute da raiz do repositório:
    python docs/exercises/mlp/code/exercicio_3.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from mlp import MLP
from exercicio_2 import LAYERS as BINARY_LAYERS, save_figure

FIGURES = Path(__file__).resolve().parents[1] / "figures"
LAYERS = [4, 16, 3]
LEARNING_RATE = 0.1
EPOCHS = 2000
COLORS = ["tab:blue", "tab:orange", "tab:green"]


def softmax(logits):
    """Converte os três logits em probabilidades que somam 1 por linha."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def categorical_cross_entropy(logits, probabilities, y):
    """Retorna a perda média e a derivada (P-Y)/N em relação aos logits."""
    labels = np.asarray(y, dtype=int)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -np.mean(log_probabilities[np.arange(len(labels)), labels])
    targets = np.zeros_like(probabilities)
    targets[np.arange(len(labels)), labels] = 1.0
    delta = (probabilities - targets) / len(labels)
    return float(loss), delta


def report_rng(seed=42):
    """Retoma a sequência do gerador após a inicialização do Exercício 2.

    Só a inicialização da MLP consome rng no exercício anterior. Para
    executar este script sozinho, reproduzimos aqueles mesmos sorteios;
    não é necessário treinar novamente o modelo binário.
    """
    rng = np.random.default_rng(seed)
    MLP(BINARY_LAYERS, rng)
    return rng


def generate_data():
    """Dados e split com todos os parâmetros fixados pelo enunciado."""
    X, y = make_classification(
        n_samples=1500, n_features=4, n_informative=4,
        n_redundant=0, n_repeated=0, n_classes=3,
        n_clusters_per_class=2, class_sep=1.0, random_state=42,
    )
    split = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X, y, split


def plot_loss(history, figures_dir=FIGURES):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.arange(len(history)), history, color="tab:purple",
            label="MLP — classes 0, 1 e 2 (treino)")
    ax.set_title("Figura 4 — Perda no treinamento multiclasse")
    ax.set_xlabel("Época (0 = inicialização)")
    ax.set_ylabel("Entropia cruzada categórica média")
    ax.legend()
    ax.grid(alpha=0.2)
    save_figure(fig, "fig4.png", figures_dir)


def plot_confusion(matrix, figures_dir=FIGURES):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    heatmap = ax.imshow(matrix, cmap="Blues", vmin=0)
    threshold = matrix.max() / 2
    for row in range(3):
        for column in range(3):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                    fontsize=13)
    labels = ["Classe 0", "Classe 1", "Classe 2"]
    ax.set_xticks(range(3), labels=labels)
    ax.set_yticks(range(3), labels=labels)
    for tick, color in zip(ax.get_xticklabels(), COLORS):
        tick.set_color(color)
    for tick, color in zip(ax.get_yticklabels(), COLORS):
        tick.set_color(color)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title("Figura 5 — Matriz de confusão no teste")
    fig.colorbar(heatmap, ax=ax, label="Número de amostras")
    ax.legend(handles=[Patch(color=color, label=f"Classe {i}")
                       for i, color in enumerate(COLORS)],
              loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    save_figure(fig, "fig5.png", figures_dir)


def most_confused_pairs(matrix):
    """Soma as confusões nas duas direções para cada par de classes."""
    pairs = [(a, b, int(matrix[a, b] + matrix[b, a]))
             for a, b in [(0, 1), (0, 2), (1, 2)]]
    largest = max(count for a, b, count in pairs)
    return [{"classes": [a, b], "count": count}
            for a, b, count in pairs if count == largest]


def run(figures_dir=FIGURES, rng=None):
    if rng is None:
        rng = report_rng()
    X, y, (X_train, X_test, y_train, y_test) = generate_data()
    baseline = LogisticRegression(max_iter=1000, random_state=42)
    baseline.fit(X_train, y_train)
    baseline_accuracy = float(np.mean(baseline.predict(X_test) == y_test))
    model = MLP(LAYERS, rng, output_activation=softmax,
                loss_and_delta=categorical_cross_entropy)
    history = model.fit(X_train, y_train, learning_rate=LEARNING_RATE, epochs=EPOCHS)
    predictions = model.predict(X_test)
    accuracy = float(np.mean(predictions == y_test))
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    plot_loss(history, figures_dir)
    plot_confusion(matrix, figures_dir)
    results = {
        "architecture": LAYERS,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "parameters": model.parameter_count,
        "samples_per_class": np.bincount(y).tolist(),
        "train_per_class": np.bincount(y_train).tolist(),
        "test_per_class": np.bincount(y_test).tolist(),
        "baseline_accuracy": baseline_accuracy,
        "mlp_accuracy": accuracy,
        "correct_test_predictions": int(np.sum(predictions == y_test)),
        "initial_train_loss": float(history[0]),
        "final_train_loss": float(history[-1]),
        "confusion_matrix": matrix.tolist(),
        "most_confused_pairs": most_confused_pairs(matrix),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results, model, history, matrix


if __name__ == "__main__":
    run()
