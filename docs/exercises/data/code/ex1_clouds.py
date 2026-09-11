"""Exercise 1 - Point clouds: geometry and spread in 2D.

Run from the repository root:

    python docs/exercises/data/code/ex1_clouds.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # renderiza sem abrir janela

import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parents[1] / "figures"
SEED = 42
N_PER_CLASS = 100

# Parametros do enunciado: label -> (vetor de medias, vetor de desvios)
CLASSES = {
    0: (np.array([2.0, 3.0]), np.array([0.8, 2.5])),
    1: (np.array([5.0, 6.0]), np.array([1.2, 1.9])),
    2: (np.array([8.0, 1.0]), np.array([0.9, 0.9])),
    3: (np.array([15.0, 4.0]), np.array([0.5, 2.0])),
}

COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}


def save(fig, name):
    """Salva a figura em docs/exercises/data/figures/<name>."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def generate(rng, scale=1.0, n_per_class=N_PER_CLASS):
    """Amostra as quatro nuvens gaussianas.

    Parameters
    ----------
    rng : np.random.Generator
    scale : float
        Multiplica todos os desvios padrao. O item A usa 1.0; o item B
        reutiliza esta funcao com 0.5, 1.0, 2.0 e 4.0.
    n_per_class : int

    Returns
    -------
    X : np.ndarray, shape (4 * n_per_class, 2)
    y : np.ndarray, shape (4 * n_per_class,)
    """
    features, labels = [], []
    for label, (mu, sigma) in CLASSES.items():
        # loc e scale como vetores de 2 elementos: o numpy aplica um por
        # coluna, entao as duas coordenadas saem de uma vez so.
        points = rng.normal(loc=mu, scale=sigma * scale, size=(n_per_class, 2))
        features.append(points)
        labels.append(np.full(n_per_class, label))
    return np.vstack(features), np.concatenate(labels)


def plot_clouds(ax, X, y):
    """Desenha o scatter das quatro classes com os centros marcados.

    Recebe um ``ax`` (nao cria a figura) porque o item B vai chamar esta
    funcao uma vez por subplot.
    """
    for label, (mu, _) in CLASSES.items():
        mask = y == label
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            alpha=0.6,
            color=COLORS[label],
            label=f"Class {label}",
        )
        # Centro verdadeiro (o mu do enunciado, nao a media empirica).
        # Sem label= para nao duplicar as entradas da legenda.
        ax.scatter(
            *mu,
            marker="X",
            s=200,
            color=COLORS[label],
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )
    ax.set_title("Figure 1 - four Gaussian clouds (class centers marked)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend()


def main():
    rng = np.random.default_rng(SEED)
    X, y = generate(rng)

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"pontos por classe: {np.bincount(y)}")

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_clouds(ax, X, y)
    save(fig, "fig1_clouds.png")


if __name__ == "__main__":
    main()
