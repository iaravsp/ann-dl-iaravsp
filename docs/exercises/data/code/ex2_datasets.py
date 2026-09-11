"""Exercise 2 - Non-linearity in higher dimensions.

Gera dois datasets em 5D com estruturas bem diferentes, projeta os dois com PCA
e mede a geometria de cada um.

Run from the repository root:

    python docs/exercises/data/code/ex2_datasets.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

FIGURES = Path(__file__).resolve().parents[1] / "figures"
SEED = 42
N_PER_CLASS = 500
DIM = 5

# Dataset I: duas gaussianas multivariadas deslocadas uma da outra.
MU_A = np.zeros(DIM)
SIGMA_A = np.array(
    [
        [1.0, 0.8, 0.1, 0.0, 0.0],
        [0.8, 1.0, 0.3, 0.0, 0.0],
        [0.1, 0.3, 1.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 1.0, 0.2],
        [0.0, 0.0, 0.0, 0.2, 1.0],
    ]
)

MU_B = np.full(DIM, 1.5)
SIGMA_B = np.array(
    [
        [1.5, -0.7, 0.2, 0.0, 0.0],
        [-0.7, 1.5, 0.4, 0.0, 0.0],
        [0.2, 0.4, 1.5, 0.6, 0.0],
        [0.0, 0.0, 0.6, 1.5, 0.3],
        [0.0, 0.0, 0.0, 0.3, 1.5],
    ]
)

# Dataset II: duas cascas esfericas concentricas.
RADIUS_C = (2.0, 0.4)  # (media, desvio) do raio da classe C (nucleo)
RADIUS_D = (5.0, 0.4)  # (media, desvio) do raio da classe D (casca)


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def dataset_i(rng, n_per_class=N_PER_CLASS):
    """Duas gaussianas multivariadas: A na origem, B deslocada para [1.5]*5."""
    class_a = rng.multivariate_normal(MU_A, SIGMA_A, size=n_per_class)
    class_b = rng.multivariate_normal(MU_B, SIGMA_B, size=n_per_class)
    X = np.vstack([class_a, class_b])
    y = np.concatenate([np.zeros(n_per_class, int), np.ones(n_per_class, int)])
    return X, y


def shell(rng, mean_radius, std_radius, n):
    """Amostra n pontos numa casca esferica de raio aleatorio.

    A direcao vem de uma normal isotropica normalizada: como N(0, I) e
    esfericamente simetrica, nenhuma direcao e privilegiada, entao v/||v||
    cai uniformemente sobre a esfera unitaria. O raio e sorteado separado e
    multiplica essa direcao.
    """
    directions = rng.normal(size=(n, DIM))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.normal(mean_radius, std_radius, size=(n, 1))
    return radii * directions


def dataset_ii(rng, n_per_class=N_PER_CLASS):
    """Duas cascas concentricas: C com raio ~2 e D com raio ~5."""
    class_c = shell(rng, *RADIUS_C, n_per_class)
    class_d = shell(rng, *RADIUS_D, n_per_class)
    X = np.vstack([class_c, class_d])
    y = np.concatenate([np.zeros(n_per_class, int), np.ones(n_per_class, int)])
    return X, y


def center_distance(X, y):
    """Distancia entre as medias empiricas das duas classes, em 5D."""
    return float(np.linalg.norm(X[y == 0].mean(axis=0) - X[y == 1].mean(axis=0)))


def project(X, y, ax, names, title):
    """Projeta em 2D com PCA, desenha o scatter e devolve a variancia explicada."""
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    for value, name in enumerate(names):
        mask = y == value
        ax.scatter(Z[mask, 0], Z[mask, 1], alpha=0.5, s=14, label=name)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    return pca.explained_variance_ratio_


def figure4(data_i, data_ii):
    """Projecoes PCA dos dois datasets, lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    var_i = project(*data_i, axes[0], ["Class A", "Class B"],
                    "Dataset I - gaussianas deslocadas")
    var_ii = project(*data_ii, axes[1], ["Class C (nucleo)", "Class D (casca)"],
                     "Dataset II - cascas concentricas")
    save(fig, "fig4_pca.png")
    return var_i, var_ii


def figure5(data_i, data_ii):
    """Histogramas sobrepostos da norma ||x|| em 5D."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (X, y), names, title in [
        (axes[0], data_i, ["Class A", "Class B"], "Dataset I"),
        (axes[1], data_ii, ["Class C (nucleo)", "Class D (casca)"], "Dataset II"),
    ]:
        for value, name in enumerate(names):
            radii = np.linalg.norm(X[y == value], axis=1)
            ax.hist(radii, bins=40, alpha=0.6, label=name)
        ax.set_title(f"{title} - distribuicao de $||x||$")
        ax.set_xlabel("$||x||$")
        ax.set_ylabel("contagem")
        ax.legend()
    save(fig, "fig5_radii.png")


def main():
    rng = np.random.default_rng(SEED)

    data_i = dataset_i(rng)
    data_ii = dataset_ii(rng)
    print(f"Dataset I:  X {data_i[0].shape}, y {data_i[1].shape}")
    print(f"Dataset II: X {data_ii[0].shape}, y {data_ii[1].shape}")

    print("\n=== C - geometria em 5D ===")
    d_i = center_distance(*data_i)
    d_ii = center_distance(*data_ii)
    print(f"distancia entre centros - Dataset I:  {d_i:.4f}")
    print(f"distancia entre centros - Dataset II: {d_ii:.4f}")
    print(f"  (teorico do Dataset I, 1.5*sqrt(5) = {1.5 * np.sqrt(5):.4f})")

    var_i, var_ii = figure4(data_i, data_ii)
    print(f"variancia explicada PC1+PC2 - Dataset I:  {var_i.sum():.4f}"
          f"  (PC1 {var_i[0]:.4f}, PC2 {var_i[1]:.4f})")
    print(f"variancia explicada PC1+PC2 - Dataset II: {var_ii.sum():.4f}"
          f"  (PC1 {var_ii[0]:.4f}, PC2 {var_ii[1]:.4f})")

    figure5(data_i, data_ii)

    X2, y2 = data_ii
    radii_c = np.linalg.norm(X2[y2 == 0], axis=1)
    radii_d = np.linalg.norm(X2[y2 == 1], axis=1)
    print(f"raio medio - classe C: {radii_c.mean():.4f}")
    print(f"raio medio - classe D: {radii_d.mean():.4f}")
    print(f"sobreposicao de raios: {np.sum(radii_c > radii_d.min())} pontos")


if __name__ == "__main__":
    main()
