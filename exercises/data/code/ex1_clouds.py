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

SCALES = [0.5, 1.0, 2.0, 4.0]

# Centros verdadeiros empilhados, na ordem dos rotulos. Usado tanto na taxa de
# mistura quanto no desenho das fronteiras.
CENTERS = np.vstack([mu for mu, _ in CLASSES.values()])


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


def nearest_center(points):
    """Rotulo do centro verdadeiro mais proximo de cada ponto.

    Distancia de cada ponto a cada um dos 4 centros via broadcasting:
    (n, 1, 2) - (1, 4, 2) -> (n, 4, 2), norma no ultimo eixo -> (n, 4).
    """
    deltas = points[:, None, :] - CENTERS[None, :, :]
    return np.linalg.norm(deltas, axis=2).argmin(axis=1)


def mixing_rate(X, y):
    """Fracao de pontos cujo centro mais proximo pertence a outra classe."""
    return float(np.mean(nearest_center(X) != y))


def separation_ratios(scale=1.0):
    """Razao de separacao r_ij para os 6 pares de classes.

    r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j), com sigma_bar sendo a
    media dos dois desvios da classe. Escalar os desvios por s nao move os mu,
    entao o numerador e constante e o denominador cresce com s.
    """
    ratios = {}
    for i in CLASSES:
        for j in CLASSES:
            if i < j:
                mu_i, sigma_i = CLASSES[i]
                mu_j, sigma_j = CLASSES[j]
                spread = scale * (sigma_i.mean() + sigma_j.mean())
                ratios[(i, j)] = float(np.linalg.norm(mu_i - mu_j) / spread)
    return ratios


def figure2(rng):
    """Quatro subplots, um por escala, com os mesmos limites de eixo."""
    datasets = [(s, generate(rng, scale=s)) for s in SCALES]

    # Limites comuns calculados sobre todas as escalas, senao a comparacao
    # visual entre os paineis nao vale nada.
    all_points = np.vstack([X for _, (X, _) in datasets])
    pad = 1.0
    xlim = (all_points[:, 0].min() - pad, all_points[:, 0].max() + pad)
    ylim = (all_points[:, 1].min() - pad, all_points[:, 1].max() + pad)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    rates = {}
    for ax, (s, (X, y)) in zip(axes.ravel(), datasets):
        rates[s] = mixing_rate(X, y)
        plot_clouds(ax, X, y)
        ax.set_title(f"s = {s}  (taxa de mistura = {rates[s]:.3f})")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    save(fig, "fig2_scales.png")
    return rates


def figure3(rates):
    """Taxa de mistura em funcao da escala."""
    fig, ax = plt.subplots(figsize=(8, 6))
    scales = sorted(rates)
    ax.plot(scales, [rates[s] for s in scales], marker="o", color="tab:purple")
    for s in scales:
        ax.annotate(f"{rates[s]:.3f}", (s, rates[s]),
                    textcoords="offset points", xytext=(8, -4))
    ax.set_title("Figure 3 - taxa de mistura vs. escala dos desvios")
    ax.set_xlabel("escala $s$ aplicada aos desvios padrao")
    ax.set_ylabel("taxa de mistura")
    ax.set_xticks(scales)
    ax.grid(alpha=0.3)
    save(fig, "fig3_mixing_rate.png")


def figure1_boundaries(X, y):
    """Figura 1 com as fronteiras do classificador de centro mais proximo.

    Serve de esboco para o item C: e a particao mais simples possivel do plano
    que respeita os quatro centros.
    """
    pad = 1.5
    xs = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 400)
    ys = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 400)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    regions = nearest_center(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, regions, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
                colors=[COLORS[k] for k in range(4)], alpha=0.15)
    ax.contour(xx, yy, regions, levels=[0.5, 1.5, 2.5],
               colors="black", linewidths=1.2)
    plot_clouds(ax, X, y)
    ax.set_title("Fronteiras de decisao por centro mais proximo")
    save(fig, "fig1_boundaries.png")


def main():
    rng = np.random.default_rng(SEED)
    X, y = generate(rng)

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"pontos por classe: {np.bincount(y)}")

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_clouds(ax, X, y)
    save(fig, "fig1_clouds.png")

    print("\n=== B - razao de separacao em s = 1 ===")
    ratios = separation_ratios(scale=1.0)
    for (i, j), r in sorted(ratios.items(), key=lambda kv: kv[1]):
        print(f"  r_{i}{j} = {r:.3f}")
    pair, smallest = min(ratios.items(), key=lambda kv: kv[1])
    print(f"menor: r_{pair[0]}{pair[1]} = {smallest:.3f}")
    print(f"previsao em s = 2: {smallest / 2:.3f}")
    print(f"conferindo:        {separation_ratios(scale=2.0)[pair]:.3f}")

    print("\n=== B - taxa de mistura por escala ===")
    rates = figure2(rng)
    for s in SCALES:
        print(f"  s = {s}: {rates[s]:.3f}")
    figure3(rates)

    print("\n=== C - esboco das fronteiras ===")
    figure1_boundaries(X, y)


if __name__ == "__main__":
    main()
