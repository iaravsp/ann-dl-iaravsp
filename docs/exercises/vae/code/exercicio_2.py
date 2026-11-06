"""Exercício 2 — um espaço latente que dá para ver (d = 2).

Treina o mesmo modelo do Exercício 1 com d = 2, mantendo dados, semente, beta
e orçamento de treino, e gera as Figuras 4, 5 e 6. A linha latente da Figura 6
usa o modelo d = 16 do Exercício 1 (o checkpoint salvo por `exercicio_1.py`;
se ele não existir, o modelo é treinado de novo aqui com a mesma semente).

Também mede, com um k-NN sobre as coordenadas latentes, quais dígitos se
sobrepõem na Figura 4 e quais ocupam o plano da Figura 5 — os números que
sustentam as análises do item 2E.

Para rodar, a partir da raiz do repositório:

    python docs/exercises/vae/code/exercicio_2.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from vae import (
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    VAE,
    as_tensor_dataset,
    evaluate,
    flatten_dataset,
    load_splits,
    train_vae,
)

LATENT_DIM = 2
REFERENCE_DIM = 16          # modelo do Exercício 1, usado na Figura 6
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
BETA = 1.0
GRID_SIDE = 15              # Figura 5: grade 15 x 15
N_STEPS = 10                # Figura 6: 10 imagens por linha
DIGIT_A, DIGIT_B = 4, 9     # dígitos interpolados na Figura 6


@torch.no_grad()
def encode_mu(model: VAE, x: torch.Tensor) -> torch.Tensor:
    """Devolve mu para um lote grande de imagens, sem amostrar z."""
    model.eval()
    chunks = [model.encode(x[i : i + 2048])[0] for i in range(0, len(x), 2048)]
    return torch.cat(chunks)


def figura_4(mu_test: np.ndarray, y_test: np.ndarray) -> None:
    """Figura 4 — scatter de mu das 10 000 imagens de teste, por dígito."""
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    cmap = plt.get_cmap("tab10")
    for digit in range(10):
        mask = y_test == digit
        ax.scatter(
            mu_test[mask, 0],
            mu_test[mask, 1],
            s=4,
            alpha=0.6,
            color=cmap(digit),
            label=str(digit),
        )
    ax.set_title("Figura 4 — Conjunto de teste codificado no latente 2-D (mu)")
    ax.set_xlabel("$\\mu_1$")
    ax.set_ylabel("$\\mu_2$")
    ax.legend(title="Dígito", markerscale=3, loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4.png", dpi=150)
    plt.close(fig)


def figura_5(model: VAE) -> np.ndarray:
    """Figura 5 — plano latente decodificado na grade do quantil.

    Os pontos não são igualmente espaçados em z: usamos z = Phi^-1(q), com q
    em 15 valores igualmente espaçados em [0,05; 0,95]. Devolve a grade de
    coordenadas latentes usada, para a análise posterior.
    """
    q = np.linspace(0.05, 0.95, GRID_SIDE)
    grid = norm.ppf(q)                       # inversa da CDF da N(0, 1)

    # z1 cresce para a direita; z2 cresce para cima (daí o grid invertido).
    z_values = np.array(
        [[grid[i], grid[j]] for j in reversed(range(GRID_SIDE)) for i in range(GRID_SIDE)],
        dtype=np.float32,
    )

    model.eval()
    with torch.no_grad():
        decoded = model.decode(torch.from_numpy(z_values)).view(-1, 28, 28).numpy()

    canvas = np.zeros((GRID_SIDE * 28, GRID_SIDE * 28))
    for k, image in enumerate(decoded):
        row, col = divmod(k, GRID_SIDE)
        canvas[row * 28 : (row + 1) * 28, col * 28 : (col + 1) * 28] = image

    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ticks = np.arange(GRID_SIDE) * 28 + 14
    labels = [f"{v:.2f}" for v in grid]
    ax.set_xticks(ticks[::2])
    ax.set_xticklabels(labels[::2], fontsize=7)
    ax.set_yticks(ticks[::2])
    ax.set_yticklabels(labels[::-1][::2], fontsize=7)
    ax.set_title("Figura 5 — Plano latente decodificado na grade $\\Phi^{-1}(q)$")
    ax.set_xlabel("$z_1 = \\Phi^{-1}(q)$")
    ax.set_ylabel("$z_2 = \\Phi^{-1}(q)$")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5.png", dpi=150)
    plt.close(fig)
    return z_values


def figura_6(
    model_d16: VAE,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
) -> tuple[int, int]:
    """Figura 6 — interpolação em pixels e no latente do modelo d = 16."""
    idx_a = int((y_test == DIGIT_A).nonzero()[0])
    idx_b = int((y_test == DIGIT_B).nonzero()[0])
    x_a, x_b = x_test[idx_a], x_test[idx_b]

    ts = torch.linspace(0.0, 1.0, N_STEPS)

    # Linha de cima: média ponderada dos pixels.
    pixel_row = torch.stack([(1 - t) * x_a + t * x_b for t in ts])

    # Linha de baixo: a mesma mistura, mas entre os mu do modelo d = 16.
    model_d16.eval()
    with torch.no_grad():
        mu_pair, _ = model_d16.encode(torch.stack([x_a, x_b]))
        mu_a, mu_b = mu_pair[0], mu_pair[1]
        z_path = torch.stack([(1 - t) * mu_a + t * mu_b for t in ts])
        latent_row = model_d16.decode(z_path)

    fig, axes = plt.subplots(2, N_STEPS, figsize=(12, 3.1))
    for col, t in enumerate(ts):
        axes[0, col].imshow(pixel_row[col].view(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"t = {t:.2f}", fontsize=8)
        axes[1, col].imshow(latent_row[col].view(28, 28), cmap="gray", vmin=0, vmax=1)
        for row in (0, 1):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    axes[0, 0].set_ylabel("pixels", fontsize=9)
    axes[1, 0].set_ylabel("latente $d=16$", fontsize=9)
    fig.suptitle(
        f"Figura 6 — Interpolação entre o teste {idx_a} (dígito {DIGIT_A}) e o "
        f"teste {idx_b} (dígito {DIGIT_B})"
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6.png", dpi=150)
    plt.close(fig)
    return idx_a, idx_b


def medir_sobreposicao(
    mu_train: np.ndarray,
    y_train: np.ndarray,
    mu_test: np.ndarray,
    y_test: np.ndarray,
    z_grid: np.ndarray,
) -> dict:
    """Quantifica a sobreposição no latente 2-D com um k-NN (k = 25).

    O classificador é ajustado nos mu de treino e aplicado (a) aos mu de teste,
    para medir quais dígitos se confundem na Figura 4, e (b) às 225 células da
    grade da Figura 5, para ver quais dígitos ocupam o plano.
    """
    knn = KNeighborsClassifier(n_neighbors=25).fit(mu_train, y_train)
    pred_test = knn.predict(mu_test)
    cm = confusion_matrix(y_test, pred_test, labels=list(range(10)))

    # Confusões somadas nos dois sentidos, para cada par de dígitos.
    pairs = []
    for a in range(10):
        for b in range(a + 1, 10):
            pairs.append(((a, b), int(cm[a, b] + cm[b, a])))
    pairs.sort(key=lambda item: item[1], reverse=True)

    grid_pred = knn.predict(z_grid)
    cells = {int(d): int((grid_pred == d).sum()) for d in range(10)}

    return {
        "knn_test_accuracy": float((pred_test == y_test).mean()),
        "top_confused_pairs": [[list(p), n] for p, n in pairs[:6]],
        "grid_cells_per_digit": cells,
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_set, val_set, test_set = load_splits()
    train_set = as_tensor_dataset(train_set)
    test_set = as_tensor_dataset(test_set)

    # --- A: o mesmo modelo, só com d = 2 -----------------------------------
    model, history = train_vae(
        latent_dim=LATENT_DIM,
        train_set=train_set,
        val_set=val_set,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        beta=BETA,
    )

    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)
    test_rec, test_kl = evaluate(model, test_loader, BETA)
    print(
        f"\nteste (d = 2): reconstrução {test_rec:.3f} | KL {test_kl:.3f} | "
        f"-ELBO {test_rec + test_kl:.3f} nats/imagem"
    )

    # --- B e C -------------------------------------------------------------
    x_test, y_test = flatten_dataset(test_set)
    mu_test = encode_mu(model, x_test).numpy()
    figura_4(mu_test, y_test.numpy())
    z_grid = figura_5(model)

    # --- D: a linha latente vem do modelo d = 16 do Exercício 1 ------------
    checkpoint = CHECKPOINTS_DIR / f"vae_d{REFERENCE_DIM}.pt"
    model_d16 = VAE(REFERENCE_DIM)
    if checkpoint.exists():
        model_d16.load_state_dict(torch.load(checkpoint))
        print(f"modelo d = {REFERENCE_DIM} carregado de {checkpoint.name}")
    else:
        print(f"checkpoint ausente; treinando o modelo d = {REFERENCE_DIM}")
        model_d16, _ = train_vae(
            latent_dim=REFERENCE_DIM,
            train_set=train_set,
            val_set=val_set,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            beta=BETA,
            verbose=False,
        )
    idx_a, idx_b = figura_6(model_d16, x_test, y_test)
    print(f"índices interpolados: {idx_a} (dígito {DIGIT_A}) e {idx_b} (dígito {DIGIT_B})")

    # --- E: números que sustentam a análise --------------------------------
    x_train, y_train = flatten_dataset(train_set)
    mu_train = encode_mu(model, x_train).numpy()
    overlap = medir_sobreposicao(
        mu_train, y_train.numpy(), mu_test, y_test.numpy(), z_grid
    )
    print("acurácia do k-NN no latente 2-D:", round(overlap["knn_test_accuracy"], 4))
    print("pares mais confundidos:", overlap["top_confused_pairs"])
    print("células da grade por dígito:", overlap["grid_cells_per_digit"])

    metrics = {
        "latent_dim": LATENT_DIM,
        "epochs": EPOCHS,
        "test_recon": test_rec,
        "test_kl": test_kl,
        "test_neg_elbo": test_rec + test_kl,
        "interpolated_indices": [idx_a, idx_b],
        "interpolated_digits": [DIGIT_A, DIGIT_B],
        "history": {
            "train_rec": history.train_rec,
            "train_kl": history.train_kl,
            "val_rec": history.val_rec,
            "val_kl": history.val_kl,
        },
        "overlap": overlap,
    }
    with open(RESULTS_DIR / "ex2.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, ensure_ascii=False)
    print("métricas salvas em", RESULTS_DIR / "ex2.json")


if __name__ == "__main__":
    main()
