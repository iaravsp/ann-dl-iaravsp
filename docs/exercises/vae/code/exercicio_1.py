"""Exercício 1 — um VAE no MNIST com d = 16.

Treina o modelo, registra os dois termos da perda separadamente em treino e
validação, avalia no teste uma única vez e gera as Figuras 1, 2 e 3.

Para rodar, a partir da raiz do repositório:

    python docs/exercises/vae/code/exercicio_1.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from vae import (
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    INPUT_DIM,
    RESULTS_DIR,
    as_tensor_dataset,
    evaluate,
    flatten_dataset,
    load_splits,
    train_vae,
)

LATENT_DIM = 16
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
BETA = 1.0


def figura_1(history) -> None:
    """Figura 1 — reconstrução x época e KL x época, treino e validação."""
    epochs = range(1, len(history.train_rec) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(epochs, history.train_rec, marker="o", ms=3, label="treino")
    axes[0].plot(epochs, history.val_rec, marker="s", ms=3, label="validação")
    axes[0].set_title("Termo de reconstrução (BCE) por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Reconstrução (nats/imagem)")
    axes[0].legend(title="Conjunto")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history.train_kl, marker="o", ms=3, label="treino")
    axes[1].plot(epochs, history.val_kl, marker="s", ms=3, label="validação")
    axes[1].set_title("Termo KL por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("KL (nats/imagem)")
    axes[1].legend(title="Conjunto")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Figura 1 — Os dois termos da perda durante o treino (d = {LATENT_DIM})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1.png", dpi=150)
    plt.close(fig)


def figura_2(model, x_test: torch.Tensor, y_test: torch.Tensor) -> list[int]:
    """Figura 2 — original e reconstrução do primeiro teste de cada dígito.

    A reconstrução decodifica mu, sem amostrar z.
    """
    indices = [int((y_test == d).nonzero()[0]) for d in range(10)]
    originals = x_test[indices]

    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(originals)
        reconstructions = model.decode(mu)

    fig, axes = plt.subplots(2, 10, figsize=(12, 3.0))
    for col, digit in enumerate(range(10)):
        axes[0, col].imshow(originals[col].view(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(str(digit), fontsize=10)
        axes[1, col].imshow(
            reconstructions[col].view(28, 28), cmap="gray", vmin=0, vmax=1
        )
        for row in (0, 1):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    axes[0, 0].set_ylabel("original", fontsize=9)
    axes[1, 0].set_ylabel("reconstrução", fontsize=9)
    fig.suptitle(
        f"Figura 2 — Originais (acima) e reconstruções de mu (abaixo), d = {LATENT_DIM}"
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2.png", dpi=150)
    plt.close(fig)
    return indices


def figura_3(model) -> None:
    """Figura 3 — grade 8 x 8 decodificada de z ~ N(0, I), semente 0."""
    torch.manual_seed(0)
    model.eval()
    samples = model.sample(64)

    fig, axes = plt.subplots(8, 8, figsize=(7.2, 7.6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].view(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Figura 3 — 64 amostras do prior z ~ N(0, I) decodificadas"
        f" (d = {LATENT_DIM}, semente 0)"
    )
    fig.supxlabel("8 colunas de amostras independentes")
    fig.supylabel("8 linhas de amostras independentes")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    train_set, val_set, test_set = load_splits()
    test_set = as_tensor_dataset(test_set)
    print(
        f"treino {len(train_set)} | validação {len(val_set)} | teste {len(test_set)}"
    )

    model, history = train_vae(
        latent_dim=LATENT_DIM,
        train_set=train_set,
        val_set=val_set,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        beta=BETA,
    )

    # O teste entra só agora, para os números finais.
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)
    test_rec, test_kl = evaluate(model, test_loader, BETA)
    print(
        f"\nteste: reconstrução {test_rec:.3f} | KL {test_kl:.3f} | "
        f"-ELBO {test_rec + test_kl:.3f} nats/imagem"
    )

    figura_1(history)
    x_test, y_test = flatten_dataset(test_set)
    indices = figura_2(model, x_test, y_test)
    print("índices usados na Figura 2:", indices)
    figura_3(model)

    torch.save(model.state_dict(), CHECKPOINTS_DIR / f"vae_d{LATENT_DIM}.pt")

    metrics = {
        "latent_dim": LATENT_DIM,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "beta": BETA,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "test_recon": test_rec,
        "test_kl": test_kl,
        "test_neg_elbo": test_rec + test_kl,
        "fig2_indices": indices,
        "history": {
            "train_rec": history.train_rec,
            "train_kl": history.train_kl,
            "val_rec": history.val_rec,
            "val_kl": history.val_kl,
        },
    }
    with open(RESULTS_DIR / "ex1.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, ensure_ascii=False)
    print("métricas salvas em", RESULTS_DIR / "ex1.json")


if __name__ == "__main__":
    main()
