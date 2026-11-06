"""Variational Autoencoder (VAE) escrito à mão para o exercício de VAE.

O módulo concentra as partes específicas do VAE, todas implementadas aqui:

* a arquitetura 784 -> 512 -> 256 -> (mu, logvar) -> 256 -> 512 -> 784;
* a reparametrização z = mu + sigma * eps, com eps ~ N(0, I);
* os dois termos da perda: a reconstrução (BCE somada nos 784 pixels) e a
  divergência KL em forma fechada, escrita termo a termo;
* a amostragem a partir do prior.

Nenhuma classe de VAE de biblioteca e nenhuma função de KL pronta
(`torch.distributions.kl_divergence`) é usada. O autograd do PyTorch cuida
apenas do backward, o que o enunciado permite.

Todas as perdas são devolvidas em nats por imagem: somadas sobre os 784
pixels (e sobre as dimensões latentes, no caso do KL) e só depois
promediadas sobre as imagens do lote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data"
CODE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = CODE_DIR.parent / "figures"
RESULTS_DIR = CODE_DIR / "results"
CHECKPOINTS_DIR = CODE_DIR / "checkpoints"

INPUT_DIM = 784


# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------
def load_splits() -> tuple[Dataset, Dataset, Dataset]:
    """Carrega o MNIST e devolve (treino, validação, teste).

    A divisão 50 000 / 10 000 e o gerador com semente 42 são exatamente os
    do enunciado. As 10 000 imagens oficiais de teste formam o teste.
    """
    transform = transforms.ToTensor()  # já coloca os pixels em [0, 1]
    train_full = datasets.MNIST(
        str(DATA_ROOT), train=True, download=True, transform=transform
    )
    test = datasets.MNIST(
        str(DATA_ROOT), train=False, download=True, transform=transform
    )
    train, val = random_split(
        train_full,
        [50_000, 10_000],
        generator=torch.Generator().manual_seed(42),
    )
    return train, val, test


def flatten_dataset(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Empilha um dataset do MNIST em tensores (N, 784) e (N,)."""
    if isinstance(dataset, TensorDataset):
        return dataset.tensors[0], dataset.tensors[1]
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    images, labels = [], []
    for x, y in loader:
        images.append(x.view(x.size(0), INPUT_DIM))
        labels.append(y)
    return torch.cat(images), torch.cat(labels)


def as_tensor_dataset(dataset: Dataset) -> TensorDataset:
    """Materializa o dataset em memória como tensores (N, 784) e (N,).

    O `datasets.MNIST` aplica `ToTensor()` imagem por imagem a cada acesso, o
    que faria a conversão das 60 000 imagens PIL ser repetida em todas as
    épocas e dominar o tempo de treino. Convertendo uma única vez, a ordem dos
    exemplos é preservada: o índice i continua apontando para a mesma imagem,
    então o embaralhamento do DataLoader (com gerador de semente fixa) produz
    exatamente os mesmos lotes.
    """
    if isinstance(dataset, TensorDataset):
        return dataset
    images, labels = flatten_dataset(dataset)
    return TensorDataset(images, labels)


# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------
class VAE(nn.Module):
    """VAE com encoder e decoder densos e saída sigmoid.

    Args:
        latent_dim: dimensão `d` do espaço latente (16 no Ex. 1, 2 no Ex. 2).
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 784 -> 512 -> 256, com ReLU depois de cada camada.
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        # Duas cabeças lineares 256 -> d: a média e o log da variância.
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder: d -> 256 -> 512 -> 784, ReLU nas ocultas e sigmoid na saída.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, INPUT_DIM),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Devolve mu e log sigma^2 da posterior aproximada q(z | x)."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Truque da reparametrização: z = mu + sigma * eps, eps ~ N(0, I).

        Amostrar z diretamente de N(mu, sigma^2) não permitiria propagar o
        gradiente até mu e logvar. Escrevendo z como uma função determinística
        de (mu, sigma) e de um ruído externo eps, o caminho do gradiente fica
        aberto e o sorteio acontece fora dos parâmetros.
        """
        sigma = torch.exp(0.5 * logvar)      # sigma = exp(logvar / 2)
        eps = torch.randn_like(sigma)        # ruído padrão, mesmo shape
        return mu + sigma * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Mapeia o latente de volta para probabilidades de pixel em [0, 1]."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Passagem completa: devolve a reconstrução, mu e logvar."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        """Decodifica n amostras do prior z ~ N(0, I)."""
        z = torch.randn(n, self.latent_dim)
        return self.decode(z)


# ----------------------------------------------------------------------------
# Perda
# ----------------------------------------------------------------------------
def reconstruction_term(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Termo de reconstrução: BCE somada nos 784 pixels, média no lote.

    É o -log p(x | z) sob uma verossimilhança de Bernoulli por pixel, que é o
    que a saída sigmoid representa. `reduction="sum"` soma pixels e imagens;
    a divisão por `x.size(0)` deixa o resultado em nats por imagem. Usar
    `reduction="mean"` dividiria também pelos 784 pixels e encolheria este
    termo 784 vezes em relação ao KL, o que equivale a um beta ~= 784.
    """
    return F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)


def kl_term(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)) em forma fechada, somada nas d dimensões.

    Para gaussianas diagonais,

        KL = 1/2 * sum_i (mu_i^2 + sigma_i^2 - 1 - log sigma_i^2),

    com sigma_i^2 = exp(logvar_i) e log sigma_i^2 = logvar_i. A expressão é
    escrita aqui à mão, sem `torch.distributions`.
    """
    sigma_sq = torch.exp(logvar)
    per_dim = mu.pow(2) + sigma_sq - 1.0 - logvar
    return 0.5 * per_dim.sum() / mu.size(0)


def vae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Devolve (perda total, reconstrução, KL), todas em nats por imagem."""
    rec = reconstruction_term(x_hat, x)
    kl = kl_term(mu, logvar)
    return rec + beta * kl, rec, kl


# ----------------------------------------------------------------------------
# Treino e avaliação
# ----------------------------------------------------------------------------
@dataclass
class History:
    """Histórico dos dois termos, por época, em nats por imagem."""

    train_rec: list[float] = field(default_factory=list)
    train_kl: list[float] = field(default_factory=list)
    val_rec: list[float] = field(default_factory=list)
    val_kl: list[float] = field(default_factory=list)


@torch.no_grad()
def evaluate(
    model: VAE, loader: DataLoader, beta: float = 1.0
) -> tuple[float, float]:
    """Média dos dois termos sobre um conjunto, em nats por imagem.

    O z continua sendo amostrado (uma amostra por imagem), como no treino:
    é o mesmo estimador do ELBO, apenas sem atualizar os parâmetros.
    """
    model.eval()
    rec_sum = kl_sum = 0.0
    n_images = 0
    for x, _ in loader:
        x = x.view(x.size(0), INPUT_DIM)
        x_hat, mu, logvar = model(x)
        _, rec, kl = vae_loss(x_hat, x, mu, logvar, beta)
        rec_sum += rec.item() * x.size(0)
        kl_sum += kl.item() * x.size(0)
        n_images += x.size(0)
    return rec_sum / n_images, kl_sum / n_images


def train_vae(
    latent_dim: int,
    train_set: Dataset,
    val_set: Dataset,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    beta: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[VAE, History]:
    """Treina o VAE e registra os dois termos a cada época.

    A semente é fixada imediatamente antes de construir o modelo, como pede o
    enunciado; o embaralhamento do DataLoader usa um gerador próprio, também
    com semente fixa, para que a ordem dos lotes seja reproduzível.
    """
    torch.manual_seed(seed)
    model = VAE(latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Tensores em memória: mesma ordem, mesmos lotes, sem reconverter as
    # imagens PIL a cada época.
    train_set = as_tensor_dataset(train_set)
    val_set = as_tensor_dataset(val_set)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_set, batch_size=1024, shuffle=False)

    history = History()
    for epoch in range(1, epochs + 1):
        model.train()
        rec_sum = kl_sum = 0.0
        n_images = 0
        for x, _ in train_loader:
            x = x.view(x.size(0), INPUT_DIM)

            x_hat, mu, logvar = model(x)
            loss, rec, kl = vae_loss(x_hat, x, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Acumula ponderando pelo tamanho do lote (o último é menor).
            rec_sum += rec.item() * x.size(0)
            kl_sum += kl.item() * x.size(0)
            n_images += x.size(0)

        history.train_rec.append(rec_sum / n_images)
        history.train_kl.append(kl_sum / n_images)

        val_rec, val_kl = evaluate(model, val_loader, beta)
        history.val_rec.append(val_rec)
        history.val_kl.append(val_kl)

        if verbose:
            print(
                f"época {epoch:2d} | "
                f"treino rec {history.train_rec[-1]:7.3f} kl {history.train_kl[-1]:6.3f}"
                f" | val rec {val_rec:7.3f} kl {val_kl:6.3f}"
            )

    return model, history
