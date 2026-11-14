"""Rede e loop de treino do Projeto 2 (classificação de AVC).

`construir_mlp` monta a MLP; `treinar_mlp` treina com AdamW e parada antecipada
pela perda de validação, restaurando os melhores pesos.
"""
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset


def fixar_semente(semente):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)


def construir_mlp(ocultas=(8,), n_entradas=24, dropout=0.0, batchnorm=False,
                  vies_saida=None):
    """MLP com ReLU. Ordem em cada camada oculta: Linear → BatchNorm → ReLU → Dropout
    (o dropout vem depois da normalização, como recomenda a aula de Regularização)."""
    camadas, k = [], n_entradas
    for largura in ocultas:
        camadas.append(nn.Linear(k, largura))
        if batchnorm:
            camadas.append(nn.BatchNorm1d(largura))
        camadas.append(nn.ReLU())
        if dropout > 0:
            camadas.append(nn.Dropout(dropout))
        k = largura
    camadas.append(nn.Linear(k, 1))  # 1 logit; a sigmoide fica na perda
    if vies_saida is not None:
        nn.init.constant_(camadas[-1].bias, vies_saida)
    return nn.Sequential(*camadas)


def _grupos_weight_decay(modelo, weight_decay):
    """Aplica weight decay só às matrizes de pesos das camadas Linear: vieses e
    parâmetros da BatchNorm ficam de fora (aula de Regularização)."""
    com, sem = [], []
    for modulo in modelo.modules():
        if isinstance(modulo, nn.Linear):
            com.append(modulo.weight)
            sem.append(modulo.bias)
        elif isinstance(modulo, nn.BatchNorm1d):
            sem += [modulo.weight, modulo.bias]
    return [{"params": com, "weight_decay": weight_decay},
            {"params": sem, "weight_decay": 0.0}]


def _probabilidades(modelo, X):
    modelo.eval()
    with torch.no_grad():
        return torch.sigmoid(modelo(X).squeeze(1)).numpy()


def treinar_mlp(X_tr, y_tr, X_val, y_val, semente, ocultas=(8,), lr=1e-3,
                weight_decay=0.0, dropout=0.0, batchnorm=False, vies_saida=None,
                batch_size=32, max_epocas=100, paciencia=10):
    """Treina a MLP e devolve (modelo com os melhores pesos, dicionário de resultados).

    `paciencia=None` desliga a parada antecipada (treina todas as épocas), mas os
    resultados "na melhor época" continuam sendo calculados.
    """
    fixar_semente(semente)
    modelo = construir_mlp(ocultas, X_tr.shape[1], dropout, batchnorm, vies_saida)
    otimizador = torch.optim.AdamW(_grupos_weight_decay(modelo, weight_decay), lr=lr)
    perda_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(semente), num_workers=0)

    hist_treino, hist_val = [], []
    melhor_perda, melhores_pesos, melhor_epoca, sem_melhora = float("inf"), None, 0, 0
    for epoca in range(1, max_epocas + 1):
        modelo.train()
        soma, n = 0.0, 0
        for xb, yb in loader:
            perda = perda_fn(modelo(xb).squeeze(1), yb)
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
            soma, n = soma + perda.item() * len(yb), n + len(yb)
        modelo.eval()
        with torch.no_grad():
            perda_val = perda_fn(modelo(X_val).squeeze(1), y_val).item()
        hist_treino.append(soma / n)
        hist_val.append(perda_val)

        if perda_val < melhor_perda:
            melhor_perda, melhor_epoca, sem_melhora = perda_val, epoca, 0
            melhores_pesos = copy.deepcopy(modelo.state_dict())
        else:
            sem_melhora += 1
            if paciencia is not None and sem_melhora >= paciencia:
                break

    ultima = {"PR-AUC treino (última época)": average_precision_score(
                  y_tr.numpy(), _probabilidades(modelo, X_tr)),
              "PR-AUC validação (última época)": average_precision_score(
                  y_val.numpy(), _probabilidades(modelo, X_val))}
    modelo.load_state_dict(melhores_pesos)
    resultado = {
        "PR-AUC treino": average_precision_score(y_tr.numpy(), _probabilidades(modelo, X_tr)),
        "PR-AUC validação": average_precision_score(y_val.numpy(), _probabilidades(modelo, X_val)),
        "melhor época": melhor_epoca,
        "épocas executadas": len(hist_val),
        "perda treino": hist_treino,
        "perda validação": hist_val,
        **ultima,
    }
    return modelo, resultado
