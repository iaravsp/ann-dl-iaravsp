# Artificial Neural Networks and Deep Learning

Portfólio de entregas da disciplina — Insper, 2026.2.

**Aluna:** Iara Vivian Sousa Pinto (`iaravsp`)
**Repositório:** [iaravsp/ann-dl-iaravsp](https://github.com/iaravsp/ann-dl-iaravsp)

## Entregas

| Exercício | Página | Prazo | Status |
|---|---|---|---|
| Data | [exercises/data](exercises/data/index.md) | 10/09/2026 | em andamento |
| Perceptron | [exercises/perceptron](exercises/perceptron/index.md) | — | não iniciado |
| MLP | [exercises/mlp](exercises/mlp/index.md) | — | não iniciado |
| VAE | [exercises/vae](exercises/vae/index.md) | — | não iniciado |
| Projeto | [projects](projects/index.md) | — | não iniciado |

## Como rodar os códigos

Os scripts de cada exercício ficam em `docs/exercises/<slug>/code/` e são executáveis a partir da raiz do repositório:

``` shell
python -m venv env
source ./env/Scripts/activate     # Windows (Git Bash); no Linux/macOS: ./env/bin/activate
python -m pip install -r requirements.txt
python docs/exercises/data/code/<script>.py
```

As figuras são geradas pelos scripts em `docs/exercises/<slug>/figures/` e versionadas no repositório.
