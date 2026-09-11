# ann-dl-iaravsp

Entregas da disciplina **Artificial Neural Networks and Deep Learning** — Insper, 2026.2.

Site publicado: https://iaravsp.github.io/ann-dl-iaravsp/

## Estrutura

```
docs/
  index.md                     # landing page
  exercises/
    data/
      index.md                 # relatório
      code/                    # scripts (.py)
      figures/                 # figuras geradas pelos scripts (.png)
    perceptron/
    mlp/
    vae/
  projects/
data/
  spaceship-titanic/           # dataset do Kaggle (não versionado por padrão)
mkdocs.yml
requirements.txt
```

## Setup

``` shell
python -m venv env
source ./env/Scripts/activate      # Windows (Git Bash) — Linux/macOS: source ./env/bin/activate
python -m pip install -r requirements.txt --upgrade
```

## Rodar os scripts dos exercícios

Sempre a partir da raiz do repositório:

``` shell
python docs/exercises/data/code/<script>.py
```

Cada script salva suas figuras em `docs/exercises/<slug>/figures/`. As figuras
são versionadas; o relatório as exibe por caminho relativo e puxa o código com
`--8<--`.

## Visualizar o site localmente

``` shell
mkdocs serve -o
```

O deploy para o GitHub Pages é automático via GitHub Actions (`.github/workflows/main.yaml`)
a cada push na `main`.
